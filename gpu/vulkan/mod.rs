//! The model on a GPU, through Vulkan compute: every operation is a
//! hand-written kernel in `kernels/`, compiled to SPIR-V at build time.
//!
//! Operations are recorded into one command buffer as they are called, with
//! a barrier between each, and submitted together the first time something
//! is read back. A forward pass is therefore one submission per batch of
//! tokens, and the GPU never waits on the CPU between kernels.

use std::{error::Error, io::Cursor, slice, sync::Mutex};

use ash::{khr::push_descriptor, vk};

use crate::{Device, Tensor};

/// Size of the staging buffer host memory is copied to and from the device
/// through.
const STAGING: usize = 64 << 20;

/// Most tokens one embedding lookup can take: the size of the tokens buffer.
const MAX_TOKENS: usize = 1024;

/// Most positions the attention kernel's workgroup memory has room for.
pub const MAX_LEN: usize = 4096;

/// A vector of f32 activations in device memory.
pub struct Buffer {
    buf: vk::Buffer,
    len: usize,
    cap: usize,
}

/// A bf16 weight tensor in device memory.
pub struct Weight {
    buf: vk::Buffer,
    shape: Vec<usize>,
}

struct Kernels {
    embed: vk::Pipeline,
    matmul: vk::Pipeline,
    add: vk::Pipeline,
    rmsnorm: vk::Pipeline,
    rope: vk::Pipeline,
    attention: vk::Pipeline,
    silu_mul: vk::Pipeline,
}

/// A host-visible buffer, mapped for as long as the device lives.
struct Mapped {
    buf: vk::Buffer,
    ptr: *mut u8,
}

pub struct Vulkan {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    push: push_descriptor::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    kernels: Kernels,
    memory_types: vk::PhysicalDeviceMemoryProperties,
    staging: Mapped,
    tokens: Mapped,
    max_groups: u32,
    name: String,
    /// Every buffer and its memory, freed when the device is dropped.
    allocations: Mutex<Vec<(vk::Buffer, vk::DeviceMemory)>>,
    /// Whether the command buffer is open with commands not yet submitted.
    recording: Mutex<bool>,
}

// The mapped pointers are only used from whichever thread owns the device.
unsafe impl Send for Vulkan {}

impl Vulkan {
    /// Opens the first GPU Vulkan finds, preferring a discrete one.
    pub fn new() -> Result<Self, Box<dyn Error>> {
        unsafe {
            let entry = ash::Entry::load()?;
            let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
            let instance = entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)?;

            let physical = instance
                .enumerate_physical_devices()?
                .into_iter()
                .min_by_key(|&pd| match instance.get_physical_device_properties(pd).device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                    _ => 3,
                })
                .ok_or("no Vulkan device")?;
            let props = instance.get_physical_device_properties(physical);
            let name = props.device_name_as_c_str()?.to_string_lossy().into_owned();
            if props.api_version < vk::API_VERSION_1_1 {
                return Err(format!("{name} does not support Vulkan 1.1").into());
            }
            let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
            instance.get_physical_device_properties2(physical, &mut props2);
            let needed = vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::ARITHMETIC;
            if !subgroup.supported_operations.contains(needed) || subgroup.subgroup_size < 16 {
                return Err(format!("{name} lacks the subgroup operations the kernels use").into());
            }
            let extensions = instance.enumerate_device_extension_properties(physical)?;
            if !extensions.iter().any(|ext| ext.extension_name_as_c_str() == Ok(push_descriptor::NAME)) {
                return Err(format!("{name} lacks {}", push_descriptor::NAME.to_string_lossy()).into());
            }

            let family = instance
                .get_physical_device_queue_family_properties(physical)
                .iter()
                .position(|family| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .ok_or("no compute queue")? as u32;
            let queue_info = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&[1.0])];
            let extension_names = [push_descriptor::NAME.as_ptr()];
            let device_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_info)
                .enabled_extension_names(&extension_names);
            let device = instance.create_device(physical, &device_info, None)?;
            let push = push_descriptor::Device::new(&instance, &device);
            let queue = device.get_device_queue(family, 0);

            let pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            let cmd = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;

            // Every kernel binds up to four storage buffers, pushed with each
            // dispatch, and takes its sizes as push constants.
            let bindings: Vec<_> = (0..4)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                    .bindings(&bindings),
                None,
            )?;
            let set_layouts = [set_layout];
            let push_range = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(32)];
            let layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push_range),
                None,
            )?;

            let kernel = |spv: &[u8]| -> Result<vk::Pipeline, Box<dyn Error>> {
                let words = ash::util::read_spv(&mut Cursor::new(spv))?;
                let module = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?;
                let stage = vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(module)
                    .name(c"main");
                let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
                let pipeline = device
                    .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
                    .map_err(|(_, e)| e)?[0];
                device.destroy_shader_module(module, None);
                Ok(pipeline)
            };
            macro_rules! spv {
                ($name:literal) => {
                    kernel(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))?
                };
            }
            let memory_types = instance.get_physical_device_memory_properties(physical);
            let kernels = Kernels {
                embed: spv!("embed"),
                matmul: spv!("matmul"),
                add: spv!("add"),
                rmsnorm: spv!("rmsnorm"),
                rope: spv!("rope"),
                attention: spv!("attention"),
                silu_mul: spv!("silu_mul"),
            };

            let mut gpu = Self {
                _entry: entry,
                instance,
                device,
                push,
                queue,
                pool,
                cmd,
                fence,
                set_layout,
                layout,
                kernels,
                memory_types,
                staging: Mapped {
                    buf: vk::Buffer::null(),
                    ptr: std::ptr::null_mut(),
                },
                tokens: Mapped {
                    buf: vk::Buffer::null(),
                    ptr: std::ptr::null_mut(),
                },
                max_groups: props.limits.max_compute_work_group_count[0],
                name,
                allocations: Mutex::new(Vec::new()),
                recording: Mutex::new(false),
            };
            gpu.staging = gpu.map(STAGING)?;
            gpu.tokens = gpu.map(MAX_TOKENS * 4)?;
            Ok(gpu)
        }
    }

    /// Name of the GPU.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Creates a buffer of `bytes` in memory of the given kind.
    fn buffer(&self, bytes: usize, flags: vk::MemoryPropertyFlags) -> Result<(vk::Buffer, vk::DeviceMemory), Box<dyn Error>> {
        unsafe {
            let info = vk::BufferCreateInfo::default()
                .size(bytes.max(4) as u64)
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buf = self.device.create_buffer(&info, None)?;
            let req = self.device.get_buffer_memory_requirements(buf);
            let types = &self.memory_types.memory_types[..self.memory_types.memory_type_count as usize];
            let index = types
                .iter()
                .enumerate()
                .position(|(i, t)| req.memory_type_bits & (1 << i) != 0 && t.property_flags.contains(flags))
                .ok_or("no suitable memory type")? as u32;
            let mem = self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(index),
                None,
            )?;
            self.device.bind_buffer_memory(buf, mem, 0)?;
            self.allocations.lock().unwrap().push((buf, mem));
            Ok((buf, mem))
        }
    }

    /// Creates a host-visible buffer and maps it.
    fn map(&self, bytes: usize) -> Result<Mapped, Box<dyn Error>> {
        let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let (buf, mem) = self.buffer(bytes, flags)?;
        let ptr = unsafe { self.device.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())? };
        Ok(Mapped { buf, ptr: ptr.cast() })
    }

    /// Creates a buffer in device memory.
    fn device_buffer(&self, bytes: usize) -> vk::Buffer {
        self.buffer(bytes, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .expect("out of GPU memory")
            .0
    }

    /// The command buffer, open for recording.
    fn cmd(&self) -> vk::CommandBuffer {
        let mut recording = self.recording.lock().unwrap();
        if !*recording {
            unsafe {
                self.device
                    .reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())
                    .unwrap();
                self.device
                    .begin_command_buffer(
                        self.cmd,
                        &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .unwrap();
            }
            *recording = true;
        }
        self.cmd
    }

    /// Makes everything recorded so far visible to everything recorded next.
    fn barrier(&self, cmd: vk::CommandBuffer) {
        let stages = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE,
            );
        unsafe {
            self.device
                .cmd_pipeline_barrier(cmd, stages, stages, vk::DependencyFlags::empty(), &[barrier], &[], &[]);
        }
    }

    /// Submits everything recorded and waits for it to finish.
    fn flush(&self) {
        let mut recording = self.recording.lock().unwrap();
        if !*recording {
            return;
        }
        unsafe {
            self.device.end_command_buffer(self.cmd).unwrap();
            let cmds = [self.cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            self.device.queue_submit(self.queue, &[submit], self.fence).unwrap();
            self.device.wait_for_fences(&[self.fence], true, u64::MAX).unwrap();
            self.device.reset_fences(&[self.fence]).unwrap();
        }
        *recording = false;
    }

    /// Records a kernel over `groups` workgroups, bound to `buffers` in
    /// order, with `params` as its push constants.
    fn dispatch<P: Copy>(&self, pipeline: vk::Pipeline, buffers: &[vk::Buffer], params: &P, groups: (u32, u32)) {
        let cmd = self.cmd();
        let infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers
            .iter()
            .map(|&buf| [vk::DescriptorBufferInfo::default().buffer(buf).offset(0).range(vk::WHOLE_SIZE)])
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        unsafe {
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.push
                .cmd_push_descriptor_set(cmd, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &writes);
            self.device
                .cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, bytes(params));
            self.device.cmd_dispatch(cmd, groups.0, groups.1, 1);
        }
        self.barrier(cmd);
    }

    /// Workgroups for `count` items of work in a one-dimensional kernel.
    fn groups(&self, count: usize, per_group: usize) -> (u32, u32) {
        let groups = count.div_ceil(per_group);
        assert!(groups <= self.max_groups as usize, "{groups} workgroups is more than the GPU allows");
        (groups as u32, 1)
    }

    /// Copies bytes from the host into a device buffer, through staging.
    fn upload_bytes(&self, dst: vk::Buffer, offset: usize, data: &[u8]) {
        for (i, chunk) in data.chunks(STAGING).enumerate() {
            self.flush();
            unsafe {
                std::ptr::copy_nonoverlapping(chunk.as_ptr(), self.staging.ptr, chunk.len());
                let cmd = self.cmd();
                let region = vk::BufferCopy::default()
                    .dst_offset((offset + i * STAGING) as u64)
                    .size(chunk.len() as u64);
                self.device.cmd_copy_buffer(cmd, self.staging.buf, dst, &[region]);
            }
            self.barrier(self.cmd);
            self.flush();
        }
    }

    /// Copies bytes from a device buffer to the host, through staging.
    fn download_bytes(&self, src: vk::Buffer, data: &mut [u8]) {
        for (i, chunk) in data.chunks_mut(STAGING).enumerate() {
            self.flush();
            unsafe {
                let cmd = self.cmd();
                let region = vk::BufferCopy::default()
                    .src_offset((i * STAGING) as u64)
                    .size(chunk.len() as u64);
                self.device.cmd_copy_buffer(cmd, src, self.staging.buf, &[region]);
                self.barrier(cmd);
                self.flush();
                std::ptr::copy_nonoverlapping(self.staging.ptr, chunk.as_mut_ptr(), chunk.len());
            }
        }
    }
}

impl Drop for Vulkan {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            let k = &self.kernels;
            for pipeline in [k.embed, k.matmul, k.add, k.rmsnorm, k.rope, k.attention, k.silu_mul] {
                self.device.destroy_pipeline(pipeline, None);
            }
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_descriptor_set_layout(self.set_layout, None);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.pool, None);
            for (buf, mem) in self.allocations.lock().unwrap().drain(..) {
                self.device.destroy_buffer(buf, None);
                self.device.free_memory(mem, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MatmulParams {
    rows: u32,
    cols: u32,
    n: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct EmbedParams {
    dim: u32,
    n: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RmsnormParams {
    dim: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RopeParams {
    n_heads: u32,
    head_dim: u32,
    pos: u32,
    n: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AttentionParams {
    n_heads: u32,
    head_dim: u32,
    n_kv_heads: u32,
    pos: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LenParams {
    len: u32,
}

impl Device for Vulkan {
    type Buffer = Buffer;
    type Weight = Weight;

    fn upload(&self, tensor: Tensor) -> Weight {
        assert!(tensor.data.len().is_multiple_of(8), "weights must come in multiples of eight");
        let buf = self.device_buffer(tensor.data.len() * 2);
        // bf16 bits, two to a 32-bit word, in memory order: the kernels take
        // the low half of a word as the first weight, as little-endian does.
        let data = unsafe { slice::from_raw_parts(tensor.data.as_ptr().cast::<u8>(), tensor.data.len() * 2) };
        self.upload_bytes(buf, 0, data);
        Weight {
            buf,
            shape: tensor.shape,
        }
    }

    fn alloc(&self, len: usize) -> Buffer {
        let buf = self.device_buffer(len * 4);
        let cmd = self.cmd();
        unsafe { self.device.cmd_fill_buffer(cmd, buf, 0, vk::WHOLE_SIZE, 0) };
        self.barrier(cmd);
        Buffer { buf, len, cap: len }
    }

    fn resize(&self, buf: &mut Buffer, len: usize) {
        assert!(len <= buf.cap, "a buffer of {} activations can't hold {len}", buf.cap);
        buf.len = len;
    }

    fn read(&self, buf: &Buffer) -> Vec<f32> {
        let mut out = vec![0.0f32; buf.len];
        let bytes = unsafe { slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), buf.len * 4) };
        self.download_bytes(buf.buf, bytes);
        out
    }

    fn write(&self, buf: &mut Buffer, data: &[f32]) {
        assert_eq!(buf.len, data.len());
        let bytes = unsafe { slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
        self.upload_bytes(buf.buf, 0, bytes);
    }

    fn copy(&self, dst: &mut Buffer, dst_offset: usize, src: &Buffer, src_offset: usize, len: usize) {
        assert!(dst_offset + len <= dst.len && src_offset + len <= src.len);
        let cmd = self.cmd();
        let region = vk::BufferCopy::default()
            .src_offset((src_offset * 4) as u64)
            .dst_offset((dst_offset * 4) as u64)
            .size((len * 4) as u64);
        unsafe { self.device.cmd_copy_buffer(cmd, src.buf, dst.buf, &[region]) };
        self.barrier(cmd);
    }

    fn embed(&self, out: &mut Buffer, table: &Weight, tokens: &[u32]) {
        let dim = table.shape[1];
        assert_eq!(out.len, tokens.len() * dim);
        assert!(tokens.len() <= MAX_TOKENS);
        // The tokens buffer is read by the pending commands, which must
        // finish before it is written over.
        self.flush();
        unsafe { std::ptr::copy_nonoverlapping(tokens.as_ptr(), self.tokens.ptr.cast::<u32>(), tokens.len()) };
        let params = EmbedParams {
            dim: dim as u32,
            n: tokens.len() as u32,
        };
        let groups = self.groups(tokens.len() * dim / 2, 256);
        self.dispatch(self.kernels.embed, &[out.buf, table.buf, self.tokens.buf], &params, groups);
    }

    fn matmul(&self, out: &mut Buffer, w: &Weight, x: &Buffer) {
        let (rows, cols) = (w.shape[0], w.shape[1]);
        let n = x.len / cols;
        assert_eq!(x.len, n * cols);
        assert_eq!(out.len, n * rows);
        assert!(cols % 8 == 0);
        // One workgroup per row, in a grid as wide as the GPU allows.
        let width = rows.min(self.max_groups as usize);
        let groups = (width as u32, rows.div_ceil(width) as u32);
        let params = MatmulParams {
            rows: rows as u32,
            cols: cols as u32,
            n: n as u32,
            stride: width as u32,
        };
        self.dispatch(self.kernels.matmul, &[out.buf, w.buf, x.buf], &params, groups);
    }

    fn add(&self, x: &mut Buffer, y: &Buffer) {
        assert_eq!(x.len, y.len);
        let params = LenParams { len: x.len as u32 };
        let groups = self.groups(x.len, 256);
        self.dispatch(self.kernels.add, &[x.buf, y.buf], &params, groups);
    }

    fn rmsnorm(&self, x: &mut Buffer, weight: &Weight, eps: f32) {
        let dim = weight.shape[0];
        assert_eq!(x.len % dim, 0);
        let params = RmsnormParams { dim: dim as u32, eps };
        let groups = self.groups(x.len / dim, 1);
        self.dispatch(self.kernels.rmsnorm, &[x.buf, weight.buf], &params, groups);
    }

    fn rope(&self, x: &mut Buffer, table: &Buffer, pos: usize, n_heads: usize, head_dim: usize) {
        let n = x.len / (n_heads * head_dim);
        assert_eq!(x.len, n * n_heads * head_dim);
        let params = RopeParams {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            pos: pos as u32,
            n: n as u32,
        };
        let groups = self.groups(n * n_heads * head_dim / 2, 256);
        self.dispatch(self.kernels.rope, &[x.buf, table.buf], &params, groups);
    }

    fn attention(
        &self,
        out: &mut Buffer,
        q: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        pos: usize,
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
    ) {
        let n = q.len / (n_heads * head_dim);
        assert_eq!(q.len, n * n_heads * head_dim);
        assert_eq!(out.len, q.len);
        assert!(head_dim.is_multiple_of(4) && head_dim <= 128);
        assert!(pos + n <= MAX_LEN, "attention over more than {MAX_LEN} positions");
        let params = AttentionParams {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            n_kv_heads: n_kv_heads as u32,
            pos: pos as u32,
        };
        let groups = self.groups(n * n_heads, 1);
        self.dispatch(
            self.kernels.attention,
            &[out.buf, q.buf, k_cache.buf, v_cache.buf],
            &params,
            groups,
        );
    }

    fn silu_mul(&self, gate: &mut Buffer, up: &Buffer) {
        assert_eq!(gate.len, up.len);
        let params = LenParams { len: gate.len as u32 };
        let groups = self.groups(gate.len, 256);
        self.dispatch(self.kernels.silu_mul, &[gate.buf, up.buf], &params, groups);
    }
}

/// The bytes of a plain struct of 32-bit fields, as push constants.
fn bytes<P: Copy>(params: &P) -> &[u8] {
    unsafe { slice::from_raw_parts((params as *const P).cast::<u8>(), size_of::<P>()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cpu;

    /// Deterministic pseudo-random values in [-1, 1).
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }

        fn floats(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next()).collect()
        }

        fn tensor(&mut self, shape: &[usize]) -> Tensor {
            let n = shape.iter().product();
            Tensor {
                shape: shape.to_vec(),
                data: (0..n).map(|_| (self.next().to_bits() >> 16) as u16).collect(),
            }
        }
    }

    fn gpu() -> Option<Vulkan> {
        match Vulkan::new() {
            Ok(gpu) => Some(gpu),
            Err(e) => {
                eprintln!("skipping: {e}");
                None
            }
        }
    }

    fn bf16_tensor(t: &Tensor) -> Tensor {
        Tensor {
            shape: t.shape.clone(),
            data: t.data.clone(),
        }
    }

    fn close(a: &[f32], b: &[f32], tolerance: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (a, b)) in a.iter().zip(b).enumerate() {
            assert!((a - b).abs() <= tolerance * (1.0 + a.abs().max(b.abs())), "element {i}: {a} vs {b}");
        }
    }

    fn buffer(gpu: &Vulkan, data: &[f32]) -> Buffer {
        let mut buf = gpu.alloc(data.len());
        gpu.write(&mut buf, data);
        buf
    }

    #[test]
    fn write_and_read_round_trip() {
        let Some(gpu) = gpu() else { return };
        let data = Rng(1).floats(100_003);
        let buf = buffer(&gpu, &data);
        assert_eq!(gpu.read(&buf), data);
    }

    #[test]
    fn alloc_is_zeroed_and_copy_moves_ranges() {
        let Some(gpu) = gpu() else { return };
        let zero = gpu.alloc(1000);
        assert!(gpu.read(&zero).iter().all(|&v| v == 0.0));
        let data = Rng(2).floats(50);
        let src = buffer(&gpu, &data);
        let mut dst = gpu.alloc(100);
        gpu.copy(&mut dst, 30, &src, 10, 20);
        let mut want = vec![0.0; 100];
        Cpu.copy(&mut want, 30, &data, 10, 20);
        assert_eq!(gpu.read(&dst), want);
    }

    #[test]
    fn embed_matches_cpu() {
        let Some(gpu) = gpu() else { return };
        let table = Rng(3).tensor(&[50, 24]);
        let tokens = [3, 49, 0, 17];
        let mut want = vec![0.0; tokens.len() * 24];
        Cpu.embed(&mut want, &bf16_tensor(&table), &tokens);
        let weight = gpu.upload(table);
        let mut out = gpu.alloc(tokens.len() * 24);
        gpu.embed(&mut out, &weight, &tokens);
        assert_eq!(gpu.read(&out), want);
    }

    #[test]
    fn matmul_matches_cpu() {
        let Some(gpu) = gpu() else { return };
        let mut rng = Rng(4);
        for (rows, cols, n) in [(1, 8, 1), (200, 192, 1), (77, 1032, 3), (70_000, 8, 2)] {
            let w = rng.tensor(&[rows, cols]);
            let x = rng.floats(n * cols);
            let mut want = vec![0.0; n * rows];
            Cpu.matmul(&mut want, &bf16_tensor(&w), &x);
            let weight = gpu.upload(w);
            let x = buffer(&gpu, &x);
            let mut out = gpu.alloc(n * rows);
            gpu.matmul(&mut out, &weight, &x);
            close(&gpu.read(&out), &want, 1e-4);
        }
    }

    #[test]
    fn rmsnorm_matches_cpu() {
        let Some(gpu) = gpu() else { return };
        let mut rng = Rng(5);
        for (dim, rows) in [(128, 5), (1024, 3), (8, 1)] {
            let w = rng.tensor(&[dim]);
            let x = rng.floats(dim * rows);
            let mut want = x.clone();
            Cpu.rmsnorm(&mut want, &bf16_tensor(&w), 1e-6);
            let weight = gpu.upload(w);
            let mut buf = buffer(&gpu, &x);
            gpu.rmsnorm(&mut buf, &weight, 1e-6);
            close(&gpu.read(&buf), &want, 1e-5);
        }
    }

    #[test]
    fn rope_matches_cpu() {
        let Some(gpu) = gpu() else { return };
        let mut rng = Rng(6);
        let (n_heads, head_dim, n, pos) = (4, 16, 3, 7);
        let table = rng.floats((pos + n) * head_dim);
        let x = rng.floats(n * n_heads * head_dim);
        let mut want = x.clone();
        Cpu.rope(&mut want, &table, pos, n_heads, head_dim);
        let table = buffer(&gpu, &table);
        let mut buf = buffer(&gpu, &x);
        gpu.rope(&mut buf, &table, pos, n_heads, head_dim);
        close(&gpu.read(&buf), &want, 1e-5);
    }

    #[test]
    fn attention_matches_cpu() {
        let Some(gpu) = gpu() else { return };
        let mut rng = Rng(7);
        for (n_heads, head_dim, n_kv_heads, n, pos) in [(4, 8, 2, 3, 5), (16, 128, 8, 2, 300), (2, 128, 1, 1, 0)] {
            let kv_dim = n_kv_heads * head_dim;
            let q = rng.floats(n * n_heads * head_dim);
            let k_cache = rng.floats((pos + n) * kv_dim);
            let v_cache = rng.floats((pos + n) * kv_dim);
            let mut want = vec![0.0; q.len()];
            Cpu.attention(&mut want, &q, &k_cache, &v_cache, pos, n_heads, head_dim, n_kv_heads);
            let (q, k, v) = (buffer(&gpu, &q), buffer(&gpu, &k_cache), buffer(&gpu, &v_cache));
            let mut out = gpu.alloc(want.len());
            gpu.attention(&mut out, &q, &k, &v, pos, n_heads, head_dim, n_kv_heads);
            close(&gpu.read(&out), &want, 1e-4);
        }
    }

    #[test]
    fn elementwise_match_cpu() {
        let Some(gpu) = gpu() else { return };
        let mut rng = Rng(8);
        let a = rng.floats(3001);
        let b = rng.floats(3001);
        let mut want = a.clone();
        Cpu.silu_mul(&mut want, &b);
        let mut gate = buffer(&gpu, &a);
        let up = buffer(&gpu, &b);
        gpu.silu_mul(&mut gate, &up);
        close(&gpu.read(&gate), &want, 1e-6);

        let mut want = a.clone();
        Cpu.add(&mut want, &b);
        let mut x = buffer(&gpu, &a);
        gpu.add(&mut x, &up);
        close(&gpu.read(&x), &want, 1e-6);
    }

    #[test]
    fn resize_keeps_capacity() {
        let Some(gpu) = gpu() else { return };
        let mut buf = gpu.alloc(64);
        gpu.resize(&mut buf, 16);
        assert_eq!(gpu.read(&buf).len(), 16);
        gpu.resize(&mut buf, 64);
        assert_eq!(gpu.read(&buf).len(), 64);
    }
}
