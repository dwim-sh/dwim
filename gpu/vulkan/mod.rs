//! The model on a GPU, through Vulkan compute: every operation is a
//! hand-written kernel in `kernels/`, compiled to SPIR-V at build time.
//!
//! Operations are recorded into a command buffer as they are called, with a
//! barrier between each, and submitted every so many of them without
//! waiting, from a ring of command buffers, so that the GPU works on the
//! first kernels of a forward pass while the CPU records the rest. Reading
//! a buffer back waits for everything submitted.

use std::{error::Error, io::Cursor, slice, sync::Mutex};

use ash::{
    ext::memory_budget,
    khr::{push_descriptor, shader_float16_int8},
    vk,
};

use crate::{
    gpu::{CACHE_BLOCK, CONV_KERNEL, Device, Tensor, cache_bytes, quant},
    sys::memory,
};

/// Size of each of the two staging buffers host memory is copied to and
/// from the device through: uploads alternate between them, so that one is
/// filled while the device copies from the other.
const STAGING: usize = 64 << 20;

/// Tokens a workgroup of the bf16 and block-quantized matmul kernels takes
/// at once.
const BF16_TOKENS: usize = 8;

/// Command buffers in the ring, and kernels recorded into one before it is
/// submitted.
const RING: usize = 4;
const SUBMIT_EVERY: usize = 128;

/// Multiply-adds of matrix products and attention recorded into a command
/// buffer before it is submitted, however few kernels that is: a driver
/// abandons a submission that runs for seconds as hung, and resets the
/// GPU under it. A batch through a model's larger matrices is this many
/// in a few kernels, and one through its experts in one, which is
/// recorded as several.
const SUBMIT_WORK: usize = 1 << 30;

/// Positions one workgroup of the attention kernel takes.
const CHUNK: usize = 128;

/// Most floats of attention partials one dispatch writes, one chunk's
/// maximum, sum, and weighted values per head of each token: the size of
/// the partials buffer. Longer batches are dispatched a few tokens at a
/// time.
const PARTIALS: usize = 16 << 20;

/// Most storage buffers a kernel binds, and the bytes of the push
/// constants it takes its sizes as.
const BINDINGS: usize = 10;
const PUSH_CONSTANTS: usize = 64;

/// A vector of f32 activations in device memory.
pub struct Buffer {
    buf: vk::Buffer,
    len: usize,
    cap: usize,
}

/// A cache of activations in device memory, in blocks of 8-bit integers
/// four to a 32-bit word, and the blocks' f16 scales two to a word.
pub struct Cache {
    quants: vk::Buffer,
    scales: vk::Buffer,
    len: usize,
}

/// A weight matrix in device memory: its weights, and for the
/// block-quantized formats their scales in a buffer of their own.
pub struct Weight {
    buf: vk::Buffer,
    scales: vk::Buffer,
    shape: Vec<usize>,
    format: Format,
}

/// The formats weights come in, as [`Tensor`] has them.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    Bf16,
    Q4_0,
    Q8_0,
    Nvfp4,
}

struct Kernels {
    matmul_bf16: vk::Pipeline,
    matmul_q4_0: vk::Pipeline,
    matmul_q8_0: vk::Pipeline,
    add: vk::Pipeline,
    rmsnorm: vk::Pipeline,
    l2norm: vk::Pipeline,
    rope: vk::Pipeline,
    attention: vk::Pipeline,
    attention_combine: vk::Pipeline,
    silu_mul: vk::Pipeline,
    sigmoid_mul: vk::Pipeline,
    store_q8: vk::Pipeline,
    conv: vk::Pipeline,
    delta_net: vk::Pipeline,
    expert_matmul_nvfp4: vk::Pipeline,
    expert_matmul_q8_0: vk::Pipeline,
    mix: vk::Pipeline,
    gate: vk::Pipeline,
    gated_update: vk::Pipeline,
    short_conv: vk::Pipeline,
    short_conv_state: vk::Pipeline,
    clip: vk::Pipeline,
    scale_rows: vk::Pipeline,
    attention_rel: vk::Pipeline,
    attention_rel_combine: vk::Pipeline,
    route: vk::Pipeline,
    situ: vk::Pipeline,
    weighted_sum: vk::Pipeline,
}

impl Kernels {
    /// The bindings a kernel writes, as bits: the first, and for a few
    /// kernels a partials, state, scales, or cache binding too.
    fn written(&self, pipeline: vk::Pipeline) -> u32 {
        if pipeline == self.attention || pipeline == self.attention_rel {
            0b100_0001
        } else if pipeline == self.store_q8 || pipeline == self.route {
            0b11
        } else if pipeline == self.delta_net {
            0b100_0001
        } else if pipeline == self.conv {
            0b1111
        } else if pipeline == self.short_conv {
            0b101
        } else {
            0b1
        }
    }

    /// Whether a kernel is a matrix product, which waits for the kernel
    /// before it to have reached the end of the pipe when a buffer it binds
    /// was written since the last such wait: see `Vulkan::events`.
    fn matmul(&self, pipeline: vk::Pipeline) -> bool {
        [
            self.matmul_bf16,
            self.matmul_q4_0,
            self.matmul_q8_0,
            self.expert_matmul_nvfp4,
            self.expert_matmul_q8_0,
            self.gate,
        ]
        .contains(&pipeline)
    }

    fn all(&self) -> [vk::Pipeline; 28] {
        [
            self.matmul_bf16,
            self.matmul_q4_0,
            self.matmul_q8_0,
            self.add,
            self.rmsnorm,
            self.l2norm,
            self.rope,
            self.attention,
            self.attention_combine,
            self.silu_mul,
            self.sigmoid_mul,
            self.store_q8,
            self.conv,
            self.delta_net,
            self.expert_matmul_nvfp4,
            self.expert_matmul_q8_0,
            self.mix,
            self.gate,
            self.gated_update,
            self.short_conv,
            self.short_conv_state,
            self.clip,
            self.scale_rows,
            self.attention_rel,
            self.attention_rel_combine,
            self.route,
            self.situ,
            self.weighted_sum,
        ]
    }
}

/// A host-visible buffer, mapped for as long as the device lives.
struct Mapped {
    buf: vk::Buffer,
    ptr: *mut u8,
}

pub struct Vulkan {
    _entry: ash::Entry,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    device: ash::Device,
    push: push_descriptor::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    cmds: Vec<vk::CommandBuffer>,
    fences: Vec<vk::Fence>,
    /// Set, waited for on the GPU, and reset before a matrix product that
    /// binds a buffer written since the last such wait: everything recorded
    /// before it has then reached the end of the pipe before the product
    /// starts. A pipeline barrier between two kernels is not enough on
    /// Navi 10 under RADV: the model's kernels on fixed data gave a
    /// different result about once in 500 steps with a barrier after every
    /// kernel, whatever the queue, the ring depth, or the number of kernels
    /// a submission, and none in 14,000 steps with a fence after every
    /// kernel, and the kernel that went wrong was a single-token matmul; the event is the fence's wait for the end of the pipe
    /// without the round trip to the CPU, before the kernels that were seen
    /// to need it. The reset comes after the wait has passed, when the
    /// set's write is known to have landed, and is the command processor's
    /// own write, so that the next wait never sees an earlier set.
    event: vk::Event,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    kernels: Kernels,
    memory_types: vk::PhysicalDeviceMemoryProperties,
    /// Whether the driver says how much of each heap is free, with
    /// `VK_EXT_memory_budget`.
    budget: bool,
    /// Whether the GPU is integrated, its memory the host's.
    integrated: bool,
    staging: [Mapped; 2],
    partials: vk::Buffer,
    /// A batch's activations as half floats, for the batch and tile kernels.
    max_groups: u32,
    name: String,
    /// Every buffer and its memory, freed when the device is dropped.
    allocations: Mutex<Vec<(vk::Buffer, vk::DeviceMemory)>>,
    ring: Mutex<Ring>,
}

/// Where recording and submission stand in the ring of command buffers.
struct Ring {
    /// The command buffer being recorded into, or to record into next.
    current: usize,
    /// Whether it is open with commands not yet submitted.
    recording: bool,
    /// Which command buffers are submitted and not yet waited for.
    in_flight: [bool; RING],
    /// Kernels recorded into the current command buffer, and the
    /// multiply-adds of the ones that are counted.
    recorded: usize,
    work: usize,
    /// Buffers written since the last wait, by kernels and copies.
    written: Vec<vk::Buffer>,
    /// The command buffer of the ring whose submission reads each staging
    /// buffer, if one may still be running, and which buffer the next
    /// upload fills.
    staging_slot: [Option<usize>; 2],
    staging_next: usize,
}

// The mapped pointers are only used from whichever thread owns the device.
unsafe impl Send for Vulkan {}

impl Vulkan {
    /// Opens the first GPU Vulkan finds, preferring a discrete one.
    pub fn new() -> Result<Self, Box<dyn Error>> {
        unsafe {
            let entry = ash::Entry::load()?;
            let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )?;

            let physical = instance
                .enumerate_physical_devices()?
                .into_iter()
                .min_by_key(
                    |&pd| match instance.get_physical_device_properties(pd).device_type {
                        vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                        _ => 3,
                    },
                )
                .ok_or("no Vulkan device")?;
            let props = instance.get_physical_device_properties(physical);
            let name = props.device_name_as_c_str()?.to_string_lossy().into_owned();
            if props.api_version < vk::API_VERSION_1_1 {
                return Err(format!("{name} does not support Vulkan 1.1").into());
            }
            let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
            instance.get_physical_device_properties2(physical, &mut props2);
            let needed = vk::SubgroupFeatureFlags::BASIC
                | vk::SubgroupFeatureFlags::ARITHMETIC
                | vk::SubgroupFeatureFlags::SHUFFLE;
            if !subgroup.supported_operations.contains(needed) || subgroup.subgroup_size < 16 {
                return Err(format!("{name} lacks the subgroup operations the kernels use").into());
            }
            let extensions = instance.enumerate_device_extension_properties(physical)?;
            let budget = extensions
                .iter()
                .any(|ext| ext.extension_name_as_c_str() == Ok(memory_budget::NAME));
            for needed in [push_descriptor::NAME, shader_float16_int8::NAME] {
                if !extensions
                    .iter()
                    .any(|ext| ext.extension_name_as_c_str() == Ok(needed))
                {
                    return Err(format!("{name} lacks {}", needed.to_string_lossy()).into());
                }
            }
            let mut float16 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
            let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut float16);
            instance.get_physical_device_features2(physical, &mut features);
            if float16.shader_float16 == 0 {
                return Err(
                    format!("{name} lacks 16-bit float arithmetic, which the kernels use").into(),
                );
            }

            // A compute-only queue family when there is one: the graphics
            // family is the display's, on the same hardware ring, so a
            // kernel that hangs there takes the display with it and gets the
            // graphics ring's shorter timeout, and its commands wait behind
            // the compositor's.
            let families = instance.get_physical_device_queue_family_properties(physical);
            let compute = |flags: vk::QueueFlags| {
                families.iter().position(|family| {
                    family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                        && !family.queue_flags.intersects(flags)
                })
            };
            let family = compute(vk::QueueFlags::GRAPHICS)
                .or_else(|| compute(vk::QueueFlags::empty()))
                .ok_or("no compute queue")? as u32;
            let queue_info = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&[1.0])];
            let extension_names = [
                push_descriptor::NAME.as_ptr(),
                shader_float16_int8::NAME.as_ptr(),
            ];
            let mut float16 =
                vk::PhysicalDeviceShaderFloat16Int8Features::default().shader_float16(true);
            let device_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_info)
                .enabled_extension_names(&extension_names)
                .push_next(&mut float16);
            let device = instance.create_device(physical, &device_info, None)?;
            let push = push_descriptor::Device::new(&instance, &device);
            let queue = device.get_device_queue(family, 0);

            let pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            let cmds = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(RING as u32),
            )?;
            let mut fences = Vec::with_capacity(RING);
            for _ in 0..RING {
                fences.push(device.create_fence(&vk::FenceCreateInfo::default(), None)?);
            }
            let event = device.create_event(&vk::EventCreateInfo::default(), None)?;

            // Every kernel binds up to `BINDINGS` storage buffers, pushed
            // with each dispatch, and takes its sizes as push constants.
            let bindings: Vec<_> = (0..BINDINGS as u32)
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
                .size(PUSH_CONSTANTS as u32)];
            let layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push_range),
                None,
            )?;

            let kernel = |spv: &[u8]| -> Result<vk::Pipeline, Box<dyn Error>> {
                let words = ash::util::read_spv(&mut Cursor::new(spv))?;
                let module = device.create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&words),
                    None,
                )?;
                let stage = vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(module)
                    .name(c"main");
                let info = vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(layout);
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
                matmul_bf16: spv!("matmul_bf16"),
                matmul_q4_0: spv!("matmul_q4_0"),
                matmul_q8_0: spv!("matmul_q8_0"),
                add: spv!("add"),
                rmsnorm: spv!("rmsnorm"),
                l2norm: spv!("l2norm"),
                rope: spv!("rope"),
                attention: spv!("attention"),
                attention_combine: spv!("attention_combine"),
                silu_mul: spv!("silu_mul"),
                sigmoid_mul: spv!("sigmoid_mul"),
                store_q8: spv!("store_q8"),
                conv: spv!("conv"),
                delta_net: spv!("delta_net"),
                expert_matmul_nvfp4: spv!("expert_matmul_nvfp4"),
                expert_matmul_q8_0: spv!("expert_matmul_q8_0"),
                mix: spv!("mix"),
                gate: spv!("gate"),
                gated_update: spv!("gated_update"),
                short_conv: spv!("short_conv"),
                short_conv_state: spv!("short_conv_state"),
                clip: spv!("clip"),
                scale_rows: spv!("scale_rows"),
                attention_rel: spv!("attention_rel"),
                attention_rel_combine: spv!("attention_rel_combine"),
                route: spv!("route"),
                situ: spv!("situ"),
                weighted_sum: spv!("weighted_sum"),
            };

            let mut gpu = Self {
                _entry: entry,
                instance,
                physical,
                device,
                push,
                queue,
                pool,
                cmds,
                fences,
                event,
                set_layout,
                layout,
                kernels,
                memory_types,
                budget,
                integrated: props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU,
                staging: [
                    Mapped {
                        buf: vk::Buffer::null(),
                        ptr: std::ptr::null_mut(),
                    },
                    Mapped {
                        buf: vk::Buffer::null(),
                        ptr: std::ptr::null_mut(),
                    },
                ],
                partials: vk::Buffer::null(),
                max_groups: props.limits.max_compute_work_group_count[0],
                name,
                allocations: Mutex::new(Vec::new()),
                ring: Mutex::new(Ring {
                    current: 0,
                    recording: false,
                    in_flight: [false; RING],
                    recorded: 0,
                    work: 0,
                    written: Vec::new(),
                    staging_slot: [None; 2],
                    staging_next: 0,
                }),
            };
            gpu.staging = [gpu.map(STAGING)?, gpu.map(STAGING)?];
            gpu.partials = gpu
                .buffer(PARTIALS * 4, vk::MemoryPropertyFlags::DEVICE_LOCAL)?
                .0;
            Ok(gpu)
        }
    }

    /// Creates a buffer of `bytes` in memory of the given kind.
    fn buffer(
        &self,
        bytes: usize,
        flags: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), Box<dyn Error>> {
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
            let types =
                &self.memory_types.memory_types[..self.memory_types.memory_type_count as usize];
            let index = types
                .iter()
                .enumerate()
                .position(|(i, t)| {
                    req.memory_type_bits & (1 << i) != 0 && t.property_flags.contains(flags)
                })
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
        let ptr = unsafe {
            self.device
                .map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?
        };
        Ok(Mapped {
            buf,
            ptr: ptr.cast(),
        })
    }

    /// Creates a buffer in device memory.
    fn device_buffer(&self, bytes: usize) -> vk::Buffer {
        self.buffer(bytes, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .expect("out of GPU memory")
            .0
    }

    /// The command buffer open for recording: the current one of the ring,
    /// begun if it is not open, once the GPU is done with it.
    fn cmd(&self) -> vk::CommandBuffer {
        let mut ring = self.ring.lock().unwrap();
        let current = ring.current;
        let cmd = self.cmds[current];
        if !ring.recording {
            if ring.in_flight[current] {
                self.wait(&mut ring, current);
            }
            unsafe {
                self.device
                    .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                    .unwrap();
                self.device
                    .begin_command_buffer(
                        cmd,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .unwrap();
            }
            ring.recording = true;
            ring.recorded = 0;
            ring.work = 0;
        }
        cmd
    }

    /// Waits for the GPU to finish command buffer `i` of the ring.
    fn wait(&self, ring: &mut Ring, i: usize) {
        unsafe {
            self.device
                .wait_for_fences(&[self.fences[i]], true, u64::MAX)
                .unwrap();
            self.device.reset_fences(&[self.fences[i]]).unwrap();
        }
        ring.in_flight[i] = false;
    }

    /// The stages kernels and copies run in, and the dependency that makes
    /// what they wrote visible to what runs next.
    fn dependency() -> (vk::PipelineStageFlags, vk::MemoryBarrier<'static>) {
        let stages = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE,
            );
        (stages, barrier)
    }

    /// Makes everything recorded so far visible to everything recorded next.
    fn barrier(&self, cmd: vk::CommandBuffer) {
        let (stages, barrier) = Self::dependency();
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                stages,
                stages,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    /// Before a matrix product that binds a buffer written since the last
    /// wait, waits, on the GPU, for everything recorded so far to have
    /// reached the end of the pipe, and notes what the kernel about to be
    /// recorded, bound to `buffers`, writes.
    fn wait_for_writes(
        &self,
        cmd: vk::CommandBuffer,
        pipeline: vk::Pipeline,
        buffers: &[vk::Buffer],
    ) {
        let mut ring = self.ring.lock().unwrap();
        if self.kernels.matmul(pipeline) && buffers.iter().any(|buf| ring.written.contains(buf)) {
            let (stages, barrier) = Self::dependency();
            unsafe {
                self.device
                    .cmd_set_event(cmd, self.event, vk::PipelineStageFlags::ALL_COMMANDS);
                self.device.cmd_wait_events(
                    cmd,
                    &[self.event],
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    stages,
                    &[barrier],
                    &[],
                    &[],
                );
                self.device
                    .cmd_reset_event(cmd, self.event, vk::PipelineStageFlags::TOP_OF_PIPE);
            }
            ring.written.clear();
        }
        let written = self.kernels.written(pipeline);
        for (i, &buf) in buffers.iter().enumerate() {
            if written >> i & 1 == 1 {
                ring.written.push(buf);
            }
        }
    }

    /// Notes a buffer a copy or a fill writes: see `wait_for_writes`.
    fn wrote(&self, buf: vk::Buffer) {
        self.ring.lock().unwrap().written.push(buf);
    }

    /// Submits what is recorded, without waiting, and moves on to the next
    /// command buffer of the ring.
    fn submit(&self) {
        let mut ring = self.ring.lock().unwrap();
        if !ring.recording {
            return;
        }
        let i = ring.current;
        unsafe {
            self.device.end_command_buffer(self.cmds[i]).unwrap();
            let cmds = [self.cmds[i]];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            self.device
                .queue_submit(self.queue, &[submit], self.fences[i])
                .unwrap();
        }
        ring.in_flight[i] = true;
        ring.recording = false;
        ring.current = (i + 1) % RING;
    }

    /// Submits everything recorded and waits for all of it to finish.
    fn flush(&self) {
        self.submit();
        let mut ring = self.ring.lock().unwrap();
        for i in 0..RING {
            if ring.in_flight[i] {
                self.wait(&mut ring, i);
            }
        }
        ring.staging_slot = [None; 2];
    }

    /// Records a kernel over `groups` workgroups, bound to `buffers` in
    /// order, with `params` as its push constants.
    fn dispatch<P: Copy>(
        &self,
        pipeline: vk::Pipeline,
        buffers: &[vk::Buffer],
        params: &P,
        groups: (u32, u32),
    ) {
        let cmd = self.cmd();
        self.wait_for_writes(cmd, pipeline, buffers);
        // A kernel binds only so many buffers; a thousand dispatches a token
        // is worth not allocating for.
        assert!(buffers.len() <= BINDINGS && size_of::<P>() <= PUSH_CONSTANTS);
        let mut infos = [vk::DescriptorBufferInfo::default(); BINDINGS];
        for (info, &buf) in infos.iter_mut().zip(buffers) {
            *info = vk::DescriptorBufferInfo::default()
                .buffer(buf)
                .offset(0)
                .range(vk::WHOLE_SIZE);
        }
        let mut writes = [vk::WriteDescriptorSet::default(); BINDINGS];
        for (i, (write, info)) in writes.iter_mut().zip(&infos).enumerate() {
            *write = vk::WriteDescriptorSet::default()
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info));
        }
        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.push.cmd_push_descriptor_set(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &writes[..buffers.len()],
            );
            self.device.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes(params),
            );
            self.device.cmd_dispatch(cmd, groups.0, groups.1, 1);
        }
        self.barrier(cmd);
        let recorded = {
            let mut ring = self.ring.lock().unwrap();
            ring.recorded += 1;
            ring.recorded
        };
        if recorded >= SUBMIT_EVERY {
            self.submit();
        }
    }

    /// Counts `work` multiply-adds of the kernel just recorded, and submits
    /// what is recorded once it comes to [`SUBMIT_WORK`].
    fn worked(&self, work: usize) {
        let total = {
            let mut ring = self.ring.lock().unwrap();
            ring.work += work;
            ring.work
        };
        if total >= SUBMIT_WORK {
            self.submit();
        }
    }

    /// Workgroups for `count` items of work in a one-dimensional kernel.
    fn groups(&self, count: usize, per_group: usize) -> (u32, u32) {
        let groups = count.div_ceil(per_group);
        assert!(
            groups <= self.max_groups as usize,
            "{groups} workgroups is more than the GPU allows"
        );
        (groups as u32, 1)
    }

    /// Copies bytes from the host into a device buffer, through staging.
    ///
    /// Each chunk of a staging buffer's size is copied into the staging
    /// buffer the last upload did not use, once whatever last read that one
    /// has finished, and its copy to the device is submitted at once behind
    /// whatever is pending, so that the device copies one chunk while the
    /// host fills the other.
    fn upload_bytes(&self, dst: vk::Buffer, offset: usize, data: &[u8]) {
        for (i, chunk) in data.chunks(STAGING).enumerate() {
            let which = {
                let mut ring = self.ring.lock().unwrap();
                let which = ring.staging_next;
                if let Some(slot) = ring.staging_slot[which].take()
                    && ring.in_flight[slot]
                {
                    self.wait(&mut ring, slot);
                }
                which
            };
            unsafe {
                std::ptr::copy_nonoverlapping(chunk.as_ptr(), self.staging[which].ptr, chunk.len())
            };
            let cmd = self.cmd();
            let region = vk::BufferCopy::default()
                .dst_offset((offset + i * STAGING) as u64)
                .size(chunk.len() as u64);
            unsafe {
                self.device
                    .cmd_copy_buffer(cmd, self.staging[which].buf, dst, &[region])
            };
            self.barrier(cmd);
            self.wrote(dst);
            let slot = self.ring.lock().unwrap().current;
            self.submit();
            let mut ring = self.ring.lock().unwrap();
            ring.staging_slot[which] = Some(slot);
            ring.staging_next = 1 - which;
        }
    }

    /// Copies bytes from a device buffer to the host, through staging: the
    /// copy is recorded behind whatever is pending, and everything is
    /// submitted together.
    fn download_bytes(&self, src: vk::Buffer, data: &mut [u8]) {
        for (i, chunk) in data.chunks_mut(STAGING).enumerate() {
            unsafe {
                let cmd = self.cmd();
                let region = vk::BufferCopy::default()
                    .src_offset((i * STAGING) as u64)
                    .size(chunk.len() as u64);
                self.device
                    .cmd_copy_buffer(cmd, src, self.staging[0].buf, &[region]);
                self.barrier(cmd);
                self.flush();
                std::ptr::copy_nonoverlapping(self.staging[0].ptr, chunk.as_mut_ptr(), chunk.len());
            }
        }
    }
}

impl Drop for Vulkan {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for &fence in &self.fences {
                self.device.destroy_fence(fence, None);
            }
            self.device.destroy_event(self.event, None);
            for pipeline in self.kernels.all() {
                self.device.destroy_pipeline(pipeline, None);
            }
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device
                .destroy_descriptor_set_layout(self.set_layout, None);
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
    splits: u32,
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
    rot_dim: u32,
    pos: u32,
    n: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ConvParams {
    n: u32,
    channels: u32,
    q_dim: u32,
    k_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DeltaNetParams {
    n: u32,
    n_k_heads: u32,
    n_v_heads: u32,
    head_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AttentionParams {
    n_heads: u32,
    head_dim: u32,
    n_kv_heads: u32,
    pos: u32,
    first: u32,
    chunks: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct StoreParams {
    offset: u32,
    len: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LenParams {
    len: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ExpertParams {
    rows: u32,
    cols: u32,
    k: u32,
    per_token: u32,
    first: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MixParams {
    dim: u32,
    count: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GateParams {
    dim: u32,
    rank: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ShortConvParams {
    n: u32,
    channels: u32,
    taps: u32,
    heads: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ClipParams {
    heads: u32,
    width: u32,
    dim: u32,
    threshold: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RowsParams {
    len: u32,
    width: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AttentionRelParams {
    n_heads: u32,
    head_dim: u32,
    n_kv_heads: u32,
    pos: u32,
    first: u32,
    chunks: u32,
    window: u32,
    ring: u32,
    n_rel: u32,
    extent: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RouteParams {
    n: u32,
    experts: u32,
    k: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SituParams {
    len: u32,
    width: u32,
    beta: f32,
    linear_beta: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct WeightedSumParams {
    len: u32,
    width: u32,
    k: u32,
}

impl Device for Vulkan {
    type Buffer = Buffer;
    type Weight = Weight;
    type Cache = Cache;

    fn name(&self) -> String {
        // Drivers append their own name in parentheses; the GPU's is enough.
        self.name
            .split(" (")
            .next()
            .unwrap_or(&self.name)
            .to_string()
    }

    fn memory(&self) -> Option<u64> {
        // The heap of the first device-local memory type, which `buffer`
        // allocates from.
        let types = &self.memory_types.memory_types[..self.memory_types.memory_type_count as usize];
        let heap = types
            .iter()
            .find(|t| {
                t.property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })?
            .heap_index as usize;
        let size = self.memory_types.memory_heaps[heap].size;
        let mut free = size;
        if self.budget {
            let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
            let mut props = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
            unsafe {
                self.instance
                    .get_physical_device_memory_properties2(self.physical, &mut props)
            };
            free = budget.heap_budget[heap].saturating_sub(budget.heap_usage[heap]);
        }
        // An integrated GPU's heap is the host's memory, which the host's
        // programs use too, unless the firmware set aside more memory for
        // the GPU than the host has for itself.
        if self.integrated
            && let Some(total) = memory::total()
            && size <= total
            && let Some(available) = memory::available()
        {
            free = free.min(available);
        }
        Some(free)
    }

    fn takes_nvfp4(&self) -> bool {
        true
    }

    fn upload(&self, tensor: Tensor) -> Weight {
        // The bytes of a vector of plain integers, in memory order: the
        // kernels take the low half of a word as the first of its 16-bit
        // weights or scales, as little-endian does.
        fn bytes<T>(data: &[T]) -> &[u8] {
            unsafe { slice::from_raw_parts(data.as_ptr().cast::<u8>(), size_of_val(data)) }
        }
        let with_bytes = |bytes: &[u8]| {
            // A whole number of words, for the kernels' arrays of them.
            let buf = self.device_buffer(bytes.len().next_multiple_of(4));
            self.upload_bytes(buf, 0, bytes);
            buf
        };
        match tensor {
            Tensor::Bf16 { shape, data } => {
                assert!(
                    data.len().is_multiple_of(8),
                    "weights must come in multiples of eight"
                );
                Weight {
                    buf: with_bytes(bytes(&data)),
                    scales: vk::Buffer::null(),
                    shape,
                    format: Format::Bf16,
                }
            }
            Tensor::Q4_0 {
                shape,
                scales,
                quants,
            } => {
                assert_eq!(quants.len(), shape[0] * shape[1] / 2);
                assert_eq!(scales.len(), shape[0] * quant::blocks(shape[1]));
                Weight {
                    buf: with_bytes(&quants),
                    scales: with_bytes(bytes(&scales)),
                    shape,
                    format: Format::Q4_0,
                }
            }
            Tensor::Q8_0 {
                shape,
                scales,
                quants,
            } => {
                assert_eq!(quants.len(), shape[0] * shape[1]);
                assert_eq!(scales.len(), shape[0] * quant::blocks(shape[1]));
                Weight {
                    buf: with_bytes(bytes(&quants)),
                    scales: with_bytes(bytes(&scales)),
                    shape,
                    format: Format::Q8_0,
                }
            }
            Tensor::Nvfp4 {
                shape,
                scales,
                quants,
            } => {
                assert_eq!(quants.len(), shape[0] * shape[1] / 2);
                assert_eq!(scales.len(), shape[0] * shape[1] / quant::NVFP4_BLOCK);
                Weight {
                    buf: with_bytes(&quants),
                    scales: with_bytes(&scales),
                    shape,
                    format: Format::Nvfp4,
                }
            }
        }
    }

    fn alloc(&self, len: usize) -> Buffer {
        let buf = self.device_buffer(len * 4);
        let cmd = self.cmd();
        unsafe { self.device.cmd_fill_buffer(cmd, buf, 0, vk::WHOLE_SIZE, 0) };
        self.barrier(cmd);
        self.wrote(buf);
        Buffer { buf, len, cap: len }
    }

    fn alloc_cache(&self, len: usize) -> Cache {
        assert!(len.is_multiple_of(CACHE_BLOCK));
        let quants = self.device_buffer(len);
        let scales = self.device_buffer((len / CACHE_BLOCK).div_ceil(2) * 4);
        let cmd = self.cmd();
        for buf in [quants, scales] {
            unsafe { self.device.cmd_fill_buffer(cmd, buf, 0, vk::WHOLE_SIZE, 0) };
        }
        self.barrier(cmd);
        self.wrote(quants);
        self.wrote(scales);
        Cache {
            quants,
            scales,
            len,
        }
    }

    fn resize(&self, buf: &mut Buffer, len: usize) {
        assert!(
            len <= buf.cap,
            "a buffer of {} activations can't hold {len}",
            buf.cap
        );
        buf.len = len;
    }

    fn read(&self, buf: &Buffer) -> Vec<f32> {
        let mut out = vec![0.0f32; buf.len];
        let bytes =
            unsafe { slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), buf.len * 4) };
        self.download_bytes(buf.buf, bytes);
        out
    }

    fn write(&self, buf: &mut Buffer, data: &[f32]) {
        assert_eq!(buf.len, data.len());
        let bytes = unsafe { slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
        self.upload_bytes(buf.buf, 0, bytes);
    }

    fn copy(
        &self,
        dst: &mut Buffer,
        dst_offset: usize,
        src: &Buffer,
        src_offset: usize,
        len: usize,
    ) {
        assert!(dst_offset + len <= dst.len && src_offset + len <= src.len);
        let cmd = self.cmd();
        let region = vk::BufferCopy::default()
            .src_offset((src_offset * 4) as u64)
            .dst_offset((dst_offset * 4) as u64)
            .size((len * 4) as u64);
        unsafe {
            self.device
                .cmd_copy_buffer(cmd, src.buf, dst.buf, &[region])
        };
        self.barrier(cmd);
        self.wrote(dst.buf);
    }

    fn store(&self, cache: &mut Cache, offset: usize, src: &Buffer) {
        assert!(offset + src.len <= cache.len);
        assert!(offset.is_multiple_of(CACHE_BLOCK) && src.len.is_multiple_of(CACHE_BLOCK));
        let params = StoreParams {
            offset: offset as u32,
            len: src.len as u32,
        };
        let (first, blocks) = (offset / CACHE_BLOCK, src.len / CACHE_BLOCK);
        let words = (first + blocks).div_ceil(2) - first / 2;
        let groups = self.groups(words, 64);
        self.dispatch(
            self.kernels.store_q8,
            &[cache.quants, cache.scales, src.buf],
            &params,
            groups,
        );
    }

    fn read_cache(&self, cache: &Cache, len: usize) -> Vec<u8> {
        assert!(len <= cache.len);
        let mut out = vec![0; crate::gpu::cache_bytes(len)];
        let (quants, scales) = out.split_at_mut(len);
        self.download_bytes(cache.quants, quants);
        self.download_bytes(cache.scales, scales);
        out
    }

    fn write_cache(&self, cache: &mut Cache, data: &[u8]) {
        let len = data.len() / (CACHE_BLOCK + 2) * CACHE_BLOCK;
        assert!(len <= cache.len && data.len() == crate::gpu::cache_bytes(len));
        let (quants, scales) = data.split_at(len);
        self.upload_bytes(cache.quants, 0, quants);
        self.upload_bytes(cache.scales, 0, scales);
    }

    fn copy_cache(&self, dst: &mut Cache, src: &Cache, len: usize) {
        assert!(len <= dst.len && len <= src.len && len.is_multiple_of(CACHE_BLOCK));
        if len == 0 {
            return;
        }
        // The quants, and the scales after them in what a cache reads
        // back as.
        let scales = cache_bytes(len) - len;
        let cmd = self.cmd();
        for (from, to, bytes) in [
            (src.quants, dst.quants, len),
            (src.scales, dst.scales, scales),
        ] {
            let region = vk::BufferCopy::default().size(bytes as u64);
            unsafe { self.device.cmd_copy_buffer(cmd, from, to, &[region]) };
            self.wrote(to);
        }
        self.barrier(cmd);
    }

    fn matmul(&self, out: &mut Buffer, w: &Weight, x: &Buffer) {
        let (rows, cols) = (w.shape[0], w.shape[1]);
        let n = x.len / cols;
        assert_eq!(x.len, n * cols);
        assert_eq!(out.len, n * rows);
        assert!(cols % 8 == 0);
        // One workgroup per row of bf16 or block-quantized weights, in a
        // grid as wide as the GPU allows, and the tokens eight at a time,
        // each group of them a row of the grid. The block-quantized kernels
        // bind the scales after the quants.
        let (kernel, per_group, tokens, x) = match w.format {
            Format::Bf16 => (self.kernels.matmul_bf16, 1, BF16_TOKENS, x.buf),
            Format::Q4_0 => (self.kernels.matmul_q4_0, 1, BF16_TOKENS, x.buf),
            Format::Q8_0 => (self.kernels.matmul_q8_0, 1, BF16_TOKENS, x.buf),
            Format::Nvfp4 => panic!("NVFP4 weights are the experts', which a token picks"),
        };
        let count = rows.div_ceil(per_group);
        let width = count.min(self.max_groups as usize);
        let groups = (
            width as u32,
            (count.div_ceil(width) * n.div_ceil(tokens)) as u32,
        );
        let params = MatmulParams {
            rows: rows as u32,
            cols: cols as u32,
            n: n as u32,
            stride: width as u32,
            splits: 1,
        };
        if w.scales == vk::Buffer::null() {
            self.dispatch(kernel, &[out.buf, w.buf, x], &params, groups);
        } else {
            self.dispatch(kernel, &[out.buf, w.buf, w.scales, x], &params, groups);
        }
        self.worked(n * rows * cols);
    }

    fn add(&self, x: &mut Buffer, y: &Buffer) {
        assert_eq!(x.len, y.len);
        let params = LenParams { len: x.len as u32 };
        let groups = self.groups(x.len, 256);
        self.dispatch(self.kernels.add, &[x.buf, y.buf], &params, groups);
    }

    fn rmsnorm(&self, x: &mut Buffer, weight: &Buffer, eps: f32) {
        let dim = weight.len;
        assert_eq!(x.len % dim, 0);
        let params = RmsnormParams {
            dim: dim as u32,
            eps,
        };
        let groups = self.groups(x.len / dim, 1);
        self.dispatch(self.kernels.rmsnorm, &[x.buf, weight.buf], &params, groups);
    }

    fn l2norm(&self, x: &mut Buffer, dim: usize, eps: f32) {
        assert_eq!(x.len % dim, 0);
        let params = RmsnormParams {
            dim: dim as u32,
            eps,
        };
        let groups = self.groups(x.len / dim, 1);
        self.dispatch(self.kernels.l2norm, &[x.buf], &params, groups);
    }

    fn rope(
        &self,
        x: &mut Buffer,
        table: &Buffer,
        pos: usize,
        n_heads: usize,
        head_dim: usize,
        rot_dim: usize,
    ) {
        let n = x.len / (n_heads * head_dim);
        assert_eq!(x.len, n * n_heads * head_dim);
        assert!(rot_dim <= head_dim && rot_dim.is_multiple_of(2));
        let params = RopeParams {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            rot_dim: rot_dim as u32,
            pos: pos as u32,
            n: n as u32,
        };
        let groups = self.groups(n * n_heads * rot_dim / 2, 256);
        self.dispatch(self.kernels.rope, &[x.buf, table.buf], &params, groups);
    }

    fn attention(
        &self,
        out: &mut Buffer,
        q: &Buffer,
        k_cache: &Cache,
        v_cache: &Cache,
        pos: usize,
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
    ) {
        let n = q.len / (n_heads * head_dim);
        assert_eq!(q.len, n * n_heads * head_dim);
        assert_eq!(out.len, q.len);
        assert!(head_dim.is_multiple_of(CACHE_BLOCK) && head_dim <= 256);
        assert!(
            n_heads / n_kv_heads <= 8,
            "the kernel holds up to eight query heads per key/value head"
        );
        // Every token's heads get as many chunks as the last token attends
        // over, as many tokens to a dispatch as the partials fit.
        let chunks = (pos + n).div_ceil(CHUNK);
        let per_token = n_heads * chunks * (head_dim + 2);
        assert!(
            per_token <= PARTIALS,
            "attention over {} positions needs more partials than {PARTIALS}",
            pos + n
        );
        // And no more than a submission has work for: a head scores and
        // weighs every position.
        let work = n_heads * (pos + n) * head_dim * 2;
        let per_dispatch = (PARTIALS / per_token).min((SUBMIT_WORK / work).max(1));
        for first in (0..n).step_by(per_dispatch) {
            let tokens = per_dispatch.min(n - first);
            let params = AttentionParams {
                n_heads: n_heads as u32,
                head_dim: head_dim as u32,
                n_kv_heads: n_kv_heads as u32,
                pos: pos as u32,
                first: first as u32,
                chunks: chunks as u32,
            };
            let groups = self.groups(tokens * n_kv_heads * chunks, 1);
            self.dispatch(
                self.kernels.attention,
                &[
                    out.buf,
                    q.buf,
                    k_cache.quants,
                    k_cache.scales,
                    v_cache.quants,
                    v_cache.scales,
                    self.partials,
                ],
                &params,
                groups,
            );
            if chunks > 1 {
                let groups = self.groups(tokens * n_heads, 1);
                self.dispatch(
                    self.kernels.attention_combine,
                    &[out.buf, self.partials],
                    &params,
                    groups,
                );
            }
            self.worked(tokens * work);
        }
    }

    fn silu_mul(&self, gate: &mut Buffer, up: &Buffer) {
        assert_eq!(gate.len, up.len);
        let params = LenParams {
            len: gate.len as u32,
        };
        let groups = self.groups(gate.len, 256);
        self.dispatch(self.kernels.silu_mul, &[gate.buf, up.buf], &params, groups);
    }

    fn sigmoid_mul(&self, x: &mut Buffer, gate: &Buffer) {
        assert_eq!(x.len, gate.len);
        let params = LenParams { len: x.len as u32 };
        let groups = self.groups(x.len, 256);
        self.dispatch(
            self.kernels.sigmoid_mul,
            &[x.buf, gate.buf],
            &params,
            groups,
        );
    }

    fn conv(
        &self,
        q: &mut Buffer,
        k: &mut Buffer,
        v: &mut Buffer,
        state_out: &mut Buffer,
        x: &Buffer,
        state: &Buffer,
        weight: &Buffer,
    ) {
        let channels = weight.len / CONV_KERNEL;
        let n = x.len / channels;
        assert_eq!(x.len, n * channels);
        assert!(state.len == (CONV_KERNEL - 1) * channels && state_out.len == state.len);
        let (q_dim, k_dim, v_dim) = (q.len / n, k.len / n, v.len / n);
        assert_eq!(q_dim + k_dim + v_dim, channels);
        let params = ConvParams {
            n: n as u32,
            channels: channels as u32,
            q_dim: q_dim as u32,
            k_dim: k_dim as u32,
        };
        let groups = self.groups(n * channels, 256);
        let buffers = [
            q.buf,
            k.buf,
            v.buf,
            state_out.buf,
            x.buf,
            state.buf,
            weight.buf,
        ];
        self.dispatch(self.kernels.conv, &buffers, &params, groups);
    }

    fn delta_net(
        &self,
        out: &mut Buffer,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        gates: &Buffer,
        decay: &Buffer,
        state: &mut Buffer,
        n_k_heads: usize,
        n_v_heads: usize,
        head_dim: usize,
    ) {
        let n = v.len / (n_v_heads * head_dim);
        assert_eq!(v.len, n * n_v_heads * head_dim);
        assert!(q.len == n * n_k_heads * head_dim && k.len == q.len && out.len == v.len);
        assert!(gates.len == n * 2 * n_v_heads && decay.len == 2 * n_v_heads);
        assert_eq!(state.len, n_v_heads * head_dim * head_dim);
        assert_eq!(head_dim, 128, "the kernel is written for 128-wide heads");
        let params = DeltaNetParams {
            n: n as u32,
            n_k_heads: n_k_heads as u32,
            n_v_heads: n_v_heads as u32,
            head_dim: head_dim as u32,
        };
        let buffers = [
            out.buf, q.buf, k.buf, v.buf, gates.buf, decay.buf, state.buf,
        ];
        self.dispatch(
            self.kernels.delta_net,
            &buffers,
            &params,
            self.groups(n_v_heads, 1),
        );
    }

    fn mix(&self, out: &mut Buffer, states: &[&Buffer], query: &Buffer, eps: f32) {
        let dim = query.len;
        assert!(
            !states.is_empty() && states.len() <= 6,
            "the kernel mixes up to six states"
        );
        assert!(dim.is_multiple_of(4), "the kernel takes rows in fours");
        assert!(states.iter().all(|state| state.len == out.len) && out.len.is_multiple_of(dim));
        let params = MixParams {
            dim: dim as u32,
            count: states.len() as u32,
            eps,
        };
        // The kernel binds six states: the ones there are not are the
        // first again, which it does not read.
        let state = |i: usize| states[i.min(states.len() - 1)].buf;
        let buffers = [
            out.buf,
            query.buf,
            state(0),
            state(1),
            state(2),
            state(3),
            state(4),
            state(5),
        ];
        let groups = self.groups(out.len / dim, 1);
        self.dispatch(self.kernels.mix, &buffers, &params, groups);
    }

    fn gated_update(&self, x: &mut Buffer, forget: &Buffer, gain: &Buffer, message: &Buffer) {
        assert!(forget.len == x.len && gain.len == x.len && message.len == x.len);
        let params = LenParams { len: x.len as u32 };
        let groups = self.groups(x.len, 256);
        self.dispatch(
            self.kernels.gated_update,
            &[x.buf, forget.buf, gain.buf, message.buf],
            &params,
            groups,
        );
    }

    fn gate(
        &self,
        x: &mut Buffer,
        message: &Buffer,
        down_input: &Weight,
        down_message: &Weight,
        gain: &Weight,
        forget: &Weight,
        dim: usize,
        rank: usize,
        eps: f32,
    ) {
        assert!(x.len.is_multiple_of(dim) && message.len == x.len);
        assert!(
            dim.is_multiple_of(8) && rank.is_multiple_of(8) && rank <= 64,
            "the kernel takes a gate of up to 64 dimensions, in eights"
        );
        for (w, shape) in [
            (down_input, [rank, dim]),
            (down_message, [rank, dim]),
            (gain, [dim, rank]),
            (forget, [dim, rank]),
        ] {
            assert!(
                w.format == Format::Bf16 && w.shape == shape,
                "a gate's weights are bf16"
            );
        }
        let params = GateParams {
            dim: dim as u32,
            rank: rank as u32,
            eps,
        };
        let buffers = [
            x.buf,
            message.buf,
            down_input.buf,
            down_message.buf,
            gain.buf,
            forget.buf,
        ];
        let groups = self.groups(x.len / dim, 1);
        self.dispatch(self.kernels.gate, &buffers, &params, groups);
    }

    fn short_conv(
        &self,
        out: &mut Buffer,
        x: &Buffer,
        state: &mut Buffer,
        weight: &Buffer,
        dynamic: Option<(&Buffer, usize)>,
    ) {
        let channels = weight.len - state.len;
        let taps = weight.len / channels;
        let n = x.len / channels;
        assert!(x.len == n * channels && out.len == x.len);
        assert_eq!(state.len, (taps - 1) * channels);
        assert!(taps <= 9, "the kernel keeps the state of up to nine taps");
        let heads = dynamic.map_or(0, |(_, heads)| heads);
        if let Some((dynamic, heads)) = dynamic {
            assert!(channels.is_multiple_of(heads) && dynamic.len == n * heads * taps);
        }
        let params = ShortConvParams {
            n: n as u32,
            channels: channels as u32,
            taps: taps as u32,
            heads: heads as u32,
        };
        // Without taps of the heads' own the kernel reads none: the
        // weight stands in for them.
        let dynamic = dynamic.map_or(weight.buf, |(dynamic, _)| dynamic.buf);
        let groups = self.groups(n * channels, 256);
        self.dispatch(
            self.kernels.short_conv,
            &[out.buf, x.buf, state.buf, weight.buf, dynamic],
            &params,
            groups,
        );
        // A token alone has its state moved on by the convolution.
        if n > 1 && taps > 1 {
            let groups = self.groups(channels, 256);
            self.dispatch(
                self.kernels.short_conv_state,
                &[state.buf, x.buf],
                &params,
                groups,
            );
        }
    }

    fn clip(&self, x: &mut Buffer, scale: &Buffer, dim: usize, threshold: f32) {
        let width = scale.len;
        assert!(x.len.is_multiple_of(width) && width.is_multiple_of(dim));
        let heads = x.len / dim;
        let params = ClipParams {
            heads: heads as u32,
            width: width as u32,
            dim: dim as u32,
            threshold,
        };
        let groups = self.groups(heads, 64);
        self.dispatch(self.kernels.clip, &[x.buf, scale.buf], &params, groups);
    }

    fn scale_rows(&self, x: &mut Buffer, factors: &Buffer) {
        assert!(x.len.is_multiple_of(factors.len));
        let params = RowsParams {
            len: x.len as u32,
            width: (x.len / factors.len) as u32,
        };
        let groups = self.groups(x.len, 256);
        self.dispatch(
            self.kernels.scale_rows,
            &[x.buf, factors.buf],
            &params,
            groups,
        );
    }

    fn attention_rel(
        &self,
        out: &mut Buffer,
        q: &Buffer,
        k_cache: &Cache,
        v_cache: &Cache,
        rel: &Buffer,
        bias: &Buffer,
        sinks: &Buffer,
        pos: usize,
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
        window: usize,
        rows: usize,
    ) {
        let n = q.len / (n_heads * head_dim);
        assert_eq!(q.len, n * n_heads * head_dim);
        assert_eq!(out.len, q.len);
        assert!(head_dim.is_multiple_of(CACHE_BLOCK) && head_dim <= 256);
        assert!(
            n_heads / n_kv_heads <= 8,
            "the kernel holds up to eight query heads per key/value head"
        );
        assert!(sinks.len == n * n_heads && rel.len.is_multiple_of(n * n_heads));
        let n_rel = rel.len / (n * n_heads);
        assert!(bias.len.is_multiple_of(n_rel));
        let kv_dim = n_kv_heads * head_dim;
        let held = (pos + n).min(rows) * kv_dim;
        assert!(held <= k_cache.len && held <= v_cache.len);
        // Every token's heads get as many chunks as the last token attends
        // over, as many tokens to a dispatch as the partials fit.
        let chunks = (pos + n).min(window).div_ceil(CHUNK);
        let per_token = n_heads * chunks * (head_dim + 2);
        assert!(
            per_token <= PARTIALS,
            "attention over {} positions needs more partials than {PARTIALS}",
            pos + n
        );
        // And no more than a submission has work for: a head scores and
        // weighs every position in the window.
        let work = n_heads * (pos + n).min(window) * head_dim * 2;
        let per_dispatch = (PARTIALS / per_token).min((SUBMIT_WORK / work).max(1));
        for first in (0..n).step_by(per_dispatch) {
            let tokens = per_dispatch.min(n - first);
            let params = AttentionRelParams {
                n_heads: n_heads as u32,
                head_dim: head_dim as u32,
                n_kv_heads: n_kv_heads as u32,
                pos: pos as u32,
                first: first as u32,
                chunks: chunks as u32,
                window: window.min(u32::MAX as usize) as u32,
                ring: rows as u32,
                n_rel: n_rel as u32,
                extent: (bias.len / n_rel) as u32,
            };
            let groups = self.groups(tokens * n_kv_heads * chunks, 1);
            self.dispatch(
                self.kernels.attention_rel,
                &[
                    out.buf,
                    q.buf,
                    k_cache.quants,
                    k_cache.scales,
                    v_cache.quants,
                    v_cache.scales,
                    self.partials,
                    rel.buf,
                    bias.buf,
                    sinks.buf,
                ],
                &params,
                groups,
            );
            if chunks > 1 {
                let groups = self.groups(tokens * n_heads, 1);
                self.dispatch(
                    self.kernels.attention_rel_combine,
                    &[out.buf, self.partials, sinks.buf],
                    &params,
                    groups,
                );
            }
            self.worked(tokens * work);
        }
    }

    fn route(
        &self,
        ids: &mut Buffer,
        weights: &mut Buffer,
        logits: &Buffer,
        bias: &Buffer,
        k: usize,
        scale: f32,
    ) {
        let experts = bias.len;
        let n = logits.len / experts;
        assert_eq!(logits.len, n * experts);
        assert!(ids.len == n * k && weights.len == n * k);
        assert!(
            k <= experts && k <= 32,
            "the kernel picks up to 32 of the experts"
        );
        let params = RouteParams {
            n: n as u32,
            experts: experts as u32,
            k: k as u32,
            scale,
        };
        let groups = self.groups(n, 64);
        self.dispatch(
            self.kernels.route,
            &[ids.buf, weights.buf, logits.buf, bias.buf],
            &params,
            groups,
        );
    }

    fn expert_matmul(
        &self,
        out: &mut Buffer,
        w: &Weight,
        scale: &Buffer,
        x: &Buffer,
        ids: &Buffer,
        k: usize,
    ) {
        let cols = w.shape[1];
        let rows = w.shape[0] / scale.len;
        assert_eq!(w.shape[0], rows * scale.len);
        let pairs = ids.len;
        assert_eq!(out.len, pairs * rows);
        // A row of the input per token, or one per expert of a token.
        let shared = x.len == pairs / k * cols;
        assert!(shared || x.len == pairs * cols);
        let kernel = match w.format {
            Format::Nvfp4 => self.kernels.expert_matmul_nvfp4,
            Format::Q8_0 => self.kernels.expert_matmul_q8_0,
            _ => panic!("the experts' weights are NVFP4 or Q8_0"),
        };
        // One workgroup per row of a pair's matrix, the pairs down the
        // grid's second dimension: as many of them to a dispatch as a
        // submission has work for.
        let most = self.max_groups as usize;
        assert!(
            rows <= most,
            "{rows} rows is more workgroups than the GPU allows"
        );
        let per_dispatch = (SUBMIT_WORK / (rows * cols)).clamp(1, most);
        for first in (0..pairs).step_by(per_dispatch) {
            let count = per_dispatch.min(pairs - first);
            let params = ExpertParams {
                rows: rows as u32,
                cols: cols as u32,
                k: k as u32,
                per_token: shared as u32,
                first: first as u32,
            };
            self.dispatch(
                kernel,
                &[out.buf, w.buf, w.scales, x.buf, ids.buf, scale.buf],
                &params,
                (rows as u32, count as u32),
            );
            self.worked(count * rows * cols);
        }
    }

    fn situ(&self, out: &mut Buffer, gate_up: &Buffer, width: usize, beta: f32, linear_beta: f32) {
        assert!(out.len.is_multiple_of(width) && gate_up.len == 2 * out.len);
        let params = SituParams {
            len: out.len as u32,
            width: width as u32,
            beta,
            linear_beta,
        };
        let groups = self.groups(out.len, 256);
        self.dispatch(self.kernels.situ, &[out.buf, gate_up.buf], &params, groups);
    }

    fn weighted_sum(&self, out: &mut Buffer, y: &Buffer, weights: &Buffer) {
        let width = y.len / weights.len;
        assert!(y.len == weights.len * width && out.len.is_multiple_of(width));
        let k = weights.len / (out.len / width);
        assert_eq!(out.len / width * k, weights.len);
        let params = WeightedSumParams {
            len: out.len as u32,
            width: width as u32,
            k: k as u32,
        };
        let groups = self.groups(out.len, 256);
        self.dispatch(
            self.kernels.weighted_sum,
            &[out.buf, y.buf, weights.buf],
            &params,
            groups,
        );
    }
}

/// The bytes of a plain struct of 32-bit fields, as push constants.
fn bytes<P: Copy>(params: &P) -> &[u8] {
    unsafe { slice::from_raw_parts((params as *const P).cast::<u8>(), size_of::<P>()) }
}

#[cfg(test)]
mod tests {
    check_against_cpu!(super::Vulkan::new());

    /// Times the Q4_0 matmul on the model's shapes; run with `--ignored
    /// --nocapture`.
    #[test]
    #[ignore]
    fn q4_0_matmul_speed() {
        match super::Vulkan::new() {
            Ok(gpu) => crate::gpu::tests::q4_0_matmul_speed(&gpu),
            Err(e) => eprintln!("skipping: {e}"),
        }
    }

    /// Times the Q8_0 matmul likewise.
    #[test]
    #[ignore]
    fn q8_0_matmul_speed() {
        match super::Vulkan::new() {
            Ok(gpu) => crate::gpu::tests::q8_0_matmul_speed(&gpu),
            Err(e) => eprintln!("skipping: {e}"),
        }
    }

    /// Times the kernels at the model's shapes, for one token and for a
    /// batch of 64; run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn kernel_speed() {
        match super::Vulkan::new() {
            Ok(gpu) => {
                for n in [1, 64] {
                    eprintln!("batch of {n}:");
                    crate::gpu::tests::kernel_speed(&gpu, n);
                }
            }
            Err(e) => eprintln!("skipping: {e}"),
        }
    }

    /// Times the batched attention kernel at depths of context; run with
    /// `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn attention_speed() {
        match super::Vulkan::new() {
            Ok(gpu) => crate::gpu::tests::attention_speed(&gpu, 64),
            Err(e) => eprintln!("skipping: {e}"),
        }
    }

    /// Times the experts' matmul at Inkling's shapes; run with `--ignored
    /// --nocapture`.
    #[test]
    #[ignore]
    fn expert_matmul_speed() {
        match super::Vulkan::new() {
            Ok(gpu) => crate::gpu::tests::expert_matmul_speed(&gpu),
            Err(e) => eprintln!("skipping: {e}"),
        }
    }

    /// Times the cost of a dispatch; run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn dispatch_overhead() {
        match super::Vulkan::new() {
            Ok(gpu) => crate::gpu::tests::dispatch_overhead(&gpu),
            Err(e) => eprintln!("skipping: {e}"),
        }
    }
}
