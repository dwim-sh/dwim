//! The model on a GPU, through Metal: every operation is a hand-written
//! kernel in `kernels/`, in the Metal Shading Language, compiled for the GPU
//! when the device opens.
//!
//! As with Vulkan, operations are encoded into one command buffer as they
//! are called and committed together the first time something is read back.
//! Buffers live in memory the CPU shares with the GPU, so reading and
//! writing them needs no staging, only waiting for the commands before.

use std::{cell::RefCell, error::Error, ptr::NonNull, slice};

use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

use crate::{CACHE_BLOCK, CONV_KERNEL, Device, HADAMARD_BLOCK, Tensor};

/// Most attention scores one dispatch of the attention kernel computes, one
/// per head of each token for each position it attends to: the size of the
/// scores buffer. Longer batches are dispatched a few tokens at a time.
const SCORES: usize = 16 << 20;

/// Threads in a threadgroup of every kernel but attention's.
const THREADS: usize = 256;

/// Rows of ternary weights one SIMD group of the matmul kernel takes.
const TERNARY_ROWS: usize = 8;

/// Threads in a threadgroup of the attention kernel: one per element of a
/// head, for heads up to this long.
const ATTENTION_THREADS: usize = 256;

type Object<P> = Retained<ProtocolObject<P>>;

/// A vector of f32 activations in memory shared with the GPU.
pub struct Buffer {
    buf: Object<dyn MTLBuffer>,
    len: usize,
    cap: usize,
}

/// A cache of activations in memory shared with the GPU, in blocks of 8-bit
/// integers, and the blocks' f16 scales.
pub struct Cache {
    quants: Object<dyn MTLBuffer>,
    scales: Object<dyn MTLBuffer>,
    len: usize,
}

/// A weight matrix in memory shared with the GPU, bf16 or ternary.
pub struct Weight {
    buf: Object<dyn MTLBuffer>,
    shape: Vec<usize>,
    ternary: bool,
}

struct Kernels {
    matmul_bf16: Object<dyn MTLComputePipelineState>,
    matmul_ternary: Object<dyn MTLComputePipelineState>,
    matmul_ternary_batch: Object<dyn MTLComputePipelineState>,
    add: Object<dyn MTLComputePipelineState>,
    rmsnorm: Object<dyn MTLComputePipelineState>,
    l2norm: Object<dyn MTLComputePipelineState>,
    rope: Object<dyn MTLComputePipelineState>,
    attention: Object<dyn MTLComputePipelineState>,
    silu_mul: Object<dyn MTLComputePipelineState>,
    sigmoid_mul: Object<dyn MTLComputePipelineState>,
    copy: Object<dyn MTLComputePipelineState>,
    store_q8: Object<dyn MTLComputePipelineState>,
    hadamard: Object<dyn MTLComputePipelineState>,
    rmsnorm_hadamard: Object<dyn MTLComputePipelineState>,
    conv: Object<dyn MTLComputePipelineState>,
    delta_net: Object<dyn MTLComputePipelineState>,
}

/// A command buffer with commands encoded but not yet committed.
struct Pending {
    cmd: Object<dyn MTLCommandBuffer>,
    encoder: Object<dyn MTLComputeCommandEncoder>,
}

pub struct Metal {
    device: Object<dyn MTLDevice>,
    queue: Object<dyn MTLCommandQueue>,
    kernels: Kernels,
    scores: Object<dyn MTLBuffer>,
    /// Rows of the weights each matmul threadgroup takes: one per SIMD group.
    matmul_rows: usize,
    name: String,
    pending: RefCell<Option<Pending>>,
}

// Commands are only encoded from whichever thread owns the device.
unsafe impl Send for Metal {}

impl Metal {
    /// Opens the system's default GPU.
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let name = device.name().to_string();
        let queue = device.newCommandQueue().ok_or("no Metal command queue")?;

        let kernel = |name: &str,
                      source: &str|
         -> Result<Object<dyn MTLComputePipelineState>, Box<dyn Error>> {
            let library = device
                .newLibraryWithSource_options_error(&NSString::from_str(source), None)
                .map_err(|e| format!("{name}.metal: {}", e.localizedDescription()))?;
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| format!("{name}.metal has no kernel named {name}"))?;
            let pipeline = device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| format!("{name}: {}", e.localizedDescription()))?;
            if pipeline.maxTotalThreadsPerThreadgroup() < THREADS {
                return Err(format!("{name} can't run {THREADS} threads to a threadgroup").into());
            }
            Ok(pipeline)
        };
        macro_rules! msl {
            ($name:literal) => {
                kernel($name, include_str!(concat!("kernels/", $name, ".metal")))?
            };
        }
        let kernels = Kernels {
            matmul_bf16: msl!("matmul_bf16"),
            matmul_ternary: msl!("matmul_ternary"),
            matmul_ternary_batch: msl!("matmul_ternary_batch"),
            add: msl!("add"),
            rmsnorm: msl!("rmsnorm"),
            l2norm: msl!("l2norm"),
            rope: msl!("rope"),
            attention: msl!("attention"),
            silu_mul: msl!("silu_mul"),
            sigmoid_mul: msl!("sigmoid_mul"),
            copy: msl!("copy"),
            store_q8: msl!("store_q8"),
            hadamard: msl!("hadamard"),
            rmsnorm_hadamard: msl!("rmsnorm_hadamard"),
            conv: msl!("conv"),
            delta_net: msl!("delta_net"),
        };
        let matmul_rows = THREADS / kernels.matmul_bf16.threadExecutionWidth();
        let scores = device
            .newBufferWithLength_options(SCORES * 4, MTLResourceOptions::StorageModePrivate)
            .ok_or("out of GPU memory")?;
        Ok(Self {
            device,
            queue,
            kernels,
            scores,
            matmul_rows,
            name,
            pending: RefCell::new(None),
        })
    }

    /// Name of the GPU.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Creates a buffer of `bytes` in memory shared with the GPU, which Metal
    /// clears to zero.
    fn buffer(&self, bytes: usize) -> Object<dyn MTLBuffer> {
        self.device
            .newBufferWithLength_options(bytes.max(4), MTLResourceOptions::StorageModeShared)
            .expect("out of GPU memory")
    }

    /// Encodes a kernel over `groups` threadgroups of `threads` threads,
    /// bound to `buffers` in order, with `params` in the slot after them.
    fn dispatch<P: Copy>(
        &self,
        kernel: &ProtocolObject<dyn MTLComputePipelineState>,
        buffers: &[&ProtocolObject<dyn MTLBuffer>],
        params: &P,
        groups: usize,
        threads: usize,
    ) {
        let mut pending = self.pending.borrow_mut();
        let encoder = &pending.get_or_insert_with(|| self.begin()).encoder;
        encoder.setComputePipelineState(kernel);
        unsafe {
            for (i, &buf) in buffers.iter().enumerate() {
                encoder.setBuffer_offset_atIndex(Some(buf), 0, i);
            }
            encoder.setBytes_length_atIndex(
                NonNull::from(params).cast(),
                size_of::<P>(),
                buffers.len(),
            );
        }
        let size = |width| MTLSize {
            width,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(threads));
    }

    /// Opens a command buffer to encode into. Its compute encoder runs each
    /// kernel only once the kernels before it are done.
    fn begin(&self) -> Pending {
        // Both come back autoreleased: the pool lets go of them now rather
        // than when the thread exits.
        autoreleasepool(|_| {
            let cmd = self.queue.commandBuffer().expect("no Metal command buffer");
            let encoder = cmd
                .computeCommandEncoder()
                .expect("no Metal compute encoder");
            Pending { cmd, encoder }
        })
    }

    /// Commits everything encoded and waits for it to finish.
    fn flush(&self) {
        let Some(Pending { cmd, encoder }) = self.pending.borrow_mut().take() else {
            return;
        };
        encoder.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if let Some(e) = cmd.error() {
            panic!("GPU commands failed: {}", e.localizedDescription());
        }
    }
}

impl Drop for Metal {
    fn drop(&mut self) {
        // An encoder must be ended before it is released.
        self.flush();
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MatmulParams {
    rows: u32,
    cols: u32,
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
    rot_dim: u32,
    pos: u32,
    n: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HadamardParams {
    width: u32,
    inverse: u32,
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
    stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CopyParams {
    dst_offset: u32,
    src_offset: u32,
    len: u32,
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

impl Device for Metal {
    type Buffer = Buffer;
    type Weight = Weight;
    type Cache = Cache;

    fn upload(&self, tensor: Tensor) -> Weight {
        let (shape, bytes, ternary) = match &tensor {
            Tensor::Bf16 { shape, data } => {
                assert!(
                    data.len().is_multiple_of(8),
                    "weights must come in multiples of eight"
                );
                let bytes =
                    unsafe { slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 2) };
                (shape, bytes, false)
            }
            Tensor::Ternary { shape, data } => {
                assert_eq!(data.len(), shape[0] * crate::ternary::row_bytes(shape[1]));
                (shape, data.as_slice(), true)
            }
        };
        let buf = unsafe {
            self.device
                .newBufferWithBytes_length_options(
                    NonNull::from(bytes).cast(),
                    bytes.len(),
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("out of GPU memory")
        };
        Weight {
            buf,
            shape: shape.clone(),
            ternary,
        }
    }

    fn alloc(&self, len: usize) -> Buffer {
        Buffer {
            buf: self.buffer(len * 4),
            len,
            cap: len,
        }
    }

    fn alloc_cache(&self, len: usize) -> Cache {
        assert!(len.is_multiple_of(CACHE_BLOCK));
        Cache {
            quants: self.buffer(len),
            scales: self.buffer(len / CACHE_BLOCK * 2),
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
        self.flush();
        unsafe {
            slice::from_raw_parts(buf.buf.contents().as_ptr().cast::<f32>(), buf.len).to_vec()
        }
    }

    fn write(&self, buf: &mut Buffer, data: &[f32]) {
        assert_eq!(buf.len, data.len());
        // The pending commands may use what the buffer holds now.
        self.flush();
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                buf.buf.contents().as_ptr().cast(),
                data.len(),
            )
        };
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
        let params = CopyParams {
            dst_offset: dst_offset as u32,
            src_offset: src_offset as u32,
            len: len as u32,
        };
        let groups = len.div_ceil(THREADS);
        self.dispatch(
            &self.kernels.copy,
            &[&dst.buf, &src.buf],
            &params,
            groups,
            THREADS,
        );
    }

    fn store(&self, cache: &mut Cache, offset: usize, src: &Buffer) {
        assert!(offset + src.len <= cache.len);
        assert!(offset.is_multiple_of(CACHE_BLOCK) && src.len.is_multiple_of(CACHE_BLOCK));
        let params = StoreParams {
            offset: offset as u32,
            len: src.len as u32,
        };
        let groups = (src.len / CACHE_BLOCK).div_ceil(THREADS);
        self.dispatch(
            &self.kernels.store_q8,
            &[&cache.quants, &cache.scales, &src.buf],
            &params,
            groups,
            THREADS,
        );
    }

    fn read_cache(&self, cache: &Cache, len: usize) -> Vec<u8> {
        assert!(len <= cache.len && len.is_multiple_of(CACHE_BLOCK));
        self.flush();
        let bytes = |buf: &Object<dyn MTLBuffer>, len: usize| unsafe {
            slice::from_raw_parts(buf.contents().as_ptr().cast::<u8>(), len)
        };
        let mut out = bytes(&cache.quants, len).to_vec();
        out.extend_from_slice(bytes(&cache.scales, len / CACHE_BLOCK * 2));
        out
    }

    fn write_cache(&self, cache: &mut Cache, data: &[u8]) {
        let len = data.len() / (CACHE_BLOCK + 2) * CACHE_BLOCK;
        assert!(len <= cache.len && data.len() == crate::cache_bytes(len));
        let (quants, scales) = data.split_at(len);
        // The pending commands may use what the cache holds now.
        self.flush();
        for (buf, data) in [(&cache.quants, quants), (&cache.scales, scales)] {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    buf.contents().as_ptr().cast(),
                    data.len(),
                )
            };
        }
    }

    fn matmul(&self, out: &mut Buffer, w: &Weight, x: &Buffer) {
        let (rows, cols) = (w.shape[0], w.shape[1]);
        let n = x.len / cols;
        assert_eq!(x.len, n * cols);
        assert_eq!(out.len, n * rows);
        assert!(cols % 8 == 0);
        let params = MatmulParams {
            rows: rows as u32,
            cols: cols as u32,
            n: n as u32,
        };
        // Each SIMD group takes one row, or eight rows of ternary weights. A
        // batch of tokens is worth unpacking the ternary weights once for
        // several.
        let (kernel, per_group) = if w.ternary && n > 1 {
            (
                &self.kernels.matmul_ternary_batch,
                self.matmul_rows * TERNARY_ROWS,
            )
        } else if w.ternary {
            (
                &self.kernels.matmul_ternary,
                self.matmul_rows * TERNARY_ROWS,
            )
        } else {
            (&self.kernels.matmul_bf16, self.matmul_rows)
        };
        self.dispatch(
            kernel,
            &[&out.buf, &w.buf, &x.buf],
            &params,
            rows.div_ceil(per_group),
            THREADS,
        );
    }

    fn add(&self, x: &mut Buffer, y: &Buffer) {
        assert_eq!(x.len, y.len);
        let params = LenParams { len: x.len as u32 };
        self.dispatch(
            &self.kernels.add,
            &[&x.buf, &y.buf],
            &params,
            x.len.div_ceil(THREADS),
            THREADS,
        );
    }

    fn rmsnorm(&self, x: &mut Buffer, weight: &Buffer, eps: f32) {
        let dim = weight.len;
        assert_eq!(x.len % dim, 0);
        let params = RmsnormParams {
            dim: dim as u32,
            eps,
        };
        self.dispatch(
            &self.kernels.rmsnorm,
            &[&x.buf, &weight.buf],
            &params,
            x.len / dim,
            THREADS,
        );
    }

    fn l2norm(&self, x: &mut Buffer, dim: usize, eps: f32) {
        assert_eq!(x.len % dim, 0);
        let params = RmsnormParams {
            dim: dim as u32,
            eps,
        };
        self.dispatch(
            &self.kernels.l2norm,
            &[&x.buf],
            &params,
            x.len / dim,
            THREADS,
        );
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
        let groups = (n * n_heads * rot_dim / 2).div_ceil(THREADS);
        self.dispatch(
            &self.kernels.rope,
            &[&x.buf, &table.buf],
            &params,
            groups,
            THREADS,
        );
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
        assert!(head_dim.is_multiple_of(CACHE_BLOCK) && head_dim <= ATTENTION_THREADS);
        // Every token's heads get a row of scores as long as the last token
        // attends over, as many tokens to a dispatch as the scores fit.
        let stride = pos + n;
        assert!(
            n_heads * stride <= SCORES,
            "attention over {stride} positions needs more scores than {SCORES}"
        );
        let per_dispatch = SCORES / (n_heads * stride);
        for first in (0..n).step_by(per_dispatch) {
            let params = AttentionParams {
                n_heads: n_heads as u32,
                head_dim: head_dim as u32,
                n_kv_heads: n_kv_heads as u32,
                pos: pos as u32,
                first: first as u32,
                stride: stride as u32,
            };
            self.dispatch(
                &self.kernels.attention,
                &[
                    &out.buf,
                    &q.buf,
                    &k_cache.quants,
                    &k_cache.scales,
                    &v_cache.quants,
                    &v_cache.scales,
                    &self.scores,
                ],
                &params,
                per_dispatch.min(n - first) * n_heads,
                ATTENTION_THREADS,
            );
        }
    }

    fn silu_mul(&self, gate: &mut Buffer, up: &Buffer) {
        assert_eq!(gate.len, up.len);
        let params = LenParams {
            len: gate.len as u32,
        };
        let groups = gate.len.div_ceil(THREADS);
        self.dispatch(
            &self.kernels.silu_mul,
            &[&gate.buf, &up.buf],
            &params,
            groups,
            THREADS,
        );
    }

    fn sigmoid_mul(&self, x: &mut Buffer, gate: &Buffer) {
        assert_eq!(x.len, gate.len);
        let params = LenParams { len: x.len as u32 };
        let groups = x.len.div_ceil(THREADS);
        self.dispatch(
            &self.kernels.sigmoid_mul,
            &[&x.buf, &gate.buf],
            &params,
            groups,
            THREADS,
        );
    }

    fn hadamard(&self, x: &mut Buffer, signs: &Buffer, inverse: bool) {
        let width = signs.len;
        assert!(x.len.is_multiple_of(width) && width.is_multiple_of(HADAMARD_BLOCK));
        let params = HadamardParams {
            width: width as u32,
            inverse: inverse as u32,
        };
        self.dispatch(
            &self.kernels.hadamard,
            &[&x.buf, &signs.buf],
            &params,
            x.len / HADAMARD_BLOCK,
            THREADS,
        );
    }

    fn rmsnorm_hadamard(
        &self,
        out: &mut Buffer,
        x: &Buffer,
        weight: &Buffer,
        signs: &Buffer,
        eps: f32,
    ) {
        let width = signs.len;
        assert!(x.len.is_multiple_of(width) && width.is_multiple_of(HADAMARD_BLOCK));
        assert!(out.len == x.len && weight.len == width);
        let params = RmsnormParams {
            dim: width as u32,
            eps,
        };
        self.dispatch(
            &self.kernels.rmsnorm_hadamard,
            &[&out.buf, &x.buf, &weight.buf, &signs.buf],
            &params,
            x.len / HADAMARD_BLOCK,
            THREADS,
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
        let buffers: [&ProtocolObject<dyn MTLBuffer>; 7] = [
            &q.buf,
            &k.buf,
            &v.buf,
            &state_out.buf,
            &x.buf,
            &state.buf,
            &weight.buf,
        ];
        self.dispatch(
            &self.kernels.conv,
            &buffers,
            &params,
            (n * channels).div_ceil(THREADS),
            THREADS,
        );
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
        let buffers: [&ProtocolObject<dyn MTLBuffer>; 7] = [
            &out.buf, &q.buf, &k.buf, &v.buf, &gates.buf, &decay.buf, &state.buf,
        ];
        self.dispatch(
            &self.kernels.delta_net,
            &buffers,
            &params,
            n_v_heads,
            THREADS,
        );
    }
}

#[cfg(test)]
mod tests {
    check_against_cpu!(super::Metal::new());
}
