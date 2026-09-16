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
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::{Device, Tensor};

/// Most tokens one embedding lookup can take: the size of the tokens buffer.
const MAX_TOKENS: usize = 1024;

/// Most attention scores one dispatch of the attention kernel computes, one
/// per head of each token for each position it attends to: the size of the
/// scores buffer. Longer batches are dispatched a few tokens at a time.
const SCORES: usize = 16 << 20;

/// Threads in a threadgroup of every kernel but attention's.
const THREADS: usize = 256;

/// Threads in a threadgroup of the attention kernel: one per element of a
/// head, for heads up to this long.
const ATTENTION_THREADS: usize = 128;

type Object<P> = Retained<ProtocolObject<P>>;

/// A vector of f32 activations in memory shared with the GPU.
pub struct Buffer {
    buf: Object<dyn MTLBuffer>,
    len: usize,
    cap: usize,
}

/// A cache of f16 activations in memory shared with the GPU.
pub struct Cache {
    buf: Object<dyn MTLBuffer>,
    len: usize,
}

/// A bf16 weight tensor in memory shared with the GPU.
pub struct Weight {
    buf: Object<dyn MTLBuffer>,
    shape: Vec<usize>,
}

struct Kernels {
    embed: Object<dyn MTLComputePipelineState>,
    matmul: Object<dyn MTLComputePipelineState>,
    add: Object<dyn MTLComputePipelineState>,
    rmsnorm: Object<dyn MTLComputePipelineState>,
    rope: Object<dyn MTLComputePipelineState>,
    attention: Object<dyn MTLComputePipelineState>,
    silu_mul: Object<dyn MTLComputePipelineState>,
    copy: Object<dyn MTLComputePipelineState>,
    store: Object<dyn MTLComputePipelineState>,
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
    tokens: Object<dyn MTLBuffer>,
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

        let kernel = |name: &str, source: &str| -> Result<Object<dyn MTLComputePipelineState>, Box<dyn Error>> {
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
            embed: msl!("embed"),
            matmul: msl!("matmul"),
            add: msl!("add"),
            rmsnorm: msl!("rmsnorm"),
            rope: msl!("rope"),
            attention: msl!("attention"),
            silu_mul: msl!("silu_mul"),
            copy: msl!("copy"),
            store: msl!("store"),
        };
        let matmul_rows = THREADS / kernels.matmul.threadExecutionWidth();
        let tokens = device
            .newBufferWithLength_options(MAX_TOKENS * 4, MTLResourceOptions::StorageModeShared)
            .ok_or("out of GPU memory")?;
        let scores = device
            .newBufferWithLength_options(SCORES * 4, MTLResourceOptions::StorageModePrivate)
            .ok_or("out of GPU memory")?;
        Ok(Self {
            device,
            queue,
            kernels,
            tokens,
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
            encoder.setBytes_length_atIndex(NonNull::from(params).cast(), size_of::<P>(), buffers.len());
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
            let encoder = cmd.computeCommandEncoder().expect("no Metal compute encoder");
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
        assert!(tensor.data.len().is_multiple_of(8), "weights must come in multiples of eight");
        let data = NonNull::from(tensor.data.as_slice()).cast();
        let buf = unsafe {
            self.device
                .newBufferWithBytes_length_options(data, tensor.data.len() * 2, MTLResourceOptions::StorageModeShared)
                .expect("out of GPU memory")
        };
        Weight {
            buf,
            shape: tensor.shape,
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
        Cache {
            buf: self.buffer(len * 2),
            len,
        }
    }

    fn resize(&self, buf: &mut Buffer, len: usize) {
        assert!(len <= buf.cap, "a buffer of {} activations can't hold {len}", buf.cap);
        buf.len = len;
    }

    fn read(&self, buf: &Buffer) -> Vec<f32> {
        self.flush();
        unsafe { slice::from_raw_parts(buf.buf.contents().as_ptr().cast::<f32>(), buf.len).to_vec() }
    }

    fn write(&self, buf: &mut Buffer, data: &[f32]) {
        assert_eq!(buf.len, data.len());
        // The pending commands may use what the buffer holds now.
        self.flush();
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), buf.buf.contents().as_ptr().cast(), data.len()) };
    }

    fn copy(&self, dst: &mut Buffer, dst_offset: usize, src: &Buffer, src_offset: usize, len: usize) {
        assert!(dst_offset + len <= dst.len && src_offset + len <= src.len);
        let params = CopyParams {
            dst_offset: dst_offset as u32,
            src_offset: src_offset as u32,
            len: len as u32,
        };
        let groups = len.div_ceil(THREADS);
        self.dispatch(&self.kernels.copy, &[&dst.buf, &src.buf], &params, groups, THREADS);
    }

    fn store(&self, cache: &mut Cache, offset: usize, src: &Buffer) {
        assert!(offset + src.len <= cache.len);
        let params = StoreParams {
            offset: offset as u32,
            len: src.len as u32,
        };
        let groups = src.len.div_ceil(THREADS);
        self.dispatch(&self.kernels.store, &[&cache.buf, &src.buf], &params, groups, THREADS);
    }

    fn embed(&self, out: &mut Buffer, table: &Weight, tokens: &[u32]) {
        let dim = table.shape[1];
        assert_eq!(out.len, tokens.len() * dim);
        assert!(tokens.len() <= MAX_TOKENS);
        // The tokens buffer is read by the pending commands, which must
        // finish before it is written over.
        self.flush();
        unsafe { std::ptr::copy_nonoverlapping(tokens.as_ptr(), self.tokens.contents().as_ptr().cast(), tokens.len()) };
        let params = EmbedParams {
            dim: dim as u32,
            n: tokens.len() as u32,
        };
        let groups = (tokens.len() * dim).div_ceil(THREADS);
        self.dispatch(&self.kernels.embed, &[&out.buf, &table.buf, &self.tokens], &params, groups, THREADS);
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
        let groups = rows.div_ceil(self.matmul_rows);
        self.dispatch(&self.kernels.matmul, &[&out.buf, &w.buf, &x.buf], &params, groups, THREADS);
    }

    fn add(&self, x: &mut Buffer, y: &Buffer) {
        assert_eq!(x.len, y.len);
        let params = LenParams { len: x.len as u32 };
        self.dispatch(&self.kernels.add, &[&x.buf, &y.buf], &params, x.len.div_ceil(THREADS), THREADS);
    }

    fn rmsnorm(&self, x: &mut Buffer, weight: &Weight, eps: f32) {
        let dim = weight.shape[0];
        assert_eq!(x.len % dim, 0);
        let params = RmsnormParams { dim: dim as u32, eps };
        self.dispatch(&self.kernels.rmsnorm, &[&x.buf, &weight.buf], &params, x.len / dim, THREADS);
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
        let groups = (n * n_heads * head_dim / 2).div_ceil(THREADS);
        self.dispatch(&self.kernels.rope, &[&x.buf, &table.buf], &params, groups, THREADS);
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
        assert!(head_dim.is_multiple_of(4) && head_dim <= ATTENTION_THREADS);
        // Every token's heads get a row of scores as long as the last token
        // attends over, as many tokens to a dispatch as the scores fit.
        let stride = pos + n;
        assert!(n_heads * stride <= SCORES, "attention over {stride} positions needs more scores than {SCORES}");
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
                &[&out.buf, &q.buf, &k_cache.buf, &v_cache.buf, &self.scores],
                &params,
                per_dispatch.min(n - first) * n_heads,
                ATTENTION_THREADS,
            );
        }
    }

    fn silu_mul(&self, gate: &mut Buffer, up: &Buffer) {
        assert_eq!(gate.len, up.len);
        let params = LenParams { len: gate.len as u32 };
        let groups = gate.len.div_ceil(THREADS);
        self.dispatch(&self.kernels.silu_mul, &[&gate.buf, &up.buf], &params, groups, THREADS);
    }
}

#[cfg(test)]
mod tests {
    check_against_cpu!(super::Metal::new());
}
