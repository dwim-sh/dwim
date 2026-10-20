//! The model on a GPU, through Metal: every operation is a hand-written
//! kernel in `kernels/`, in the Metal Shading Language, compiled for the GPU
//! when the device opens.
//!
//! As with Vulkan, operations are encoded into a command buffer as they are
//! called, and committed every so many of them without waiting, so that the
//! GPU works on the first kernels of a forward pass while the CPU encodes
//! the rest; reading a buffer back waits for everything committed.
//! Buffers live in memory the CPU shares with the GPU, so reading and
//! writing them needs no staging, only waiting for the commands before.

use std::{cell::RefCell, error::Error, ptr::NonNull, slice};

use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLGPUFamily, MTLLanguageVersion, MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::{CACHE_BLOCK, CONV_KERNEL, Device, HADAMARD_BLOCK, Tensor};

/// Positions one threadgroup of the attention kernel takes.
const CHUNK: usize = 128;

/// Fewest tokens that go through the flash attention kernel, the positions
/// and the rows (a token's query head each) a threadgroup of it takes.
const FLASH_TOKENS: usize = 2;
const FLASH_CHUNK: usize = 512;
const FLASH_ROWS: usize = 32;

/// Kernels encoded into a command buffer before it is committed, without
/// waiting, so that the GPU works on the first kernels of a forward pass
/// while the CPU encodes the rest.
const SUBMIT_EVERY: usize = 128;

/// Threads in a threadgroup of every kernel but the matmuls' and
/// attention's.
const THREADS: usize = 256;

/// Threads in a threadgroup of the kernels on the matrix units: the tile
/// kernels and flash attention, four SIMD groups.
const MATRIX_THREADS: usize = 128;

/// Threads of a threadgroup of the sums kernel, the rows each takes, and
/// the threadgroups it splits a matrix's blocks to reach.
const SUMS_THREADS: usize = 256;
const SUMS_PER: usize = 4;
const SUMS_GROUPS: usize = 40;
/// Most tokens that go through the sums kernel rather than the tile kernel:
/// the threadgroups of a few tokens read the weights from the cache one
/// after another, so that eight cost about what a tile of them does.
const SUMS_TOKENS: usize = 8;

/// Most floats of partial results the split matmul kernels or the attention
/// kernel write, a chunk's maximum, sum, and weighted values per head of
/// each token for attention: the size of the partials buffer. Longer
/// batches of attention are dispatched a few tokens at a time.
const PARTIALS: usize = 16 << 20;

/// Rows of a tile of the tile kernel, and the most half floats a batch
/// through it packs its activations into: the size of the buffer.
const TILE_ROWS: usize = 64;
const PACKED: usize = 4 << 20;

/// Tokens a SIMD group of the bf16 matmul kernel takes at once.
const BF16_TOKENS: usize = 8;

/// Rows and tokens of a tile of the kernel on the matrix accelerators.
const MMA_ROWS: usize = 64;
const MMA_TOKENS: usize = 64;

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
    matmul_ternary_sums: Object<dyn MTLComputePipelineState>,
    matmul_ternary_reduce: Object<dyn MTLComputePipelineState>,
    /// The tile kernel on the matrix accelerators, where the GPU has them.
    matmul_ternary_mma: Option<Object<dyn MTLComputePipelineState>>,
    /// The tile kernel for tiles of 16, 32 and 64 tokens.
    matmul_ternary_tile: [Object<dyn MTLComputePipelineState>; 3],
    pack_halves: Object<dyn MTLComputePipelineState>,
    add: Object<dyn MTLComputePipelineState>,
    rmsnorm: Object<dyn MTLComputePipelineState>,
    l2norm: Object<dyn MTLComputePipelineState>,
    rope: Object<dyn MTLComputePipelineState>,
    attention: Object<dyn MTLComputePipelineState>,
    attention_combine: Object<dyn MTLComputePipelineState>,
    /// The flash attention kernel for heads of 128 and of 256.
    attention_flash: [Object<dyn MTLComputePipelineState>; 2],
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
    /// Kernels encoded into it.
    dispatched: usize,
}

pub struct Metal {
    device: Object<dyn MTLDevice>,
    queue: Object<dyn MTLCommandQueue>,
    kernels: Kernels,
    partials: Object<dyn MTLBuffer>,
    /// A batch's activations as half floats, for the tile kernels.
    packed: Object<dyn MTLBuffer>,
    /// Rows of the bf16 weights each threadgroup of their matmul takes: one
    /// per SIMD group.
    matmul_rows: usize,
    name: String,
    pending: RefCell<Option<Pending>>,
    /// Command buffers committed and not yet waited for, in order.
    committed: RefCell<Vec<Object<dyn MTLCommandBuffer>>>,
}

// Commands are only encoded from whichever thread owns the device.
unsafe impl Send for Metal {}

impl Metal {
    /// Opens the system's default GPU.
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let name = device.name().to_string();
        let queue = device.newCommandQueue().ok_or("no Metal command queue")?;

        let compile = |name: &str,
                       source: &str,
                       options: Option<&MTLCompileOptions>|
         -> Result<Object<dyn MTLComputePipelineState>, Box<dyn Error>> {
            let library = device
                .newLibraryWithSource_options_error(&NSString::from_str(source), options)
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
        let kernel = |name: &str, source: &str| compile(name, source, None);
        macro_rules! msl {
            ($name:literal) => {
                kernel($name, include_str!(concat!("kernels/", $name, ".metal")))?
            };
            // A kernel compiled with the definitions given ahead of it.
            ($name:literal, $($define:literal = $value:expr),+) => {
                kernel(
                    $name,
                    &format!(
                        concat!($("#define ", $define, " {}\n"),+, "{}"),
                        $($value),+,
                        include_str!(concat!("kernels/", $name, ".metal"))
                    ),
                )?
            };
        }
        // The matrix accelerators of an Apple GPU of the tenth family, which
        // Metal 4's tensor operations run on; a GPU without them takes the
        // tile kernel instead.
        // A system whose compiler lacks them takes it too.
        let matmul_ternary_mma = if device.supportsFamily(MTLGPUFamily::Apple10) {
            let metal4 = MTLCompileOptions::new();
            metal4.setLanguageVersion(MTLLanguageVersion::Version4_0);
            let source = include_str!("kernels/matmul_ternary_mma.metal");
            compile("matmul_ternary_mma", source, Some(&metal4)).ok()
        } else {
            None
        };
        let kernels = Kernels {
            matmul_bf16: msl!("matmul_bf16"),
            matmul_ternary_sums: msl!(
                "matmul_ternary_sums",
                "THREADS" = SUMS_THREADS,
                "PER" = SUMS_PER
            ),
            matmul_ternary_reduce: msl!("matmul_ternary_reduce"),
            matmul_ternary_tile: [
                msl!("matmul_ternary_tile", "TOKENS" = 16),
                msl!("matmul_ternary_tile", "TOKENS" = 32),
                msl!("matmul_ternary_tile", "TOKENS" = 64),
            ],
            matmul_ternary_mma,
            pack_halves: msl!("pack_halves"),
            add: msl!("add"),
            rmsnorm: msl!("rmsnorm"),
            l2norm: msl!("l2norm"),
            rope: msl!("rope"),
            attention: msl!("attention"),
            attention_combine: msl!("attention_combine"),
            attention_flash: [
                msl!("attention_flash", "DIM" = 128),
                msl!("attention_flash", "DIM" = 256),
            ],
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
        let partials = device
            .newBufferWithLength_options(PARTIALS * 4, MTLResourceOptions::StorageModePrivate)
            .ok_or("out of GPU memory")?;
        let packed = device
            .newBufferWithLength_options(PACKED * 2, MTLResourceOptions::StorageModePrivate)
            .ok_or("out of GPU memory")?;
        Ok(Self {
            device,
            queue,
            kernels,
            partials,
            packed,
            matmul_rows,
            name,
            pending: RefCell::new(None),
            committed: RefCell::new(Vec::new()),
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
        self.dispatch_grid(kernel, buffers, params, (groups, 1), threads);
    }

    /// Encodes a kernel as `dispatch` does, over a grid of threadgroups
    /// two deep.
    fn dispatch_grid<P: Copy>(
        &self,
        kernel: &ProtocolObject<dyn MTLComputePipelineState>,
        buffers: &[&ProtocolObject<dyn MTLBuffer>],
        params: &P,
        groups: (usize, usize),
        threads: usize,
    ) {
        let mut pending = self.pending.borrow_mut();
        let current = pending.get_or_insert_with(|| self.begin());
        let encoder = &current.encoder;
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
        let size = |width, height| MTLSize {
            width,
            height,
            depth: 1,
        };
        encoder
            .dispatchThreadgroups_threadsPerThreadgroup(size(groups.0, groups.1), size(threads, 1));
        // Enough encoded for the GPU to get going on while the rest is.
        current.dispatched += 1;
        if current.dispatched >= SUBMIT_EVERY {
            let Pending { cmd, encoder, .. } = pending.take().unwrap();
            encoder.endEncoding();
            cmd.commit();
            self.committed.borrow_mut().push(cmd);
        }
    }

    /// Packs the `n` tokens of `cols` activations of `x` as half floats, with
    /// zeros for the tokens up to `padded`, for the tile kernels.
    fn pack(&self, x: &Buffer, n: usize, cols: usize, padded: usize) {
        assert!(
            padded * cols <= PACKED,
            "a batch of {n} tokens of {cols} is more than the kernels pack"
        );
        let pack = PackParams {
            n: n as u32,
            cols: cols as u32,
            padded: padded as u32,
        };
        self.dispatch(
            &self.kernels.pack_halves,
            &[&self.packed, &x.buf],
            &pack,
            (padded * cols / 4).div_ceil(THREADS),
            THREADS,
        );
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
            Pending {
                cmd,
                encoder,
                dispatched: 0,
            }
        })
    }

    /// Commits everything encoded and waits for it to finish.
    fn flush(&self) {
        let mut committed = self.committed.borrow_mut();
        if let Some(Pending { cmd, encoder, .. }) = self.pending.borrow_mut().take() {
            encoder.endEncoding();
            cmd.commit();
            committed.push(cmd);
        }
        for cmd in committed.drain(..) {
            cmd.waitUntilCompleted();
            if let Some(e) = cmd.error() {
                panic!("GPU commands failed: {}", e.localizedDescription());
            }
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
    /// Splits of the columns, for the kernels that take them.
    splits: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PackParams {
    n: u32,
    cols: u32,
    padded: u32,
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
    chunks: u32,
    /// Positions to a chunk.
    chunk: u32,
    /// Tokens the dispatch covers, for the flash kernel.
    tokens: u32,
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

    fn memory(&self) -> Option<u64> {
        // Metal's estimate of the most the GPU can use without hurting
        // performance: with memory unified, a share of the host's.
        let allocated = self.device.currentAllocatedSize() as u64;
        Some(
            self.device
                .recommendedMaxWorkingSetSize()
                .saturating_sub(allocated),
        )
    }

    fn upload(&self, tensor: Tensor) -> Weight {
        // Ternary blocks go block-major, so that the rows a SIMD group
        // takes read adjacent blocks.
        let block_major;
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
                block_major = crate::ternary::block_major(data, shape[0], shape[1]);
                (shape, block_major.as_slice(), true)
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
        let mut params = MatmulParams {
            rows: rows as u32,
            cols: cols as u32,
            n: n as u32,
            splits: 1,
        };
        // A token or a few through ternary weights look each byte up in
        // tables of their activations' sums, each token reading the weights
        // for itself, with the blocks of a matrix of few rows split among
        // threadgroups whose partial sums are added up after.
        if w.ternary && n <= SUMS_TOKENS {
            let tiles = rows.div_ceil(SUMS_THREADS * SUMS_PER);
            let splits = (SUMS_GROUPS / tiles).clamp(1, cols / 128);
            let splits = if splits * n * rows <= PARTIALS {
                splits
            } else {
                1
            };
            params.splits = splits as u32;
            self.dispatch_grid(
                &self.kernels.matmul_ternary_sums,
                &[&out.buf, &w.buf, &x.buf, &self.partials],
                &params,
                (tiles * splits, n),
                SUMS_THREADS,
            );
            if splits > 1 {
                self.dispatch(
                    &self.kernels.matmul_ternary_reduce,
                    &[&out.buf, &self.partials],
                    &params,
                    (n * rows).div_ceil(THREADS),
                    THREADS,
                );
            }
            return;
        }
        // A batch through ternary weights is a tiled matrix product on the
        // SIMD groups' matrix units, over the activations packed as half
        // floats: a threadgroup per tile of rows and tokens.
        if w.ternary
            && let Some(kernel) = &self.kernels.matmul_ternary_mma
        {
            let padded = n.next_multiple_of(MMA_TOKENS);
            self.pack(x, n, cols, padded);
            self.dispatch_grid(
                kernel,
                &[&out.buf, &w.buf, &self.packed],
                &params,
                (rows.div_ceil(MMA_ROWS), padded / MMA_TOKENS),
                MATRIX_THREADS,
            );
            return;
        }
        if w.ternary {
            // The smallest tile that takes the batch, or the largest.
            let which = match n {
                ..=16 => 0,
                17..=32 => 1,
                _ => 2,
            };
            let tile_tokens = 16 << which;
            let padded = n.next_multiple_of(tile_tokens);
            self.pack(x, n, cols, padded);
            self.dispatch_grid(
                &self.kernels.matmul_ternary_tile[which],
                &[&out.buf, &w.buf, &self.packed],
                &params,
                (rows.div_ceil(TILE_ROWS), padded / tile_tokens),
                MATRIX_THREADS,
            );
            return;
        }
        // Each SIMD group takes one row of bf16 weights for eight tokens.
        self.dispatch_grid(
            &self.kernels.matmul_bf16,
            &[&out.buf, &w.buf, &x.buf],
            &params,
            (rows.div_ceil(self.matmul_rows), n.div_ceil(BF16_TOKENS)),
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
        assert!(
            dim.is_multiple_of(4) && dim <= 8192,
            "the kernel takes rows of up to 8192 in fours"
        );
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
        assert!(
            n_heads / n_kv_heads <= 8,
            "the kernel holds up to eight query heads per key/value head"
        );
        // A batch of tokens goes through the flash kernel, on the matrix
        // units, where the head is one it is compiled for.
        let flash = match head_dim {
            128 if n >= FLASH_TOKENS => Some(&self.kernels.attention_flash[0]),
            256 if n >= FLASH_TOKENS => Some(&self.kernels.attention_flash[1]),
            _ => None,
        };
        let chunk = if flash.is_some() { FLASH_CHUNK } else { CHUNK };
        // Every token's heads get as many chunks as the last token attends
        // over, as many tokens to a dispatch as the partials fit.
        let chunks = (pos + n).div_ceil(chunk);
        let per_token = n_heads * chunks * (head_dim + 2);
        assert!(
            per_token <= PARTIALS,
            "attention over {} positions needs more partials than {PARTIALS}",
            pos + n
        );
        let per_dispatch = PARTIALS / per_token;
        for first in (0..n).step_by(per_dispatch) {
            let tokens = per_dispatch.min(n - first);
            let params = AttentionParams {
                n_heads: n_heads as u32,
                head_dim: head_dim as u32,
                n_kv_heads: n_kv_heads as u32,
                pos: pos as u32,
                first: first as u32,
                chunks: chunks as u32,
                chunk: chunk as u32,
                tokens: tokens as u32,
            };
            let buffers: [&ProtocolObject<dyn MTLBuffer>; 7] = [
                &out.buf,
                &q.buf,
                &k_cache.quants,
                &k_cache.scales,
                &v_cache.quants,
                &v_cache.scales,
                &self.partials,
            ];
            if let Some(kernel) = flash {
                let rows = tokens * (n_heads / n_kv_heads);
                self.dispatch_grid(
                    kernel,
                    &buffers,
                    &params,
                    (rows.div_ceil(FLASH_ROWS), n_kv_heads * chunks),
                    MATRIX_THREADS,
                );
            } else {
                self.dispatch(
                    &self.kernels.attention,
                    &buffers,
                    &params,
                    tokens * n_kv_heads * chunks,
                    ATTENTION_THREADS,
                );
            }
            if chunks > 1 {
                self.dispatch(
                    &self.kernels.attention_combine,
                    &[&out.buf, &self.partials],
                    &params,
                    tokens * n_heads,
                    ATTENTION_THREADS,
                );
            }
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
        assert!(width <= 8192, "the kernel takes rows of up to 8192");
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

    /// Checks the tile kernel a GPU without matrix accelerators takes, on
    /// one that has them too.
    #[test]
    fn ternary_matmul_without_accelerators_matches_cpu() {
        match super::Metal::new() {
            Ok(mut gpu) => {
                gpu.kernels.matmul_ternary_mma = None;
                crate::tests::ternary_matmul_matches_cpu(&gpu);
            }
            Err(e) => eprintln!("skipping: {e}"),
        }
    }

    /// Times the ternary matmul on the model's shapes; run with `--ignored
    /// --nocapture`.
    #[test]
    #[ignore]
    fn ternary_matmul_speed() {
        match super::Metal::new() {
            Ok(gpu) => crate::tests::ternary_matmul_speed(&gpu),
            Err(e) => eprintln!("skipping: {e}"),
        }
    }

    /// Times the kernels at the model's shapes, for one token and for a
    /// batch of 64; run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn kernel_speed() {
        match super::Metal::new() {
            Ok(gpu) => {
                for n in [1, 64] {
                    eprintln!("batch of {n}:");
                    crate::tests::kernel_speed(&gpu, n);
                }
            }
            Err(e) => eprintln!("skipping: {e}"),
        }
    }

    /// Times the cost of a dispatch; run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn dispatch_overhead() {
        match super::Metal::new() {
            Ok(gpu) => crate::tests::dispatch_overhead(&gpu),
            Err(e) => eprintln!("skipping: {e}"),
        }
    }
}
