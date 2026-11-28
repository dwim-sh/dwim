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

use std::{
    cell::{Cell, RefCell},
    error::Error,
    ptr::NonNull,
    slice,
};

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

use crate::{
    gpu::{CACHE_BLOCK, CONV_KERNEL, Device, Tensor, quant},
    sys::memory,
};

/// Positions one threadgroup of the attention kernel takes.
const CHUNK: usize = 128;

/// Fewest tokens that go through the flash attention kernel, the positions
/// and the rows (a token's query head each) a threadgroup of it takes.
const FLASH_TOKENS: usize = 2;
const FLASH_CHUNK: usize = 512;
const FLASH_ROWS: usize = 32;

/// Rows a threadgroup of the attention kernel on the matrix accelerators
/// takes, and its threads, eight SIMD groups.
const MMA_ROWS: usize = 64;
const MMA_THREADS: usize = 256;
/// Most chunks of positions it splits a context into.
const MMA_CHUNKS: usize = 32;

/// Rows of the threadgroups of the accelerators' attention for a token
/// decoded at a time, and their threads: a token has as many rows as a
/// key/value head has query heads, six for the model, so that a tile of
/// 64 is mostly padding the accelerators multiply for nothing.
const DECODE_ROWS: usize = 16;
const DECODE_THREADS: usize = 128;

/// Positions of context from which a token decoded at a time takes the
/// accelerators' attention rather than `attention.metal`, whose own loop
/// reads the caches at under half the rate the accelerators' does at 8k
/// positions and more, and is no slower at 1k.
const DECODE_DEPTH: usize = 2048;

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

/// Most floats of partial results the attention kernel writes, a chunk's
/// maximum, sum, and weighted values per head of each token: the size of
/// the partials buffer. Longer batches of attention are dispatched a few
/// tokens at a time.
const PARTIALS: usize = 16 << 20;

/// Rows of a tile of the tile kernels, the tokens of their three tiles,
/// and the most half floats a batch through them packs its activations
/// into: the size of the buffer.
const TILE_ROWS: usize = 64;
const TILE_TOKENS: [usize; 3] = [16, 32, 64];
const PACKED: usize = 4 << 20;

/// Tokens a SIMD group of the bf16 matmul kernel takes at once.
const BF16_TOKENS: usize = 8;

/// Most tokens that go through the block-quantized row kernels rather than
/// their tile kernels: the row kernels read the weights once for eight
/// tokens but are bound by their arithmetic past two, so that on an M5
/// four tokens through them cost what a tile of 64 does; and the columns
/// a step of the tile kernels takes, which a matrix must be a whole number
/// of.
const QUANT_TOKENS: usize = 4;
const QUANT_COLS: usize = 128;

/// Fewest tokens of a batch whose experts are multiplied in tiles on the
/// matrix accelerators, an expert at a time, and the pairs of a token and
/// an expert a tile takes. Below it a pair at a time is faster: it reads
/// fewer of the experts' weights than unpacking each expert's matrix for
/// a tile, and does not wait for the router's picks, which the tiles are
/// laid out from on the host each time a layer is run. Most batches being
/// verified, a token and the drafts after it, are below it.
const EXPERT_TOKENS: usize = 7;
const EXPERT_TILE: usize = 16;

/// Threads in a threadgroup of the attention kernel: one per element of a
/// head, for heads up to this long.
const ATTENTION_THREADS: usize = 256;

type Object<P> = Retained<ProtocolObject<P>>;

/// A vector of f32 activations in memory shared with the GPU.
pub struct Buffer {
    buf: Object<dyn MTLBuffer>,
    len: usize,
    cap: usize,
    /// Which buffer this is, of all the device has allocated, and how many
    /// times it has been written: together, what it holds.
    id: u64,
    version: u64,
}

impl Buffer {
    /// Notes that the buffer is written by the operation being encoded.
    fn touch(&mut self) {
        self.version += 1;
    }
}

/// What the packed activations are of: a buffer's contents, and the batch
/// they were packed as.
#[derive(Clone, Copy, PartialEq)]
struct Packed {
    id: u64,
    version: u64,
    n: usize,
    cols: usize,
    padded: usize,
}

/// A cache of activations in memory shared with the GPU, in blocks of 8-bit
/// integers, and the blocks' f16 scales.
pub struct Cache {
    quants: Object<dyn MTLBuffer>,
    scales: Object<dyn MTLBuffer>,
    len: usize,
}

/// A weight matrix in memory shared with the GPU: its weights, and for
/// the block-quantized formats their scales in a buffer of their own.
pub struct Weight {
    buf: Object<dyn MTLBuffer>,
    scales: Option<Object<dyn MTLBuffer>>,
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

/// The index of a block-quantized format's kernels in their arrays.
fn quant_index(format: Format) -> usize {
    match format {
        Format::Q4_0 => 0,
        Format::Q8_0 => 1,
        Format::Bf16 => unreachable!("not block-quantized"),
        Format::Nvfp4 => unreachable!("only the experts' matrices are NVFP4"),
    }
}

struct Kernels {
    matmul_bf16: Object<dyn MTLComputePipelineState>,
    pack_halves: Object<dyn MTLComputePipelineState>,
    /// The row kernels for Q4_0 and Q8_0 weights, in that order, as
    /// [`quant_index`] gives it; their tile kernels on the matrix
    /// accelerators, where the GPU has them, for tiles of 16, 32 and 64
    /// tokens; and their tile kernels on the SIMD groups' matrices for the
    /// same tiles.
    matmul_quant: [Object<dyn MTLComputePipelineState>; 2],
    matmul_quant_mma: Option<[[Object<dyn MTLComputePipelineState>; 3]; 2]>,
    matmul_quant_tile: [[Object<dyn MTLComputePipelineState>; 3]; 2],
    add: Object<dyn MTLComputePipelineState>,
    rmsnorm: Object<dyn MTLComputePipelineState>,
    l2norm: Object<dyn MTLComputePipelineState>,
    rope: Object<dyn MTLComputePipelineState>,
    attention: Object<dyn MTLComputePipelineState>,
    attention_combine: Object<dyn MTLComputePipelineState>,
    /// The flash attention kernel for heads of 128 and of 256.
    attention_flash: [Object<dyn MTLComputePipelineState>; 2],
    /// Batched attention on the matrix accelerators, where the GPU has them,
    /// and the packing of the queries for it.
    /// For heads of 128 and 256, in tiles of [`MMA_ROWS`] rows, and in
    /// tiles of [`DECODE_ROWS`] for a token decoded at a time.
    attention_mma: Option<[Object<dyn MTLComputePipelineState>; 4]>,
    pack_queries: Object<dyn MTLComputePipelineState>,
    silu_mul: Object<dyn MTLComputePipelineState>,
    sigmoid_mul: Object<dyn MTLComputePipelineState>,
    copy: Object<dyn MTLComputePipelineState>,
    store_q8: Object<dyn MTLComputePipelineState>,
    conv: Object<dyn MTLComputePipelineState>,
    delta_net: Object<dyn MTLComputePipelineState>,
    /// The experts' matrix multiplication over NVFP4 and over Q8_0 weights,
    /// a pair of a token and an expert at a time; and for a batch, in
    /// tiles on the matrix accelerators, where the GPU has them, with the
    /// packing of the tiles' inputs and the putting back of their results.
    expert_matmul: [Object<dyn MTLComputePipelineState>; 2],
    expert_matmul_mma: Option<[Object<dyn MTLComputePipelineState>; 2]>,
    expert_pack: Object<dyn MTLComputePipelineState>,
    expert_scatter: Object<dyn MTLComputePipelineState>,
    mix: Object<dyn MTLComputePipelineState>,
    gated_update: Object<dyn MTLComputePipelineState>,
    gate: Object<dyn MTLComputePipelineState>,
    short_conv: Object<dyn MTLComputePipelineState>,
    short_conv_state: Object<dyn MTLComputePipelineState>,
    clip: Object<dyn MTLComputePipelineState>,
    scale_rows: Object<dyn MTLComputePipelineState>,
    block_conv: Object<dyn MTLComputePipelineState>,
    /// Attention with a bias by distance and a sink: a token at a time,
    /// the merging of its chunks, and for heads of 128 the kernel on the
    /// matrix accelerators, where the GPU has them, in tiles of
    /// [`MMA_ROWS`] rows for a batch and of [`DECODE_ROWS`] for a token
    /// deep into a context.
    attention_rel: Object<dyn MTLComputePipelineState>,
    attention_rel_combine: Object<dyn MTLComputePipelineState>,
    attention_rel_mma: Option<[Object<dyn MTLComputePipelineState>; 2]>,
    route: Object<dyn MTLComputePipelineState>,
    situ: Object<dyn MTLComputePipelineState>,
    weighted_sum: Object<dyn MTLComputePipelineState>,
    /// The copy of a cache's quants, four bytes at a time, and of its
    /// scales.
    copy_words: [Object<dyn MTLComputePipelineState>; 2],
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
    /// A batch's activations as half floats, for the tile kernels, and
    /// what they are of, so that the matmuls over one buffer pack it once.
    packed: Object<dyn MTLBuffer>,
    packed_for: Cell<Option<Packed>>,
    /// Buffers allocated so far, for their ids.
    allocated: Cell<u64>,
    /// Rows of the bf16 weights each threadgroup of their matmul takes: one
    /// per SIMD group.
    matmul_rows: usize,
    name: String,
    /// What a batch's experts are multiplied in tiles with.
    experts: RefCell<Experts>,
    pending: RefCell<Option<Pending>>,
    /// Command buffers committed and not yet waited for, in order.
    committed: RefCell<Vec<Object<dyn MTLCommandBuffer>>>,
}

/// What a batch's experts are multiplied in tiles with: the pairs of a
/// token and an expert in the order of the tiles, an expert's together;
/// the tiles, each its expert, its first pair in that order and how many
/// it has; the inputs packed in that order; and the results in it. The
/// order and the tiles are laid out on the host from the experts the
/// router picked, once for both of a layer's multiplications.
#[derive(Default)]
struct Experts {
    order: Scratch,
    tiles: Scratch,
    packed: Scratch,
    sorted: Scratch,
    /// The buffer of picks the order and the tiles are of, as it was, and
    /// how many tiles there are.
    planned: Option<(u64, u64)>,
    count: usize,
}

/// A buffer that is made, and made again longer, as it is needed.
#[derive(Default)]
struct Scratch(Option<(Object<dyn MTLBuffer>, usize)>);

// Commands are only encoded from whichever thread owns the device.
unsafe impl Send for Metal {}

/// The source of the kernel `$name` in `kernels/`.
macro_rules! source {
    ($name:literal) => {
        include_str!(concat!("kernels/", $name, ".metal"))
    };
}

/// Compiles the kernel `name` for threadgroups of `threads`, which the GPU
/// must be able to run, from `defines` ahead of `parts`: the snippets it
/// shares with other kernels, then its own source.
fn compile(
    device: &ProtocolObject<dyn MTLDevice>,
    name: &str,
    threads: usize,
    defines: &[(&str, usize)],
    parts: &[&str],
    options: Option<&MTLCompileOptions>,
) -> Result<Object<dyn MTLComputePipelineState>, Box<dyn Error>> {
    let mut source = String::new();
    for (define, value) in defines {
        source.push_str(&format!("#define {define} {value}\n"));
    }
    for part in parts {
        source.push_str(part);
        source.push('\n');
    }
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(&source), options)
        .map_err(|e| format!("{name}.metal: {}", e.localizedDescription()))?;
    let function = library
        .newFunctionWithName(&NSString::from_str(name))
        .ok_or_else(|| format!("{name}.metal has no kernel named {name}"))?;
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| format!("{name}: {}", e.localizedDescription()))?;
    if pipeline.maxTotalThreadsPerThreadgroup() < threads {
        return Err(format!("{name} can't run {threads} threads to a threadgroup").into());
    }
    Ok(pipeline)
}

impl Metal {
    /// Opens the system's default GPU.
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let name = device.name().to_string();
        let queue = device.newCommandQueue().ok_or("no Metal command queue")?;

        // A kernel from `kernels/`, with the definitions given after its
        // threads ahead of it, and the snippets given after a semicolon.
        macro_rules! msl {
            ($name:literal, $threads:expr $(, $define:literal = $value:expr)* $(; $($snippet:literal),+)?) => {
                compile(
                    &device,
                    $name,
                    $threads,
                    &[$(($define, $value)),*],
                    &[$($(source!($snippet),)+)? source!($name)],
                    None,
                )?
            };
        }
        // The matrix accelerators of an Apple GPU of the tenth family, which
        // Metal 4's tensor operations run on; a GPU without them takes the
        // tile kernel instead, as does a system without Metal 4. Where both
        // are there, the kernels must compile.
        let matmul_quant_mma = if device.supportsFamily(MTLGPUFamily::Apple10)
            && device.supportsFamily(MTLGPUFamily::Metal4)
        {
            let metal4 = MTLCompileOptions::new();
            metal4.setLanguageVersion(MTLLanguageVersion::Version4_0);
            let quant = |bits, tokens| {
                compile(
                    &device,
                    "matmul_quant_mma",
                    MATRIX_THREADS,
                    &[("BITS", bits), ("TOKENS", tokens)],
                    &[source!("matmul_quant_mma")],
                    Some(&metal4),
                )
            };
            Some([
                [quant(4, 16)?, quant(4, 32)?, quant(4, 64)?],
                [quant(8, 16)?, quant(8, 32)?, quant(8, 64)?],
            ])
        } else {
            None
        };
        let expert_matmul_mma = if matmul_quant_mma.is_some() {
            let metal4 = MTLCompileOptions::new();
            metal4.setLanguageVersion(MTLLanguageVersion::Version4_0);
            let tiled = |bits| {
                compile(
                    &device,
                    "expert_matmul_mma",
                    MATRIX_THREADS,
                    &[("BITS", bits), ("TOKENS", EXPERT_TILE)],
                    &[source!("expert_matmul_mma")],
                    Some(&metal4),
                )
            };
            Some([tiled(4)?, tiled(8)?])
        } else {
            None
        };
        let attention_mma = if matmul_quant_mma.is_some() {
            let metal4 = MTLCompileOptions::new();
            metal4.setLanguageVersion(MTLLanguageVersion::Version4_0);
            let mma = |dim, rows, threads| {
                compile(
                    &device,
                    "attention_mma",
                    threads,
                    &[("DIM", dim), ("ROWS", rows), ("THREADS", threads)],
                    &[source!("attention_mma")],
                    Some(&metal4),
                )
            };
            Some([
                mma(128, MMA_ROWS, MMA_THREADS)?,
                mma(256, MMA_ROWS, MMA_THREADS)?,
                mma(128, DECODE_ROWS, DECODE_THREADS)?,
                mma(256, DECODE_ROWS, DECODE_THREADS)?,
            ])
        } else {
            None
        };
        let attention_rel_mma = if matmul_quant_mma.is_some() {
            let metal4 = MTLCompileOptions::new();
            metal4.setLanguageVersion(MTLLanguageVersion::Version4_0);
            let mma = |rows, threads| {
                compile(
                    &device,
                    "attention_rel_mma",
                    threads,
                    &[("DIM", 128), ("ROWS", rows), ("THREADS", threads)],
                    &[source!("attention_rel_mma")],
                    Some(&metal4),
                )
            };
            Some([
                mma(MMA_ROWS, MMA_THREADS)?,
                mma(DECODE_ROWS, DECODE_THREADS)?,
            ])
        } else {
            None
        };
        let kernels = Kernels {
            matmul_bf16: msl!("matmul_bf16", THREADS),
            attention_mma,
            matmul_quant: [msl!("matmul_q4_0", THREADS), msl!("matmul_q8_0", THREADS)],
            matmul_quant_mma,
            matmul_quant_tile: [
                [
                    msl!(
                        "matmul_quant_tile",
                        MATRIX_THREADS,
                        "BITS" = 4,
                        "TOKENS" = 16
                    ),
                    msl!(
                        "matmul_quant_tile",
                        MATRIX_THREADS,
                        "BITS" = 4,
                        "TOKENS" = 32
                    ),
                    msl!(
                        "matmul_quant_tile",
                        MATRIX_THREADS,
                        "BITS" = 4,
                        "TOKENS" = 64
                    ),
                ],
                [
                    msl!(
                        "matmul_quant_tile",
                        MATRIX_THREADS,
                        "BITS" = 8,
                        "TOKENS" = 16
                    ),
                    msl!(
                        "matmul_quant_tile",
                        MATRIX_THREADS,
                        "BITS" = 8,
                        "TOKENS" = 32
                    ),
                    msl!(
                        "matmul_quant_tile",
                        MATRIX_THREADS,
                        "BITS" = 8,
                        "TOKENS" = 64
                    ),
                ],
            ],
            pack_halves: msl!("pack_halves", THREADS),
            pack_queries: msl!("pack_queries", THREADS),
            add: msl!("add", THREADS),
            rmsnorm: msl!("rmsnorm", THREADS),
            l2norm: msl!("l2norm", THREADS),
            rope: msl!("rope", THREADS),
            attention: msl!("attention", ATTENTION_THREADS, "CHUNK" = CHUNK),
            attention_combine: msl!("attention_combine", ATTENTION_THREADS),
            attention_flash: [
                msl!(
                    "attention_flash",
                    MATRIX_THREADS,
                    "DIM" = 128,
                    "CHUNK" = FLASH_CHUNK
                ),
                msl!(
                    "attention_flash",
                    MATRIX_THREADS,
                    "DIM" = 256,
                    "CHUNK" = FLASH_CHUNK
                ),
            ],
            silu_mul: msl!("silu_mul", THREADS),
            sigmoid_mul: msl!("sigmoid_mul", THREADS),
            copy: msl!("copy", THREADS),
            store_q8: msl!("store_q8", THREADS),
            conv: msl!("conv", THREADS),
            delta_net: msl!("delta_net", THREADS),
            expert_matmul: [
                msl!("expert_matmul", THREADS, "BITS" = 4),
                msl!("expert_matmul", THREADS, "BITS" = 8),
            ],
            expert_matmul_mma,
            expert_pack: msl!("expert_pack", THREADS),
            expert_scatter: msl!("expert_scatter", THREADS),
            mix: msl!("mix", THREADS),
            gated_update: msl!("gated_update", THREADS),
            gate: msl!("gate", THREADS),
            short_conv: msl!("short_conv", THREADS),
            short_conv_state: msl!("short_conv_state", THREADS),
            clip: msl!("clip", THREADS),
            scale_rows: msl!("scale_rows", THREADS),
            block_conv: msl!("block_conv", THREADS),
            attention_rel: msl!("attention_rel", ATTENTION_THREADS, "CHUNK" = CHUNK),
            attention_rel_combine: msl!("attention_rel_combine", ATTENTION_THREADS),
            attention_rel_mma,
            route: msl!("route", THREADS),
            situ: msl!("situ", THREADS),
            weighted_sum: msl!("weighted_sum", THREADS),
            copy_words: [
                msl!("copy_words", THREADS, "BYTES" = 4),
                msl!("copy_words", THREADS, "BYTES" = 2),
            ],
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
            packed_for: Cell::new(None),
            allocated: Cell::new(0),
            matmul_rows,
            name,
            experts: RefCell::new(Experts::default()),
            pending: RefCell::new(None),
            committed: RefCell::new(Vec::new()),
        })
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
        current.dispatched += 1;
        if current.dispatched >= SUBMIT_EVERY {
            let Pending { cmd, encoder, .. } = pending.take().unwrap();
            encoder.endEncoding();
            cmd.commit();
            self.committed.borrow_mut().push(cmd);
        }
    }

    /// Packs the `n` tokens of `cols` activations of `x` as half floats, with
    /// zeros for the tokens up to `padded`, for the tile kernels, unless
    /// they are what is packed already.
    fn pack(&self, x: &Buffer, n: usize, cols: usize, padded: usize) {
        assert!(
            padded * cols <= PACKED,
            "a batch of {n} tokens of {cols} is more than the kernels pack"
        );
        let packed = Packed {
            id: x.id,
            version: x.version,
            n,
            cols,
            padded,
        };
        if self.packed_for.get() == Some(packed) {
            return;
        }
        self.packed_for.set(Some(packed));
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

    /// The scratch buffer, with room for `bytes`.
    fn scratch<'a>(&self, scratch: &'a mut Scratch, bytes: usize) -> &'a Object<dyn MTLBuffer> {
        if scratch.0.as_ref().is_none_or(|(_, len)| *len < bytes) {
            scratch.0 = Some((self.buffer(bytes), bytes));
        }
        &scratch.0.as_ref().expect("made above").0
    }

    /// Multiplies a batch's pairs of a token and an expert in tiles of one
    /// expert's pairs, with `kernel` on the matrix accelerators: what
    /// [`Device::expert_matmul`] does, reading each expert's matrix once
    /// for a tile rather than once for each token that goes through it.
    #[allow(clippy::too_many_arguments)]
    fn expert_tiles(
        &self,
        kernel: &ProtocolObject<dyn MTLComputePipelineState>,
        out: &mut Buffer,
        w: &Weight,
        scale: &Buffer,
        x: &Buffer,
        ids: &Buffer,
        k: usize,
        shared: bool,
    ) {
        let (pairs, experts) = (ids.len, scale.len);
        let (rows, cols) = (w.shape[0] / experts, w.shape[1]);
        let mut scratch = self.experts.borrow_mut();
        let Experts {
            order,
            tiles,
            packed,
            sorted,
            planned,
            count,
        } = &mut *scratch;
        // The pairs in the order of their experts, and each expert's in
        // tiles: laid out once for the picks, which both of a layer's
        // multiplications are of. Reading the picks waits for the router.
        if *planned != Some((ids.id, ids.version)) {
            let picks: Vec<usize> = self.read(ids).iter().map(|&e| e as usize).collect();
            let mut counts = vec![0u32; experts];
            for &expert in &picks {
                counts[expert] += 1;
            }
            let mut first = vec![0u32; experts];
            for expert in 1..experts {
                first[expert] = first[expert - 1] + counts[expert - 1];
            }
            let mut sorted_pairs = vec![0u32; pairs];
            let mut next = first.clone();
            for (pair, &expert) in picks.iter().enumerate() {
                sorted_pairs[next[expert] as usize] = pair as u32;
                next[expert] += 1;
            }
            let mut laid = Vec::new();
            for expert in 0..experts {
                for at in (0..counts[expert]).step_by(EXPERT_TILE) {
                    let of = (counts[expert] - at).min(EXPERT_TILE as u32);
                    laid.extend([expert as u32, first[expert] + at, of]);
                }
            }
            for (scratch, words) in [(&mut *order, &sorted_pairs), (&mut *tiles, &laid)] {
                let buf = self.scratch(scratch, words.len() * 4);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        words.as_ptr(),
                        buf.contents().as_ptr().cast(),
                        words.len(),
                    )
                };
            }
            *planned = Some((ids.id, ids.version));
            *count = laid.len() / 3;
        }
        // The last tile reaches a whole tile past its first pair.
        let padded = pairs + EXPERT_TILE;
        let order = &order.0.as_ref().expect("laid out above").0;
        let tiles = &tiles.0.as_ref().expect("laid out above").0;
        let packed = self.scratch(packed, padded * cols * 2);
        let sorted = self.scratch(sorted, pairs * rows * 4);
        out.touch();
        let pack = ExpertPackParams {
            pairs: pairs as u32,
            cols: cols as u32,
            padded: padded as u32,
            share: if shared { k as u32 } else { 1 },
        };
        self.dispatch(
            &self.kernels.expert_pack,
            &[packed, &x.buf, order],
            &pack,
            (padded * cols / 4).div_ceil(THREADS),
            THREADS,
        );
        let params = ExpertTileParams {
            rows: rows as u32,
            cols: cols as u32,
            padded: padded as u32,
        };
        let scales = w.scales.as_ref().expect("quantized weights have scales");
        self.dispatch_grid(
            kernel,
            &[sorted, &w.buf, scales, packed, tiles, &scale.buf],
            &params,
            (rows.div_ceil(TILE_ROWS), *count),
            MATRIX_THREADS,
        );
        let scatter = ExpertScatterParams {
            pairs: pairs as u32,
            rows: rows as u32,
        };
        self.dispatch(
            &self.kernels.expert_scatter,
            &[&out.buf, sorted, order],
            &scatter,
            (pairs * rows / 4).div_ceil(THREADS),
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
    /// Rows of a key/value head's packed queries of the dispatch, for the
    /// kernel on the matrix accelerators.
    padded: u32,
}

/// A kernel a batch of tokens of attention goes through, the rows (a
/// token's query head each) a threadgroup of it takes and its threads, and
/// whether it is the one on the matrix accelerators, which reads the
/// queries packed by key/value head and takes chunks sized by the depth.
struct Batched<'a> {
    kernel: &'a ProtocolObject<dyn MTLComputePipelineState>,
    rows: usize,
    threads: usize,
    accelerators: bool,
}

/// What `pack_queries.metal` is told.
#[repr(C)]
#[derive(Clone, Copy)]
struct PackQueriesParams {
    n_heads: u32,
    n_kv_heads: u32,
    dim: u32,
    /// The first token of the dispatch, and its tokens.
    first: u32,
    n: u32,
    padded: u32,
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

/// What `attention_rel.metal` and the kernel that merges its chunks are
/// told.
#[repr(C)]
#[derive(Clone, Copy)]
struct AttentionRelParams {
    n_heads: u32,
    head_dim: u32,
    n_kv_heads: u32,
    pos: u32,
    first: u32,
    chunks: u32,
    /// Positions to a chunk.
    chunk: u32,
    /// Tokens the dispatch covers, and the rows of a key/value head's
    /// packed queries of it, for the kernel on the matrix accelerators.
    tokens: u32,
    padded: u32,
    window: u32,
    ring: u32,
    n_rel: u32,
    extent: u32,
    /// For that kernel, whose chunks are the dispatch's rather than each
    /// token's: the first position the dispatch attends over.
    base: u32,
    batched: u32,
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

/// What the kernels over rows of a width are told: `scale_rows.metal`,
/// and `weighted_sum.metal`, with how many experts a token has.
#[repr(C)]
#[derive(Clone, Copy)]
struct RowsParams {
    len: u32,
    width: u32,
    k: u32,
}

/// What `block_conv.metal` is told: the rows and their channels, the
/// distances a row is convolved over, the channels to a group, and how
/// long a row of taps is and where the taps to take start in it.
#[repr(C)]
#[derive(Clone, Copy)]
struct BlockConvParams {
    n: u32,
    channels: u32,
    kernel: u32,
    group: u32,
    width: u32,
    from: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ExpertPackParams {
    pairs: u32,
    cols: u32,
    padded: u32,
    share: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ExpertTileParams {
    rows: u32,
    cols: u32,
    padded: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ExpertScatterParams {
    pairs: u32,
    rows: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ExpertParams {
    rows: u32,
    cols: u32,
    k: u32,
    shared: u32,
}

/// The most of the machine's memory a model may take, as a fraction: the
/// weights and caches are wired while the model runs, and the system and
/// the applications beside it need the rest. Nine sixteenths is 18 GiB
/// of a 32 GiB Mac, which gives Qwen3.8-27B, whose weights and states
/// take 16.9 GB before any token of context, some fifty thousand tokens
/// in 19 GB. Half, 16 GiB, held Bonsai with room to spare, but holds this
/// model with none: the context fell to the floor, and the conversation
/// was compacted at its second message. With the kernel's own few
/// gigabytes wired too, this share has left the machine paging while a
/// browser was open; less would leave the model no context to work in.
const SHARE: (u64, u64) = (9, 16);

impl Device for Metal {
    type Buffer = Buffer;
    type Weight = Weight;
    type Cache = Cache;

    fn name(&self) -> String {
        self.name.clone()
    }

    fn memory(&self) -> Option<u64> {
        // Metal's estimate of the most the GPU can use without hurting
        // performance: with memory unified, a share of the host's. It
        // leaves out what other processes hold, so it is no more than the
        // host has available; and since what a model takes is wired, it is
        // no more than [`SHARE`] of the machine, so that the system and
        // whatever else runs keep the rest and nothing is paged out.
        let allocated = self.device.currentAllocatedSize() as u64;
        let mut free = self
            .device
            .recommendedMaxWorkingSetSize()
            .saturating_sub(allocated);
        if let Some(available) = memory::available() {
            free = free.min(available);
        }
        if let Some(total) = memory::total() {
            free = free.min((total / SHARE.1 * SHARE.0).saturating_sub(allocated));
        }
        Some(free)
    }

    fn takes_nvfp4(&self) -> bool {
        true
    }

    fn upload(&self, tensor: Tensor) -> Weight {
        let with_bytes = |bytes: &[u8]| unsafe {
            self.device
                .newBufferWithBytes_length_options(
                    NonNull::from(bytes).cast(),
                    bytes.len(),
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("out of GPU memory")
        };
        // The bytes of a vector of plain integers, in memory order.
        fn bytes<T>(data: &[T]) -> &[u8] {
            unsafe { slice::from_raw_parts(data.as_ptr().cast::<u8>(), size_of_val(data)) }
        }
        let (shape, buf, scales, format) = match &tensor {
            Tensor::Bf16 { shape, data } => {
                assert!(
                    data.len().is_multiple_of(8),
                    "weights must come in multiples of eight"
                );
                (shape, with_bytes(bytes(data)), None, Format::Bf16)
            }
            Tensor::Q4_0 {
                shape,
                scales,
                quants,
            } => {
                assert_eq!(quants.len(), shape[0] * shape[1] / 2);
                assert_eq!(scales.len(), shape[0] * quant::blocks(shape[1]));
                (
                    shape,
                    with_bytes(quants),
                    Some(with_bytes(bytes(scales))),
                    Format::Q4_0,
                )
            }
            Tensor::Q8_0 {
                shape,
                scales,
                quants,
            } => {
                assert_eq!(quants.len(), shape[0] * shape[1]);
                assert_eq!(scales.len(), shape[0] * quant::blocks(shape[1]));
                (
                    shape,
                    with_bytes(bytes(quants)),
                    Some(with_bytes(bytes(scales))),
                    Format::Q8_0,
                )
            }
            Tensor::Nvfp4 {
                shape,
                scales,
                quants,
            } => {
                assert_eq!(quants.len(), shape[0] * shape[1] / 2);
                assert_eq!(scales.len(), shape[0] * shape[1] / quant::NVFP4_BLOCK);
                (
                    shape,
                    with_bytes(quants),
                    Some(with_bytes(scales)),
                    Format::Nvfp4,
                )
            }
        };
        Weight {
            buf,
            scales,
            shape: shape.clone(),
            format,
        }
    }

    fn alloc(&self, len: usize) -> Buffer {
        let id = self.allocated.get();
        self.allocated.set(id + 1);
        Buffer {
            buf: self.buffer(len * 4),
            len,
            cap: len,
            id,
            version: 0,
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
        buf.touch();
    }

    fn read(&self, buf: &Buffer) -> Vec<f32> {
        self.flush();
        unsafe {
            slice::from_raw_parts(buf.buf.contents().as_ptr().cast::<f32>(), buf.len).to_vec()
        }
    }

    fn write(&self, buf: &mut Buffer, data: &[f32]) {
        assert_eq!(buf.len, data.len());
        buf.touch();
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
        dst.touch();
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
        assert!(len <= cache.len && data.len() == crate::gpu::cache_bytes(len));
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
        out.touch();
        let params = MatmulParams {
            rows: rows as u32,
            cols: cols as u32,
            n: n as u32,
            splits: 1,
        };
        // Each SIMD group takes one row of bf16 weights for eight tokens.
        if w.format == Format::Bf16 {
            self.dispatch_grid(
                &self.kernels.matmul_bf16,
                &[&out.buf, &w.buf, &x.buf],
                &params,
                (rows.div_ceil(self.matmul_rows), n.div_ceil(BF16_TOKENS)),
                THREADS,
            );
            return;
        }
        // The block-quantized kernels bind the scales after the quants.
        let which = quant_index(w.format);
        let scales = w
            .scales
            .as_ref()
            .expect("block-quantized weights have scales");
        // A batch through block-quantized weights is a tiled matrix product
        // too, on the accelerators or the SIMD groups' matrix units, over
        // the activations packed as half floats, where the matrix is a whole
        // number of the steps the tile kernels take.
        if n > QUANT_TOKENS && cols.is_multiple_of(QUANT_COLS) {
            let size = TILE_TOKENS
                .iter()
                .position(|&tokens| n <= tokens)
                .unwrap_or(TILE_TOKENS.len() - 1);
            let tile_tokens = TILE_TOKENS[size];
            let padded = n.next_multiple_of(tile_tokens);
            self.pack(x, n, cols, padded);
            let kernel = match &self.kernels.matmul_quant_mma {
                Some(mma) => &mma[which][size],
                None => &self.kernels.matmul_quant_tile[which][size],
            };
            self.dispatch_grid(
                kernel,
                &[&out.buf, &w.buf, scales, &self.packed],
                &params,
                (rows.div_ceil(TILE_ROWS), padded / tile_tokens),
                MATRIX_THREADS,
            );
            return;
        }
        // Each SIMD group takes one row of block-quantized weights, the
        // tokens eight at a time.
        self.dispatch(
            &self.kernels.matmul_quant[which],
            &[&out.buf, &w.buf, scales, &x.buf],
            &params,
            rows.div_ceil(self.matmul_rows),
            THREADS,
        );
    }

    fn add(&self, x: &mut Buffer, y: &Buffer) {
        assert_eq!(x.len, y.len);
        x.touch();
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
        x.touch();
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
        x.touch();
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
        x.touch();
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
        out.touch();
        // A batch of tokens goes through a batched kernel where the head is
        // one they are compiled for: the one on the matrix accelerators
        // where the GPU has them, and the flash kernel on the SIMD groups'
        // matrices elsewhere.
        // A token decoded at a time takes the accelerators too, past a
        // depth of context where reading the caches is the work, in tiles
        // of its own size.
        let decode = n == 1 && pos + 1 >= DECODE_DEPTH && self.kernels.attention_mma.is_some();
        let batched = match head_dim {
            128 if n >= FLASH_TOKENS || decode => Some(0),
            256 if n >= FLASH_TOKENS || decode => Some(1),
            _ => None,
        }
        .map(|which| match &self.kernels.attention_mma {
            Some(mma) if decode => Batched {
                kernel: &mma[2 + which],
                rows: DECODE_ROWS,
                threads: DECODE_THREADS,
                accelerators: true,
            },
            Some(mma) => Batched {
                kernel: &mma[which],
                rows: MMA_ROWS,
                threads: MMA_THREADS,
                accelerators: true,
            },
            None => Batched {
                kernel: &self.kernels.attention_flash[which],
                rows: FLASH_ROWS,
                threads: MATRIX_THREADS,
                accelerators: false,
            },
        });
        let accelerators = batched.as_ref().is_some_and(|b| b.accelerators);
        // The kernel on the accelerators takes chunks that keep a context
        // to [`MMA_CHUNKS`] of them, so that a deep one is not split into
        // dispatches a few tokens each by the partials, nor each token's
        // heads into many partials to combine. A batch of a few tokens, one
        // tile of rows, is left the key/value heads' chunks of
        // threadgroups, 128 for the model's heads: on an M5 at 64k of
        // context, two to eight tokens take no longer so than with four
        // times the threadgroups over a quarter of the chunk.
        let chunk = match &batched {
            Some(_) if accelerators => (pos + n).div_ceil(MMA_CHUNKS).next_multiple_of(FLASH_CHUNK),
            Some(_) => FLASH_CHUNK,
            None => CHUNK,
        };
        // Every token's heads get as many chunks as the last token attends
        // over, as many tokens to a dispatch as the partials fit, and, on
        // the accelerators, as the packed queries fit: each key/value
        // head's rows of the dispatch's tokens, padded to whole tiles.
        let chunks = (pos + n).div_ceil(chunk);
        let per_token = n_heads * chunks * (head_dim + 2);
        assert!(
            per_token <= PARTIALS,
            "attention over {} positions needs more partials than {PARTIALS}",
            pos + n
        );
        let group = n_heads / n_kv_heads;
        let mut per_dispatch = PARTIALS / per_token;
        if accelerators {
            let rows = PACKED / (n_kv_heads * head_dim) / MMA_ROWS * MMA_ROWS;
            assert!(
                rows >= group,
                "the kernels pack fewer queries than {n_heads} heads of {head_dim} take"
            );
            per_dispatch = per_dispatch.min(rows / group);
        }
        let queries: &ProtocolObject<dyn MTLBuffer> =
            if accelerators { &self.packed } else { &q.buf };
        for first in (0..n).step_by(per_dispatch) {
            let tokens = per_dispatch.min(n - first);
            let padded =
                (tokens * group).next_multiple_of(batched.as_ref().map_or(MMA_ROWS, |b| b.rows));
            if accelerators {
                self.packed_for.set(None);
                let pack = PackQueriesParams {
                    n_heads: n_heads as u32,
                    n_kv_heads: n_kv_heads as u32,
                    dim: head_dim as u32,
                    first: first as u32,
                    n: tokens as u32,
                    padded: padded as u32,
                };
                self.dispatch(
                    &self.kernels.pack_queries,
                    &[&self.packed, &q.buf],
                    &pack,
                    (n_kv_heads * padded * head_dim / 4).div_ceil(THREADS),
                    THREADS,
                );
            }
            let params = AttentionParams {
                n_heads: n_heads as u32,
                head_dim: head_dim as u32,
                n_kv_heads: n_kv_heads as u32,
                pos: pos as u32,
                first: first as u32,
                chunks: chunks as u32,
                chunk: chunk as u32,
                tokens: tokens as u32,
                padded: padded as u32,
            };
            let buffers: [&ProtocolObject<dyn MTLBuffer>; 7] = [
                &out.buf,
                queries,
                &k_cache.quants,
                &k_cache.scales,
                &v_cache.quants,
                &v_cache.scales,
                &self.partials,
            ];
            if let Some(b) = &batched {
                let rows = tokens * group;
                self.dispatch_grid(
                    b.kernel,
                    &buffers,
                    &params,
                    (rows.div_ceil(b.rows), n_kv_heads * chunks),
                    b.threads,
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
        gate.touch();
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
        x.touch();
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
        for buf in [&mut *q, &mut *k, &mut *v, &mut *state_out] {
            buf.touch();
        }
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
        out.touch();
        state.touch();
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
        let pairs = ids.len;
        assert_eq!(out.len, pairs * rows);
        let shared = x.len == pairs / k * cols;
        assert!(shared || x.len == pairs * cols);
        let which = match w.format {
            Format::Nvfp4 => 0,
            Format::Q8_0 => 1,
            Format::Bf16 | Format::Q4_0 => {
                panic!("the experts' matrices are NVFP4 or Q8_0")
            }
        };
        // A batch goes expert by expert in tiles on the matrix
        // accelerators, where the matrices are whole steps of the tiles.
        if let Some(tiled) = &self.kernels.expert_matmul_mma
            && pairs >= EXPERT_TOKENS * k
            && cols.is_multiple_of(128)
            && rows.is_multiple_of(4)
        {
            self.expert_tiles(&tiled[which], out, w, scale, x, ids, k, shared);
            return;
        }
        let kernel = &self.kernels.expert_matmul[which];
        let scales = w.scales.as_ref().expect("quantized weights have scales");
        out.touch();
        let params = ExpertParams {
            rows: rows as u32,
            cols: cols as u32,
            k: k as u32,
            shared: shared as u32,
        };
        self.dispatch_grid(
            kernel,
            &[&out.buf, &w.buf, scales, &x.buf, &ids.buf, &scale.buf],
            &params,
            (rows.div_ceil(self.matmul_rows), pairs),
            THREADS,
        );
    }

    fn copy_cache(&self, dst: &mut Cache, src: &Cache, len: usize) {
        assert!(len <= dst.len && len <= src.len && len.is_multiple_of(CACHE_BLOCK));
        for (kernel, dst, src, words) in [
            (&self.kernels.copy_words[0], &dst.quants, &src.quants, len / 4),
            (
                &self.kernels.copy_words[1],
                &dst.scales,
                &src.scales,
                len / CACHE_BLOCK,
            ),
        ] {
            let params = LenParams { len: words as u32 };
            self.dispatch(
                kernel,
                &[dst, src],
                &params,
                words.div_ceil(THREADS),
                THREADS,
            );
        }
    }

    fn mix(&self, out: &mut Buffer, states: &[&Buffer], query: &Buffer, eps: f32) {
        let dim = query.len;
        assert!(
            !states.is_empty() && states.len() <= 6,
            "the kernel mixes up to six states"
        );
        assert!(
            dim.is_multiple_of(4) && dim <= 4096,
            "the kernel takes rows of up to 4096 in fours"
        );
        assert!(states.iter().all(|state| state.len == out.len) && out.len.is_multiple_of(dim));
        out.touch();
        let params = MixParams {
            dim: dim as u32,
            count: states.len() as u32,
            eps,
        };
        // The kernel binds six states: the ones there are not are the
        // first again, which it does not read.
        let state = |i: usize| &*states[i.min(states.len() - 1)].buf;
        let buffers: [&ProtocolObject<dyn MTLBuffer>; 8] = [
            &out.buf,
            &query.buf,
            state(0),
            state(1),
            state(2),
            state(3),
            state(4),
            state(5),
        ];
        self.dispatch(&self.kernels.mix, &buffers, &params, out.len / dim, THREADS);
    }

    fn gated_update(&self, x: &mut Buffer, forget: &Buffer, gain: &Buffer, message: &Buffer) {
        assert!(forget.len == x.len && gain.len == x.len && message.len == x.len);
        x.touch();
        let params = LenParams { len: x.len as u32 };
        self.dispatch(
            &self.kernels.gated_update,
            &[&x.buf, &forget.buf, &gain.buf, &message.buf],
            &params,
            x.len.div_ceil(THREADS),
            THREADS,
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
        x.touch();
        let params = GateParams {
            dim: dim as u32,
            rank: rank as u32,
            eps,
        };
        self.dispatch(
            &self.kernels.gate,
            &[
                &x.buf,
                &message.buf,
                &down_input.buf,
                &down_message.buf,
                &gain.buf,
                &forget.buf,
            ],
            &params,
            x.len / dim,
            THREADS,
        );
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
        assert!(taps <= 9, "the kernel keeps up to eight tokens of state");
        if let Some((dynamic, heads)) = dynamic {
            assert!(dynamic.len == n * heads * taps && channels.is_multiple_of(heads));
        }
        out.touch();
        state.touch();
        let params = ShortConvParams {
            n: n as u32,
            channels: channels as u32,
            taps: taps as u32,
            heads: dynamic.map_or(0, |(_, heads)| heads) as u32,
        };
        // Without taps of the heads' own, the kernel is bound the weight
        // in their place, which it does not read.
        let taps = dynamic.map_or(&weight.buf, |(dynamic, _)| &dynamic.buf);
        self.dispatch(
            &self.kernels.short_conv,
            &[&out.buf, &x.buf, &state.buf, &weight.buf, taps],
            &params,
            (n * channels).div_ceil(THREADS),
            THREADS,
        );
        // A token alone has its state moved on by the threads that
        // convolve it, a thread a channel.
        if n > 1 {
            self.dispatch(
                &self.kernels.short_conv_state,
                &[&state.buf, &x.buf],
                &params,
                channels.div_ceil(THREADS),
                THREADS,
            );
        }
    }

    fn clip(&self, x: &mut Buffer, scale: &Buffer, dim: usize, threshold: f32) {
        assert!(x.len.is_multiple_of(scale.len) && scale.len.is_multiple_of(dim));
        x.touch();
        let heads = x.len / dim;
        let params = ClipParams {
            heads: heads as u32,
            width: scale.len as u32,
            dim: dim as u32,
            threshold,
        };
        self.dispatch(
            &self.kernels.clip,
            &[&x.buf, &scale.buf],
            &params,
            heads.div_ceil(THREADS),
            THREADS,
        );
    }

    fn scale_rows(&self, x: &mut Buffer, factors: &Buffer) {
        assert!(x.len.is_multiple_of(factors.len));
        x.touch();
        let params = RowsParams {
            len: x.len as u32,
            width: (x.len / factors.len) as u32,
            k: 0,
        };
        self.dispatch(
            &self.kernels.scale_rows,
            &[&x.buf, &factors.buf],
            &params,
            x.len.div_ceil(THREADS),
            THREADS,
        );
    }

    fn block_conv(
        &self,
        out: &mut Buffer,
        x: &Buffer,
        taps: &Buffer,
        from: usize,
        base: &Buffer,
        kernel: usize,
        group: usize,
    ) {
        let channels = base.len / kernel;
        let n = x.len / channels;
        assert!(x.len == n * channels && out.len == x.len);
        assert!(base.len == kernel * channels && channels.is_multiple_of(group));
        let width = taps.len / n;
        assert!(taps.len == n * width && from + kernel * (channels / group) <= width);
        out.touch();
        let params = BlockConvParams {
            n: n as u32,
            channels: channels as u32,
            kernel: kernel as u32,
            group: group as u32,
            width: width as u32,
            from: from as u32,
        };
        self.dispatch(
            &self.kernels.block_conv,
            &[&out.buf, &x.buf, &taps.buf, &base.buf],
            &params,
            x.len.div_ceil(THREADS),
            THREADS,
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
        assert!(head_dim.is_multiple_of(CACHE_BLOCK) && head_dim <= ATTENTION_THREADS);
        assert!(
            n_heads / n_kv_heads <= 8,
            "the kernel holds up to eight query heads per key/value head"
        );
        assert!(sinks.len == n * n_heads && rel.len.is_multiple_of(n * n_heads));
        let n_rel = rel.len / (n * n_heads);
        assert!(bias.len.is_multiple_of(n_rel));
        out.touch();
        let group = n_heads / n_kv_heads;
        let extent = bias.len / n_rel;
        let buffers: [&ProtocolObject<dyn MTLBuffer>; 10] = [
            &out.buf,
            &q.buf,
            &k_cache.quants,
            &k_cache.scales,
            &v_cache.quants,
            &v_cache.scales,
            &self.partials,
            &rel.buf,
            &bias.buf,
            &sinks.buf,
        ];
        let params = |first: usize, tokens: usize, chunks: usize, chunk: usize| AttentionRelParams {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            n_kv_heads: n_kv_heads as u32,
            pos: pos as u32,
            first: first as u32,
            chunks: chunks as u32,
            chunk: chunk as u32,
            tokens: tokens as u32,
            padded: 0,
            window: window.min(u32::MAX as usize) as u32,
            ring: rows as u32,
            n_rel: n_rel as u32,
            extent: extent as u32,
            base: 0,
            batched: 0,
        };
        // A batch of tokens of heads of 128 goes through the kernel on the
        // matrix accelerators, where the GPU has them, and so does a token
        // that attends over enough positions that reading the caches is
        // the work, in tiles of its own size. The queries are packed for
        // it: as many tokens to a dispatch as the partials and the packed
        // queries fit at the most chunks a dispatch has, over chunks that
        // start where the dispatch's first token attends from.
        let decode = n == 1 && (pos + 1).min(window) >= DECODE_DEPTH;
        if let Some(kernels) = &self.kernels.attention_rel_mma
            && (n >= FLASH_TOKENS || decode)
            && head_dim == 128
            && n_rel.is_multiple_of(4)
            && n_rel <= 16
        {
            let (kernel, tile_rows, threads) = match decode {
                false => (&kernels[0], MMA_ROWS, MMA_THREADS),
                true => (&kernels[1], DECODE_ROWS, DECODE_THREADS),
            };
            let packed_rows = PACKED / (n_kv_heads * head_dim) / tile_rows * tile_rows;
            let per_dispatch = (PARTIALS / (n_heads * MMA_CHUNKS * (head_dim + 2)))
                .min(packed_rows / group)
                .max(1);
            for first in (0..n).step_by(per_dispatch) {
                let tokens = per_dispatch.min(n - first);
                let base = (pos + first + 1).saturating_sub(window);
                let span = pos + first + tokens - base;
                let chunk = span.div_ceil(MMA_CHUNKS).next_multiple_of(FLASH_CHUNK);
                let chunks = span.div_ceil(chunk);
                let padded = (tokens * group).next_multiple_of(tile_rows);
                self.packed_for.set(None);
                let pack = PackQueriesParams {
                    n_heads: n_heads as u32,
                    n_kv_heads: n_kv_heads as u32,
                    dim: head_dim as u32,
                    first: first as u32,
                    n: tokens as u32,
                    padded: padded as u32,
                };
                self.dispatch(
                    &self.kernels.pack_queries,
                    &[&self.packed, &q.buf],
                    &pack,
                    (n_kv_heads * padded * head_dim / 4).div_ceil(THREADS),
                    THREADS,
                );
                let params = AttentionRelParams {
                    padded: padded as u32,
                    base: base as u32,
                    batched: 1,
                    ..params(first, tokens, chunks, chunk)
                };
                let mut packed = buffers;
                packed[1] = &self.packed;
                self.dispatch_grid(
                    kernel,
                    &packed,
                    &params,
                    ((tokens * group).div_ceil(tile_rows), n_kv_heads * chunks),
                    threads,
                );
                if chunks > 1 {
                    self.dispatch(
                        &self.kernels.attention_rel_combine,
                        &[&out.buf, &self.partials, &sinks.buf],
                        &params,
                        tokens * n_heads,
                        ATTENTION_THREADS,
                    );
                }
            }
            return;
        }
        // Every token's heads get as many chunks as the last token attends
        // over, as many tokens to a dispatch as the partials fit.
        let chunks = (pos + n).min(window).div_ceil(CHUNK);
        let per_token = n_heads * chunks * (head_dim + 2);
        assert!(
            per_token <= PARTIALS,
            "attention over {} positions needs more partials than {PARTIALS}",
            pos + n
        );
        let per_dispatch = PARTIALS / per_token;
        for first in (0..n).step_by(per_dispatch) {
            let tokens = per_dispatch.min(n - first);
            let params = params(first, tokens, chunks, CHUNK);
            self.dispatch(
                &self.kernels.attention_rel,
                &buffers,
                &params,
                tokens * n_kv_heads * chunks,
                ATTENTION_THREADS,
            );
            if chunks > 1 {
                self.dispatch(
                    &self.kernels.attention_rel_combine,
                    &[&out.buf, &self.partials, &sinks.buf],
                    &params,
                    tokens * n_heads,
                    ATTENTION_THREADS,
                );
            }
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
        assert!(logits.len == n * experts && ids.len == n * k && weights.len == n * k);
        assert!(
            experts <= 256 && k <= 32 && k <= experts,
            "the kernel picks up to 32 of up to 256 experts"
        );
        ids.touch();
        weights.touch();
        let params = RouteParams {
            n: n as u32,
            experts: experts as u32,
            k: k as u32,
            scale,
        };
        self.dispatch(
            &self.kernels.route,
            &[&ids.buf, &weights.buf, &logits.buf, &bias.buf],
            &params,
            n.div_ceil(THREADS),
            THREADS,
        );
    }

    fn situ(&self, out: &mut Buffer, gate_up: &Buffer, width: usize, beta: f32, linear_beta: f32) {
        assert!(gate_up.len == 2 * out.len && out.len.is_multiple_of(width));
        out.touch();
        let params = SituParams {
            len: out.len as u32,
            width: width as u32,
            beta,
            linear_beta,
        };
        self.dispatch(
            &self.kernels.situ,
            &[&out.buf, &gate_up.buf],
            &params,
            out.len.div_ceil(THREADS),
            THREADS,
        );
    }

    fn weighted_sum(&self, out: &mut Buffer, y: &Buffer, weights: &Buffer) {
        let width = y.len / weights.len;
        assert!(y.len == weights.len * width && out.len.is_multiple_of(width));
        let k = weights.len / (out.len / width);
        out.touch();
        let params = RowsParams {
            len: out.len as u32,
            width: width as u32,
            k: k as u32,
        };
        self.dispatch(
            &self.kernels.weighted_sum,
            &[&out.buf, &y.buf, &weights.buf],
            &params,
            out.len.div_ceil(THREADS),
            THREADS,
        );
    }
}

#[cfg(test)]
mod tests {
    use objc2_metal::{MTLCompileOptions, MTLLanguageVersion};

    use crate::gpu::Device;
    check_against_cpu!(super::Metal::new());

    /// Checks the tile kernels for block-quantized weights, which a GPU
    /// without matrix accelerators takes, on one that has them too.
    #[test]
    fn quant_matmul_without_accelerators_matches_cpu() {
        match super::Metal::new() {
            Ok(mut gpu) => {
                gpu.kernels.matmul_quant_mma = None;
                crate::gpu::tests::q4_0_matmul_matches_cpu(&gpu);
                crate::gpu::tests::q8_0_matmul_matches_cpu(&gpu);
            }
            Err(e) => eprintln!("skipping: {e}"),
        }
    }

    #[test]
    #[ignore]
    fn q4_0_matmul_speed() {
        match super::Metal::new() {
            Ok(gpu) => crate::gpu::tests::q4_0_matmul_speed(&gpu),
            Err(e) => eprintln!("skipping: {e}"),
        }
    }

    #[test]
    #[ignore]
    fn q8_0_matmul_speed() {
        match super::Metal::new() {
            Ok(gpu) => crate::gpu::tests::q8_0_matmul_speed(&gpu),
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
                    crate::gpu::tests::kernel_speed(&gpu, n);
                }
            }
            Err(e) => eprintln!("skipping: {e}"),
        }
    }

    /// Checks the layout of a SIMD group's 8x8 matrix over its lanes that
    /// `attention_flash.metal` rescales the rows of its sums by: lanes 0-15
    /// hold rows 0-3 and 16-31 rows 4-7, two lanes to a row for columns
    /// 0-3, then two for columns 4-7.
    #[test]
    fn simdgroup_matrix_layout() {
        const SOURCE: &str = r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void simdgroup_matrix_layout(
                device float* out [[buffer(0)]],
                uint lane [[thread_index_in_simdgroup]])
            {
                threadgroup float m[64];
                for (uint i = lane; i < 64; i += 32) {
                    m[i] = float(i);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_float8x8 a;
                simdgroup_load(a, m, 8);
                out[2 * lane] = a.thread_elements()[0];
                out[2 * lane + 1] = a.thread_elements()[1];
            }
        "#;
        let gpu = match super::Metal::new() {
            Ok(gpu) => gpu,
            Err(e) => return eprintln!("skipping: {e}"),
        };
        let kernel = super::compile(
            &gpu.device,
            "simdgroup_matrix_layout",
            32,
            &[],
            &[SOURCE],
            None,
        )
        .unwrap();
        let out = gpu.alloc(64);
        gpu.dispatch(&kernel, &[&out.buf], &0u32, 1, 32);
        let got = gpu.read(&out);
        for lane in 0..32 {
            let row = ((lane >> 1) & 3) + ((lane >> 4) << 2);
            let col = (lane & 1) * 2 + ((lane >> 3) & 1) * 4;
            let want = [(row * 8 + col) as f32, (row * 8 + col + 1) as f32];
            assert_eq!(got[2 * lane..2 * lane + 2], want, "lane {lane}");
        }
    }

    /// Times the experts' matmul at Inkling's shapes; run with `--ignored
    /// --nocapture`.
    #[test]
    #[ignore]
    fn expert_matmul_speed() {
        match super::Metal::new() {
            Ok(gpu) => crate::gpu::tests::expert_matmul_speed(&gpu),
            Err(e) => eprintln!("skipping: {e}"),
        }
    }

    /// Times attention for one token, as a step of decoding takes it, at
    /// depths of context: it reads the caches once, so its speed is a
    /// rate of reading them. Run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn decode_attention_speed() {
        match super::Metal::new() {
            Ok(gpu) => crate::gpu::tests::attention_speed(&gpu, 1),
            Err(e) => eprintln!("skipping: {e}"),
        }
    }

    /// Checks the batched attention kernel for the SIMD groups' matrices
    /// against the CPU on a GPU that would take the accelerators' instead.
    #[test]
    fn attention_flash_matches_cpu() {
        match super::Metal::new() {
            Ok(mut gpu) => {
                gpu.kernels.attention_mma = None;
                crate::gpu::tests::attention_matches_cpu(&gpu);
            }
            Err(e) => eprintln!("skipping: {e}"),
        }
    }

    /// Times the batched attention kernels at depths of context: the one
    /// on the matrix accelerators, where the GPU has them, and the one on
    /// the SIMD groups' matrices, whole and with its dequantization or its
    /// matrix multiplies left out, to tell which bounds it; run with
    /// `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn attention_speed() {
        let mut gpu = match super::Metal::new() {
            Ok(gpu) => gpu,
            Err(e) => return eprintln!("skipping: {e}"),
        };
        if gpu.kernels.attention_mma.is_some() {
            let metal4 = MTLCompileOptions::new();
            metal4.setLanguageVersion(MTLLanguageVersion::Version4_0);
            for (ablate, what) in [
                (0, "whole"),
                (1, "without dequantizing"),
                (2, "without multiplying"),
                (4, "without rescaling"),
                (7, "without any of them"),
            ] {
                let mma = super::compile(
                    &gpu.device,
                    "attention_mma",
                    super::MMA_THREADS,
                    &[("DIM", 256), ("ABLATE", ablate)],
                    &[source!("attention_mma")],
                    Some(&metal4),
                )
                .unwrap();
                gpu.kernels.attention_mma.as_mut().unwrap()[1] = mma;
                eprintln!("on the matrix accelerators, {what}:");
                crate::gpu::tests::attention_speed(&gpu, 64);
            }
            gpu.kernels.attention_mma = None;
        }
        for (ablate, what) in [
            (0, "whole"),
            (1, "without dequantizing"),
            (2, "without multiplying"),
            (3, "without either"),
            (7, "without either or rescaling"),
            (15, "without either, rescaling, or the softmax"),
        ] {
            gpu.kernels.attention_flash[1] = super::compile(
                &gpu.device,
                "attention_flash",
                super::MATRIX_THREADS,
                &[
                    ("DIM", 256),
                    ("CHUNK", super::FLASH_CHUNK),
                    ("ABLATE", ablate),
                ],
                &[source!("attention_flash")],
                None,
            )
            .unwrap();
            eprintln!("{what}:");
            crate::gpu::tests::attention_speed(&gpu, 64);
        }
    }

    /// Times the cost of a dispatch; run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn dispatch_overhead() {
        match super::Metal::new() {
            Ok(gpu) => crate::gpu::tests::dispatch_overhead(&gpu),
            Err(e) => eprintln!("skipping: {e}"),
        }
    }
}
