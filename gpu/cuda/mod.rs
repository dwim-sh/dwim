//! The model on an NVIDIA GPU, through CUDA: every operation is a
//! hand-written kernel in `kernels/`, compiled from CUDA C by NVRTC when
//! the device opens, for the GPU's own architecture.
//!
//! Operations go to one stream in the order they are called, which runs
//! them in that order, so nothing needs a barrier; reading a buffer back
//! waits for the stream. The driver and NVRTC are loaded when the device
//! opens, so a build needs no CUDA, and a machine without them gets an
//! error from [`Cuda::new`] rather than a device.

use std::{
    cell::{Cell, RefCell},
    error::Error,
    slice,
    sync::Arc,
};

use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchConfig,
        PushKernelArg,
        sys::{self, CUfunction_attribute},
    },
    nvrtc::{self, CompileError, CompileOptions, Ptx},
};
use rayon::prelude::*;

use crate::gpu::{CACHE_BLOCK, CONV_KERNEL, Device, Tensor, quant};

/// Tokens a block of the bf16 matmul kernel takes at once.
const MATMUL_TOKENS: usize = 8;

/// Rows and columns of a tile of block-quantized weights: a group of rows
/// a block of the quantized matmul kernels takes, and two of the format's
/// blocks of columns.
const TILE_ROWS: usize = 128;
const TILE_COLS: usize = 2 * quant::BLOCK;

/// Tokens a block of the row kernels takes at once, of a batch.
const ROWS_TOKENS: usize = 8;

/// Tokens a block of the wide tile kernel takes, and of the narrow one,
/// which takes what is left of a batch after the wide one's.
const TILE_TOKENS: usize = 64;
const TILE_TOKENS_NARROW: usize = 16;

/// Tokens to a batch on a GPU with tensor cores.
const TILE_BATCH: usize = 256;

/// Tiles a block of the tile kernels holds in shared memory at once: the
/// one it is multiplying and the ones on their way.
const TILE_STAGES: usize = 3;

/// The head the attention tile kernels are written for, lines of query
/// heads a block of the narrow and of the wide one takes, and the positions
/// they take at a step.
const ATTENTION_TILE_DIM: usize = 256;
const ATTENTION_TILE_LINES: [usize; 2] = [16, 64];
const ATTENTION_TILE_KEYS: usize = 32;

/// Blocks an attention over the tiles is spread over, about, and the
/// fewest positions worth a chunk of their own.
const ATTENTION_TILE_BLOCKS: usize = 192;
const ATTENTION_TILE_CHUNK: usize = 256;

/// The compute capability the tile kernels need, for their tensor cores
/// and the copies into shared memory that are not waited for.
const TILE_CAPABILITY: i32 = 80;

/// Positions one block of the attention kernel takes.
const CHUNK: usize = 128;

/// Most floats of attention partials one dispatch writes, one chunk's
/// maximum, sum, and weighted values per head of each token: the size of
/// the partials buffer. Longer batches are dispatched a few tokens at a
/// time.
const PARTIALS: usize = 16 << 20;

/// Most blocks along each dimension of a grid.
const MAX_GRID_X: usize = i32::MAX as usize;
const MAX_GRID_Y: usize = 65535;

/// The kernels, compiled together in this order: a kernel may use what an
/// earlier one defines.
const KERNELS: [(&str, &str); 17] = [
    ("prelude", include_str!("kernels/prelude.cu")),
    ("add", include_str!("kernels/add.cu")),
    ("silu_mul", include_str!("kernels/silu_mul.cu")),
    ("sigmoid_mul", include_str!("kernels/sigmoid_mul.cu")),
    ("rope", include_str!("kernels/rope.cu")),
    ("rmsnorm", include_str!("kernels/rmsnorm.cu")),
    ("l2norm", include_str!("kernels/l2norm.cu")),
    ("store_q8", include_str!("kernels/store_q8.cu")),
    ("attention", include_str!("kernels/attention.cu")),
    (
        "attention_combine",
        include_str!("kernels/attention_combine.cu"),
    ),
    ("conv", include_str!("kernels/conv.cu")),
    ("delta_net", include_str!("kernels/delta_net.cu")),
    ("matmul_bf16", include_str!("kernels/matmul_bf16.cu")),
    ("to_f16", include_str!("kernels/to_f16.cu")),
    ("matmul_rows", include_str!("kernels/matmul_rows.cu")),
    ("matmul_tile", include_str!("kernels/matmul_tile.cu")),
    ("attention_tile", include_str!("kernels/attention_tile.cu")),
];

/// A vector of activations in device memory: as f32, and as halves too
/// once something has them so, which the tile kernels read and write.
/// Either may be behind the other, as `form` says, until an operation
/// wants the one that is.
pub struct Buffer {
    buf: CudaSlice<f32>,
    halves: RefCell<Option<CudaSlice<u16>>>,
    form: Cell<Form>,
    len: usize,
    cap: usize,
}

/// Which of a buffer's two forms hold its activations.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Form {
    Floats,
    Halves,
    Both,
}

/// A cache of activations in device memory, in blocks of 8-bit integers,
/// and the blocks' scales as f16 bits.
pub struct Cache {
    quants: CudaSlice<i8>,
    scales: CudaSlice<u16>,
    len: usize,
}

/// A weight matrix in device memory: its weights, and for the
/// block-quantized formats their scales in a buffer of their own, both in
/// tiles, as [`tile`] lays them out.
pub struct Weight {
    buf: CudaSlice<u8>,
    scales: CudaSlice<u16>,
    shape: Vec<usize>,
    format: Format,
}

/// The formats weights come in, as [`Tensor`] has them.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    Bf16,
    Q4_0,
    Q8_0,
}

/// The matmul kernels of a block-quantized format.
struct QuantKernels {
    /// A token in single precision, and a batch eight tokens at a time.
    row: CudaFunction,
    rows: CudaFunction,
    /// The tokens on the tensor cores, 16 and 64 at a time, where the GPU
    /// has them.
    tiles: Option<Tiles>,
}

/// The narrow and the wide tile kernel of a format.
type Tiles = (CudaFunction, CudaFunction);

/// What a tile matmul does with its products, as the kernels number it.
#[derive(Clone, Copy)]
enum Products {
    Floats = 0,
    Halves = 1,
    Add = 2,
    Swiglu = 3,
}

struct Kernels {
    matmul_bf16: CudaFunction,
    matmul_bf16_f16: CudaFunction,
    from_f16: CudaFunction,
    normalize_f16: CudaFunction,
    q4_0: QuantKernels,
    q8_0: QuantKernels,
    to_f16: CudaFunction,
    /// Attention on the tensor cores, narrow and wide, and the kernel that
    /// merges their chunks, where the GPU has them.
    attention_tiles: Option<([CudaFunction; 2], CudaFunction)>,
    add: CudaFunction,
    rmsnorm: CudaFunction,
    l2norm: CudaFunction,
    rope: CudaFunction,
    attention: CudaFunction,
    attention_combine: CudaFunction,
    silu_mul: CudaFunction,
    sigmoid_mul: CudaFunction,
    store_q8: CudaFunction,
    conv: CudaFunction,
    delta_net: CudaFunction,
}

pub struct Cuda {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernels: Kernels,
    partials: CudaSlice<f32>,
    name: String,
}

impl Cuda {
    /// Opens the first NVIDIA GPU, compiling the kernels for it.
    pub fn new() -> Result<Self, Box<dyn Error>> {
        Self::open(true)
    }

    /// Opens the GPU as one without tensor cores is opened, whatever it
    /// has, to check the kernels such a GPU runs.
    #[cfg(test)]
    fn without_tensor_cores() -> Result<Self, Box<dyn Error>> {
        Self::open(false)
    }

    /// Opens the first NVIDIA GPU, with the tile kernels if `tiles` and
    /// the GPU has the tensor cores for them.
    fn open(tiles: bool) -> Result<Self, Box<dyn Error>> {
        if !unsafe { sys::is_culib_present() } {
            return Err("no CUDA driver".into());
        }
        if !unsafe { nvrtc::sys::is_culib_present() } {
            return Err("no NVRTC, which the CUDA toolkit has, to compile the kernels".into());
        }
        let ctx = CudaContext::new(0).map_err(|e| format!("no CUDA device: {e}"))?;
        // Everything goes to one stream, so the events that would order
        // several are not needed.
        unsafe { ctx.disable_event_tracking() };
        let name = ctx.name()?;
        let (major, minor) = ctx.compute_capability()?;
        let (ptx, arch) = compile(major * 10 + minor).map_err(|e| format!("{name}: {e}"))?;
        let module = ctx.load_module(ptx)?;
        let stream = ctx.default_stream();
        let kernel = |name: &str| module.load_function(name);
        let quant = |format: &str, bytes: usize| -> Result<QuantKernels, Box<dyn Error>> {
            let tiles = if tiles && arch >= TILE_CAPABILITY {
                let tile = |tokens: usize| -> Result<CudaFunction, Box<dyn Error>> {
                    let f = kernel(&format!("matmul_tile{tokens}_{format}"))?;
                    // More shared memory than a kernel may have unasked.
                    f.set_attribute(
                        CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        tile_shared(tokens, bytes) as i32,
                    )?;
                    Ok(f)
                };
                Some((tile(TILE_TOKENS_NARROW)?, tile(TILE_TOKENS)?))
            } else {
                None
            };
            Ok(QuantKernels {
                row: kernel(&format!("matmul_rows_{format}"))?,
                rows: kernel(&format!("matmul_rows{ROWS_TOKENS}_{format}"))?,
                tiles,
            })
        };
        let attention_tiles = if tiles && arch >= TILE_CAPABILITY {
            let tile = |lines: usize| -> Result<CudaFunction, Box<dyn Error>> {
                let f = kernel(&format!("attention_tile{lines}"))?;
                f.set_attribute(
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    attention_tile_shared(lines) as i32,
                )?;
                Ok(f)
            };
            Some((
                [tile(ATTENTION_TILE_LINES[0])?, tile(ATTENTION_TILE_LINES[1])?],
                kernel("attention_tile_combine")?,
            ))
        } else {
            None
        };
        let kernels = Kernels {
            attention_tiles,
            matmul_bf16: kernel("matmul_bf16")?,
            matmul_bf16_f16: kernel("matmul_bf16_f16")?,
            from_f16: kernel("from_f16")?,
            normalize_f16: kernel("normalize_f16")?,
            q4_0: quant("q4_0", 1)?,
            q8_0: quant("q8_0", 2)?,
            to_f16: kernel("to_f16")?,
            add: kernel("add")?,
            rmsnorm: kernel("rmsnorm")?,
            l2norm: kernel("l2norm")?,
            rope: kernel("rope")?,
            attention: kernel("attention")?,
            attention_combine: kernel("attention_combine")?,
            silu_mul: kernel("silu_mul")?,
            sigmoid_mul: kernel("sigmoid_mul")?,
            store_q8: kernel("store_q8")?,
            conv: kernel("conv")?,
            delta_net: kernel("delta_net")?,
        };
        let partials = stream.alloc_zeros::<f32>(PARTIALS)?;
        Ok(Self {
            ctx,
            stream,
            kernels,
            partials,
            name,
        })
    }

    /// The address of a slice, for a kernel.
    fn ptr<T>(&self, slice: &CudaSlice<T>) -> u64 {
        slice.device_ptr(&self.stream).0
    }

    /// The address of a buffer's floats, made of its halves first if only
    /// those hold its activations.
    fn floats(&self, b: &Buffer) -> u64 {
        if b.form.get() == Form::Halves {
            let halves = b.halves.borrow();
            let halves = halves.as_ref().expect("a buffer in halves has them");
            self.launch(
                &self.kernels.from_f16,
                &[self.ptr(&b.buf), self.ptr(halves)],
                &LenParams { len: b.len as u32 },
                self.grid(b.len.div_ceil(4), 256),
                256,
            );
            b.form.set(Form::Both);
        }
        self.ptr(&b.buf)
    }

    /// The address of a buffer's floats for an operation that changes
    /// some of them, which leaves its halves behind.
    fn floats_mut(&self, b: &mut Buffer) -> u64 {
        let floats = self.floats(b);
        b.form.set(Form::Floats);
        floats
    }

    /// The address of a buffer's floats for an operation that writes them
    /// all, whatever the buffer held.
    fn floats_out(&self, b: &mut Buffer) -> u64 {
        b.form.set(Form::Floats);
        self.ptr(&b.buf)
    }

    /// The address of a buffer's halves, made of its floats first if only
    /// those hold its activations.
    fn halves(&self, b: &Buffer) -> u64 {
        let halves = self.halves_of(b);
        if b.form.get() == Form::Floats {
            self.launch(
                &self.kernels.to_f16,
                &[halves, self.ptr(&b.buf)],
                &LenParams { len: b.len as u32 },
                self.grid(b.len.div_ceil(4), 256),
                256,
            );
            b.form.set(Form::Both);
        }
        halves
    }

    /// The address of a buffer's halves as they are, allocated if it has
    /// had none: for an operation that writes them all.
    fn halves_of(&self, b: &Buffer) -> u64 {
        let mut halves = b.halves.borrow_mut();
        let halves = halves.get_or_insert_with(|| {
            self.stream
                .alloc_zeros::<u16>(b.cap.next_multiple_of(4).max(4))
                .expect("out of GPU memory")
        });
        self.ptr(halves)
    }

    /// The address of a buffer's halves for an operation that writes them
    /// as it writes the floats, where the GPU has the tensor cores that
    /// read them, and none otherwise.
    fn halves_too(&self, b: &Buffer) -> u64 {
        if self.kernels.q4_0.tiles.is_none() {
            return 0;
        }
        b.form.set(Form::Both);
        self.halves_of(b)
    }

    /// The tile kernels for a weight, if it is block-quantized and the GPU
    /// has the tensor cores.
    fn tiles(&self, w: &Weight) -> Option<&Tiles> {
        match w.format {
            Format::Bf16 => None,
            Format::Q4_0 => self.kernels.q4_0.tiles.as_ref(),
            Format::Q8_0 => self.kernels.q8_0.tiles.as_ref(),
        }
    }

    /// A matmul on the tensor cores of the `n` tokens whose halves are at
    /// `x`, into `out` as `products` has it, with the halves at `other`
    /// for the products to be multiplied by if that is what it has: as
    /// many whole wide tiles of tokens as there are, and the rest in
    /// narrow ones.
    #[allow(clippy::too_many_arguments)]
    fn matmul_tiles(
        &self,
        (narrow, wide): &Tiles,
        out: u64,
        w: &Weight,
        x: u64,
        other: u64,
        n: usize,
        products: Products,
    ) {
        let (rows, cols) = (w.shape[0], w.shape[1]);
        let bytes = if w.format == Format::Q8_0 { 2 } else { 1 };
        let whole = n / TILE_TOKENS * TILE_TOKENS;
        for (kernel, tokens, first, count) in [
            (wide, TILE_TOKENS, 0, whole),
            (narrow, TILE_TOKENS_NARROW, whole, n - whole),
        ] {
            if count == 0 {
                continue;
            }
            let params = TileParams {
                rows: rows as u32,
                cols: cols as u32,
                tiles: cols.div_ceil(TILE_COLS) as u32,
                first: first as u32,
                count: count as u32,
                mode: products as u32,
            };
            self.launch_shared(
                kernel,
                &[out, self.ptr(&w.buf), self.ptr(&w.scales), x, other],
                &params,
                (
                    count.div_ceil(tokens) as u32,
                    rows.div_ceil(TILE_ROWS) as u32,
                ),
                128,
                tile_shared(tokens, bytes),
            );
        }
    }

    /// Copies plain integers from the host into a new buffer of their
    /// bytes, in memory order: the kernels take the low half of a word as
    /// the first of its 16-bit weights or scales, as little-endian does.
    fn upload_bytes<T>(&self, data: &[T]) -> CudaSlice<u8> {
        let bytes = unsafe { slice::from_raw_parts(data.as_ptr().cast::<u8>(), size_of_val(data)) };
        // A whole number of words, for the kernels' arrays of them, and
        // never nothing.
        let mut buf = self
            .stream
            .alloc_zeros::<u8>(bytes.len().next_multiple_of(4).max(4))
            .expect("out of GPU memory");
        self.stream
            .memcpy_htod(bytes, &mut buf.slice_mut(..bytes.len()))
            .expect("uploading weights");
        buf
    }

    /// Runs a kernel over a grid of blocks of `threads`, with the buffers
    /// at `buffers` and then `params` as its arguments.
    fn launch<P: DeviceRepr>(
        &self,
        kernel: &CudaFunction,
        buffers: &[u64],
        params: &P,
        grid: (u32, u32),
        threads: u32,
    ) {
        self.launch_shared(kernel, buffers, params, grid, threads, 0);
    }

    /// As [`Self::launch`], with `shared` bytes of shared memory to a block
    /// beside what the kernel declares.
    fn launch_shared<P: DeviceRepr>(
        &self,
        kernel: &CudaFunction,
        buffers: &[u64],
        params: &P,
        grid: (u32, u32),
        threads: u32,
        shared: usize,
    ) {
        assert!(
            grid.1 as usize <= MAX_GRID_Y,
            "{} rows of blocks is more than the GPU allows",
            grid.1
        );
        let config = LaunchConfig {
            grid_dim: (grid.0, grid.1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: shared as u32,
        };
        let mut launch = self.stream.launch_builder(kernel);
        for buffer in buffers {
            launch.arg(buffer);
        }
        launch.arg(params);
        unsafe { launch.launch(config) }.expect("launching a kernel");
    }

    /// Blocks for `count` items of work in a one-dimensional kernel.
    fn grid(&self, count: usize, per_block: usize) -> (u32, u32) {
        let blocks = count.div_ceil(per_block);
        assert!(
            blocks <= MAX_GRID_X,
            "{blocks} blocks is more than the GPU allows"
        );
        (blocks as u32, 1)
    }
    /// Attention on the tensor cores, for heads of [`ATTENTION_TILE_DIM`]:
    /// the positions are cut into chunks so that there are blocks enough
    /// to fill the GPU, each a chunk of a key/value head for as many
    /// tokens as have a tile's lines of query heads between them, and the
    /// chunks are merged after.
    #[allow(clippy::too_many_arguments)]
    fn attention_tiles(
        &self,
        tiles: &[CudaFunction; 2],
        combine: &CudaFunction,
        out: &mut Buffer,
        q: &Buffer,
        k_cache: &Cache,
        v_cache: &Cache,
        pos: usize,
        n_heads: usize,
        n_kv_heads: usize,
    ) {
        let n = q.len / (n_heads * ATTENTION_TILE_DIM);
        let group = n_heads / n_kv_heads;
        // The narrow kernel for the tokens it takes in two blocks' worth,
        // the wide one for more.
        let narrow = ATTENTION_TILE_LINES[0] / group;
        let (kernel, lines, threads) = if n <= 2 * narrow {
            (&tiles[0], ATTENTION_TILE_LINES[0], 128)
        } else {
            (&tiles[1], ATTENTION_TILE_LINES[1], 256)
        };
        let tokens = lines / group;
        let len = pos + n;
        let chunks = ATTENTION_TILE_BLOCKS
            .div_ceil(n.div_ceil(tokens) * n_kv_heads)
            .min(len / ATTENTION_TILE_CHUNK)
            .max(1);
        let chunk = len.div_ceil(chunks).next_multiple_of(ATTENTION_TILE_KEYS);
        let chunks = len.div_ceil(chunk);
        let per_token = n_heads * chunks * (ATTENTION_TILE_DIM + 2);
        let per_dispatch = PARTIALS / per_token / tokens * tokens;
        assert!(
            per_dispatch > 0,
            "attention over {len} positions needs more partials than {PARTIALS}"
        );
        for first in (0..n).step_by(per_dispatch) {
            let count = per_dispatch.min(n - first);
            let params = AttentionTileParams {
                n_heads: n_heads as u32,
                n_kv_heads: n_kv_heads as u32,
                pos: pos as u32,
                first: first as u32,
                count: count as u32,
                chunks: chunks as u32,
                chunk: chunk as u32,
            };
            self.launch_shared(
                kernel,
                &[
                    self.ptr(&out.buf),
                    self.floats(q),
                    self.ptr(&k_cache.quants),
                    self.ptr(&k_cache.scales),
                    self.ptr(&v_cache.quants),
                    self.ptr(&v_cache.scales),
                    self.ptr(&self.partials),
                ],
                &params,
                self.grid(count.div_ceil(tokens) * n_kv_heads * chunks, 1),
                threads,
                attention_tile_shared(lines),
            );
            if chunks > 1 {
                self.launch(
                    combine,
                    &[self.ptr(&out.buf), self.ptr(&self.partials)],
                    &params,
                    self.grid(count * n_heads, 1),
                    ATTENTION_TILE_DIM as u32,
                );
            }
        }
    }

    /// A matmul through bf16 weights: one block of 64 threads per row and
    /// group of eight tokens, the rows across the grid and each group of
    /// tokens a row of it.
    fn matmul_bf16(&self, out: &mut Buffer, w: &Weight, x: &Buffer, n: usize) {
        let (rows, cols) = (w.shape[0], w.shape[1]);
        let width = rows.min(MAX_GRID_X);
        let grid = (
            width as u32,
            (rows.div_ceil(width) * n.div_ceil(MATMUL_TOKENS)) as u32,
        );
        let params = MatmulParams {
            rows: rows as u32,
            cols: cols as u32,
            n: n as u32,
            stride: width as u32,
            splits: 1,
        };
        // The activations as the buffer has them.
        let (kernel, x) = match x.form.get() {
            Form::Halves => (&self.kernels.matmul_bf16_f16, self.halves(x)),
            _ => (&self.kernels.matmul_bf16, self.floats(x)),
        };
        self.launch(
            kernel,
            &[self.floats_out(out), self.ptr(&w.buf), x],
            &params,
            grid,
            64,
        );
    }
}

/// Bytes of shared memory a block of a tile kernel of `tokens` tokens
/// takes, for weights of `bytes` bytes to a pair: a stage holds the
/// tokens' halves for a tile's columns, the tile, and its scales.
fn tile_shared(tokens: usize, bytes: usize) -> usize {
    let stage = tokens * TILE_COLS * 2 + TILE_ROWS * TILE_COLS / 2 * bytes + 2 * TILE_ROWS * 2;
    TILE_STAGES * stage
}

/// Bytes of shared memory a block of an attention tile kernel of `lines`
/// lines takes: each line's query and its scores and weights for a step,
/// what it is scaled by, its maximum and its sum, and a step's keys and
/// values as halves.
fn attention_tile_shared(lines: usize) -> usize {
    let line = ATTENTION_TILE_DIM * 2 + ATTENTION_TILE_KEYS * (4 + 2) + 3 * 4;
    lines * line + 2 * ATTENTION_TILE_KEYS * ATTENTION_TILE_DIM * 2
}

/// Lays a block-quantized matrix out in tiles of [`TILE_ROWS`] rows by
/// [`TILE_COLS`] columns, the tiles of a group of rows one after another,
/// for a kernel to read a tile as one run: each row of a tile as `words`
/// packs its two blocks, and the tile's scales a block of columns at a
/// time. Rows and blocks past the matrix's are zeros with a scale of zero.
fn tile<const W: usize>(
    quants: &[u8],
    scales: &[u16],
    rows: usize,
    cols: usize,
    words: fn(&[u8], &[u8]) -> [u32; W],
) -> (Vec<u32>, Vec<u16>) {
    let blocks = quant::blocks(cols);
    let per_block = quants.len() / (rows * blocks);
    let tiles = cols.div_ceil(TILE_COLS);
    let groups = rows.div_ceil(TILE_ROWS);
    let zeros = vec![0u8; per_block];
    let mut tiled = vec![0u32; groups * tiles * TILE_ROWS * W];
    let mut tiled_scales = vec![0u16; groups * tiles * 2 * TILE_ROWS];
    tiled
        .par_chunks_mut(tiles * TILE_ROWS * W)
        .zip(tiled_scales.par_chunks_mut(tiles * 2 * TILE_ROWS))
        .enumerate()
        .for_each(|(group, (tiled, tiled_scales))| {
            let first = group * TILE_ROWS;
            for row in first..rows.min(first + TILE_ROWS) {
                let block = |b: usize| {
                    if b < blocks {
                        &quants[(row * blocks + b) * per_block..][..per_block]
                    } else {
                        &zeros[..]
                    }
                };
                for tile in 0..tiles {
                    let at = tile * TILE_ROWS + row - first;
                    tiled[at * W..][..W].copy_from_slice(&words(block(tile * 2), block(tile * 2 + 1)));
                    for v in 0..2 {
                        if tile * 2 + v < blocks {
                            tiled_scales[(tile * 2 + v) * TILE_ROWS + row - first] =
                                scales[row * blocks + tile * 2 + v];
                        }
                    }
                }
            }
        });
    (tiled, tiled_scales)
}

/// A row's two `Q4_0` blocks of a tile as the kernels take them: eight
/// words, the lane's of each block side by side. Word `i` of a block has
/// its bytes `2i`, `2i + 8`, `2i + 1` and `2i + 9`, whose low nibbles are
/// the elements at those places and whose high nibbles the ones 16 after:
/// the four of each 16 that a lane of the tensor cores' warp takes.
fn q4_0_words(a: &[u8], b: &[u8]) -> [u32; 8] {
    let mut words = [0; 8];
    for (v, block) in [a, b].into_iter().enumerate() {
        for i in 0..4 {
            words[i * 2 + v] =
                u32::from_le_bytes([block[2 * i], block[2 * i + 8], block[2 * i + 1], block[2 * i + 9]]);
        }
    }
    words
}

/// A row's two `Q8_0` blocks of a tile likewise: sixteen words, word `i`
/// of each of the tile's four runs of 16 columns side by side, with the
/// run's elements `2i`, `2i + 1`, `2i + 8` and `2i + 9`.
fn q8_0_words(a: &[u8], b: &[u8]) -> [u32; 16] {
    let mut words = [0; 16];
    for (v, block) in [a, b].into_iter().enumerate() {
        for (half, run) in block.chunks_exact(16).enumerate() {
            for i in 0..4 {
                words[i * 4 + v * 2 + half] =
                    u32::from_le_bytes([run[2 * i], run[2 * i + 1], run[2 * i + 8], run[2 * i + 9]]);
            }
        }
    }
    words
}

/// Compiles the kernels to PTX for a GPU of compute capability `cc`, as
/// `major * 10 + minor`: for the newest architecture NVRTC knows that is
/// no newer than the GPU's, since PTX runs on the architectures after its
/// own.
fn compile(cc: i32) -> Result<(Ptx, i32), Box<dyn Error>> {
    // What the kernels need: half-precision conversions came with 5.3.
    const ARCHS: [(i32, &str); 12] = [
        (120, "compute_120"),
        (100, "compute_100"),
        (90, "compute_90"),
        (89, "compute_89"),
        (87, "compute_87"),
        (86, "compute_86"),
        (80, "compute_80"),
        (75, "compute_75"),
        (70, "compute_70"),
        (61, "compute_61"),
        (60, "compute_60"),
        (53, "compute_53"),
    ];
    let source = format!(
        "#define TILE_STAGES {TILE_STAGES}\n{}",
        KERNELS.map(|(_, source)| source).join("\n")
    );
    let mut log = String::new();
    for &(known, arch) in ARCHS.iter().filter(|&&(known, _)| known <= cc) {
        let options = CompileOptions {
            arch: Some(arch),
            name: Some("dwim.cu".into()),
            ..Default::default()
        };
        match nvrtc::compile_ptx_with_opts(&source, options) {
            Ok(ptx) => return Ok((ptx, known)),
            Err(CompileError::CompileError { log: text, .. }) => {
                log = text.to_string_lossy().into_owned();
                // An NVRTC older than the GPU does not know its
                // architecture; the newest it knows serves.
                if !log.contains("gpu-architecture") {
                    break;
                }
            }
            Err(e) => return Err(e.into()),
        }
    }
    if log.is_empty() {
        return Err(format!(
            "compute capability {}.{} is too old for the kernels",
            cc / 10,
            cc % 10
        )
        .into());
    }
    Err(format!("compiling the kernels:\n{log}").into())
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
struct TileParams {
    rows: u32,
    cols: u32,
    tiles: u32,
    first: u32,
    count: u32,
    mode: u32,
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
struct AttentionTileParams {
    n_heads: u32,
    n_kv_heads: u32,
    pos: u32,
    first: u32,
    count: u32,
    chunks: u32,
    chunk: u32,
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

// Plain structs of 32-bit fields, passed to the kernels by value.
unsafe impl DeviceRepr for MatmulParams {}
unsafe impl DeviceRepr for TileParams {}
unsafe impl DeviceRepr for RmsnormParams {}
unsafe impl DeviceRepr for RopeParams {}
unsafe impl DeviceRepr for ConvParams {}
unsafe impl DeviceRepr for DeltaNetParams {}
unsafe impl DeviceRepr for AttentionParams {}
unsafe impl DeviceRepr for AttentionTileParams {}
unsafe impl DeviceRepr for StoreParams {}
unsafe impl DeviceRepr for LenParams {}

impl Device for Cuda {
    type Buffer = Buffer;
    type Weight = Weight;
    type Cache = Cache;

    fn name(&self) -> String {
        self.name.clone()
    }

    fn memory(&self) -> Option<u64> {
        let (free, _total) = self.ctx.mem_get_info().ok()?;
        Some(free as u64)
    }

    fn batch(&self) -> usize {
        // The tensor cores get through a longer batch in the time the
        // row kernels take for a short one, and read the weights once
        // for all of it.
        match self.kernels.q4_0.tiles {
            Some(_) => TILE_BATCH,
            None => 64,
        }
    }

    fn upload(&self, tensor: Tensor) -> Weight {
        match tensor {
            Tensor::Bf16 { shape, data } => {
                assert!(
                    data.len().is_multiple_of(8),
                    "weights must come in multiples of eight"
                );
                Weight {
                    buf: self.upload_bytes(&data),
                    scales: self.stream.alloc_zeros(1).expect("out of GPU memory"),
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
                let (quants, scales) = tile(&quants, &scales, shape[0], shape[1], q4_0_words);
                Weight {
                    buf: self.upload_bytes(&quants),
                    scales: self.stream.clone_htod(&scales).expect("out of GPU memory"),
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
                let bytes = unsafe { slice::from_raw_parts(quants.as_ptr().cast::<u8>(), quants.len()) };
                let (quants, scales) = tile(bytes, &scales, shape[0], shape[1], q8_0_words);
                Weight {
                    buf: self.upload_bytes(&quants),
                    scales: self.stream.clone_htod(&scales).expect("out of GPU memory"),
                    shape,
                    format: Format::Q8_0,
                }
            }
        }
    }

    fn alloc(&self, len: usize) -> Buffer {
        // Whole fours, which the conversions between floats and halves
        // take.
        let buf = self
            .stream
            .alloc_zeros::<f32>(len.next_multiple_of(4).max(4))
            .expect("out of GPU memory");
        Buffer {
            buf,
            halves: RefCell::new(None),
            form: Cell::new(Form::Floats),
            len,
            cap: len,
        }
    }

    fn alloc_cache(&self, len: usize) -> Cache {
        assert!(len.is_multiple_of(CACHE_BLOCK));
        let quants = self
            .stream
            .alloc_zeros::<i8>(len.max(1))
            .expect("out of GPU memory");
        let scales = self
            .stream
            .alloc_zeros::<u16>((len / CACHE_BLOCK).max(1))
            .expect("out of GPU memory");
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
        self.floats(buf);
        self.stream
            .clone_dtoh(&buf.buf.slice(..buf.len))
            .expect("reading a buffer")
    }

    fn write(&self, buf: &mut Buffer, data: &[f32]) {
        assert_eq!(buf.len, data.len());
        buf.form.set(Form::Floats);
        self.stream
            .memcpy_htod(data, &mut buf.buf.slice_mut(..buf.len))
            .expect("writing a buffer");
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
        self.floats(src);
        self.floats_mut(dst);
        self.stream
            .memcpy_dtod(
                &src.buf.slice(src_offset..src_offset + len),
                &mut dst.buf.slice_mut(dst_offset..dst_offset + len),
            )
            .expect("copying a buffer");
    }

    fn store(&self, cache: &mut Cache, offset: usize, src: &Buffer) {
        assert!(offset + src.len <= cache.len);
        assert!(offset.is_multiple_of(CACHE_BLOCK) && src.len.is_multiple_of(CACHE_BLOCK));
        let params = StoreParams {
            offset: offset as u32,
            len: src.len as u32,
        };
        // A thread per block.
        let grid = self.grid(src.len / CACHE_BLOCK, 64);
        self.launch(
            &self.kernels.store_q8,
            &[
                self.ptr(&cache.quants),
                self.ptr(&cache.scales),
                self.floats(src),
            ],
            &params,
            grid,
            64,
        );
    }

    fn read_cache(&self, cache: &Cache, len: usize) -> Vec<u8> {
        assert!(len <= cache.len);
        let quants: Vec<i8> = self
            .stream
            .clone_dtoh(&cache.quants.slice(..len))
            .expect("reading a cache");
        let scales: Vec<u16> = self
            .stream
            .clone_dtoh(&cache.scales.slice(..len / CACHE_BLOCK))
            .expect("reading a cache");
        let quants = quants.iter().map(|&q| q as u8);
        quants
            .chain(scales.iter().flat_map(|s| s.to_le_bytes()))
            .collect()
    }

    fn write_cache(&self, cache: &mut Cache, data: &[u8]) {
        let len = data.len() / (CACHE_BLOCK + 2) * CACHE_BLOCK;
        assert!(len <= cache.len && data.len() == crate::gpu::cache_bytes(len));
        let (quants, scales) = data.split_at(len);
        let quants = unsafe { slice::from_raw_parts(quants.as_ptr().cast::<i8>(), len) };
        self.stream
            .memcpy_htod(quants, &mut cache.quants.slice_mut(..len))
            .expect("writing a cache");
        let scales: Vec<u16> = scales
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        self.stream
            .memcpy_htod(&scales, &mut cache.scales.slice_mut(..len / CACHE_BLOCK))
            .expect("writing a cache");
    }

    fn matmul(&self, out: &mut Buffer, w: &Weight, x: &Buffer) {
        let (rows, cols) = (w.shape[0], w.shape[1]);
        let n = x.len / cols;
        assert_eq!(x.len, n * cols);
        assert_eq!(out.len, n * rows);
        assert!(cols % 8 == 0);
        if w.format == Format::Bf16 {
            return self.matmul_bf16(out, w, x, n);
        }
        if let Some(tiles) = self.tiles(w) {
            let x = self.halves(x);
            let out = self.floats_out(out);
            return self.matmul_tiles(tiles, out, w, x, 0, n, Products::Floats);
        }
        // A GPU without tensor cores multiplies in single precision, a
        // token to a block or eight of a batch.
        let kernels = match w.format {
            Format::Q8_0 => &self.kernels.q8_0,
            _ => &self.kernels.q4_0,
        };
        let (kernel, per_block) = match n {
            1 => (&kernels.row, 1),
            _ => (&kernels.rows, ROWS_TOKENS),
        };
        let params = TileParams {
            rows: rows as u32,
            cols: cols as u32,
            tiles: cols.div_ceil(TILE_COLS) as u32,
            first: 0,
            count: n as u32,
            mode: Products::Floats as u32,
        };
        self.launch(
            kernel,
            &[
                self.floats_out(out),
                self.ptr(&w.buf),
                self.ptr(&w.scales),
                self.floats(x),
            ],
            &params,
            (
                rows.div_ceil(TILE_ROWS) as u32,
                n.div_ceil(per_block) as u32,
            ),
            256,
        );
    }

    fn matmul_add(&self, x: &mut Buffer, w: &Weight, input: &Buffer, scratch: &mut Buffer) {
        let Some(tiles) = self.tiles(w) else {
            self.matmul(scratch, w, input);
            return self.add(x, scratch);
        };
        let (rows, cols) = (w.shape[0], w.shape[1]);
        let n = input.len / cols;
        assert_eq!(input.len, n * cols);
        assert_eq!(x.len, n * rows);
        // The products are added to the floats where they are.
        let input = self.halves(input);
        let x = self.floats_mut(x);
        self.matmul_tiles(tiles, x, w, input, 0, n, Products::Add);
    }

    fn swiglu(
        &self,
        out: &mut Buffer,
        gate: &Weight,
        up: &Weight,
        x: &Buffer,
        scratch: &mut Buffer,
    ) {
        let (Some(gate_tiles), Some(up_tiles)) = (self.tiles(gate), self.tiles(up)) else {
            self.matmul(out, gate, x);
            self.matmul(scratch, up, x);
            return self.silu_mul(out, scratch);
        };
        let (rows, cols) = (gate.shape[0], gate.shape[1]);
        let n = x.len / cols;
        assert_eq!(x.len, n * cols);
        assert_eq!(up.shape, gate.shape);
        assert!(out.len == n * rows && scratch.len == out.len);
        // Both in halves and nothing else: `up`'s products, and then
        // `gate`'s through the activation times those as they come.
        let x = self.halves(x);
        let other = self.halves_of(scratch);
        scratch.form.set(Form::Halves);
        self.matmul_tiles(up_tiles, other, up, x, 0, n, Products::Halves);
        let halves = self.halves_of(out);
        out.form.set(Form::Halves);
        self.matmul_tiles(gate_tiles, halves, gate, x, other, n, Products::Swiglu);
    }

    fn normalize(&self, out: &mut Buffer, x: &Buffer, weight: &Buffer, len: usize, eps: f32) {
        // Without tensor cores nothing reads halves.
        if self.kernels.q4_0.tiles.is_none() {
            self.copy(out, 0, x, 0, len);
            return self.rmsnorm(out, weight, eps);
        }
        let dim = weight.len;
        assert!(out.len == len && len <= x.len && len.is_multiple_of(dim));
        let params = RmsnormParams {
            dim: dim as u32,
            eps,
        };
        let halves = self.halves_of(out);
        out.form.set(Form::Halves);
        self.launch(
            &self.kernels.normalize_f16,
            &[halves, self.floats(x), self.floats(weight)],
            &params,
            self.grid(len / dim, 1),
            256,
        );
    }

    fn add(&self, x: &mut Buffer, y: &Buffer) {
        assert_eq!(x.len, y.len);
        let params = LenParams { len: x.len as u32 };
        let grid = self.grid(x.len, 256);
        self.launch(
            &self.kernels.add,
            &[self.floats_mut(x), self.floats(y)],
            &params,
            grid,
            256,
        );
    }

    fn rmsnorm(&self, x: &mut Buffer, weight: &Buffer, eps: f32) {
        let dim = weight.len;
        assert_eq!(x.len % dim, 0);
        let params = RmsnormParams {
            dim: dim as u32,
            eps,
        };
        let grid = self.grid(x.len / dim, 1);
        self.launch(
            &self.kernels.rmsnorm,
            &[self.floats_mut(x), self.floats(weight)],
            &params,
            grid,
            256,
        );
    }

    fn l2norm(&self, x: &mut Buffer, dim: usize, eps: f32) {
        assert_eq!(x.len % dim, 0);
        let params = RmsnormParams {
            dim: dim as u32,
            eps,
        };
        let grid = self.grid(x.len / dim, 1);
        self.launch(
            &self.kernels.l2norm,
            &[self.floats_mut(x)],
            &params,
            grid,
            256,
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
        let grid = self.grid(n * n_heads * rot_dim / 2, 256);
        self.launch(
            &self.kernels.rope,
            &[self.floats_mut(x), self.floats(table)],
            &params,
            grid,
            256,
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
        assert!(head_dim.is_multiple_of(CACHE_BLOCK) && head_dim <= 256);
        out.form.set(Form::Floats);
        assert!(
            n_heads / n_kv_heads <= 8,
            "the kernel holds up to eight query heads per key/value head"
        );
        if let Some((tiles, combine)) = &self.kernels.attention_tiles
            && head_dim == ATTENTION_TILE_DIM
        {
            return self.attention_tiles(
                tiles, combine, out, q, k_cache, v_cache, pos, n_heads, n_kv_heads,
            );
        }
        // Every token's heads get as many chunks as the last token attends
        // over, as many tokens to a dispatch as the partials fit.
        let chunks = (pos + n).div_ceil(CHUNK);
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
            };
            let grid = self.grid(tokens * n_kv_heads * chunks, 1);
            self.launch(
                &self.kernels.attention,
                &[
                    self.ptr(&out.buf),
                    self.floats(q),
                    self.ptr(&k_cache.quants),
                    self.ptr(&k_cache.scales),
                    self.ptr(&v_cache.quants),
                    self.ptr(&v_cache.scales),
                    self.ptr(&self.partials),
                ],
                &params,
                grid,
                256,
            );
            if chunks > 1 {
                let grid = self.grid(tokens * n_heads, 1);
                self.launch(
                    &self.kernels.attention_combine,
                    &[self.ptr(&out.buf), self.ptr(&self.partials)],
                    &params,
                    grid,
                    256,
                );
            }
        }
    }

    fn silu_mul(&self, gate: &mut Buffer, up: &Buffer) {
        assert_eq!(gate.len, up.len);
        let params = LenParams {
            len: gate.len as u32,
        };
        let grid = self.grid(gate.len, 256);
        self.launch(
            &self.kernels.silu_mul,
            &[self.floats_mut(gate), self.halves_too(gate), self.floats(up)],
            &params,
            grid,
            256,
        );
    }

    fn sigmoid_mul(&self, x: &mut Buffer, gate: &Buffer) {
        assert_eq!(x.len, gate.len);
        let params = LenParams { len: x.len as u32 };
        let grid = self.grid(x.len, 256);
        self.launch(
            &self.kernels.sigmoid_mul,
            &[self.floats_mut(x), self.halves_too(x), self.floats(gate)],
            &params,
            grid,
            256,
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
        let grid = self.grid(n * channels, 256);
        self.launch(
            &self.kernels.conv,
            &[
                self.floats_out(q),
                self.floats_out(k),
                self.floats_out(v),
                self.floats_out(state_out),
                self.floats(x),
                self.floats(state),
                self.floats(weight),
            ],
            &params,
            grid,
            256,
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
        // Two blocks to a head, 64 of its values each.
        let grid = self.grid(2 * n_v_heads, 1);
        self.launch(
            &self.kernels.delta_net,
            &[
                self.floats_out(out),
                self.floats(q),
                self.floats(k),
                self.floats(v),
                self.floats(gates),
                self.floats(decay),
                self.floats_mut(state),
            ],
            &params,
            grid,
            128,
        );
    }
}

#[cfg(test)]
mod tests {
    // With tensor cores, every matmul through block-quantized weights is
    // in half precision; without them, none is.
    check_against_cpu!(
        super::Cuda::new();
        write_and_read_round_trip,
        alloc_is_zeroed_and_copy_moves_ranges,
        matmul_matches_cpu,
        q4_0_matmul_in_halves_matches_cpu,
        q8_0_matmul_in_halves_matches_cpu,
        rmsnorm_matches_cpu,
        l2norm_matches_cpu,
        rope_matches_cpu,
        attention_matches_cpu,
        store_quantizes_like_cpu,
        cache_round_trips,
        elementwise_match_cpu,
        normalize_matches_cpu,
        swiglu_matches_cpu,
        matmul_add_matches_cpu,
        conv_matches_cpu,
        delta_net_matches_cpu,
        resize_keeps_capacity
    );

    mod without_tensor_cores {
        check_against_cpu!(
            super::super::Cuda::without_tensor_cores();
            q4_0_matmul_matches_cpu,
            q8_0_matmul_matches_cpu,
            normalize_matches_cpu,
            swiglu_matches_cpu,
            matmul_add_matches_cpu
        );
    }

    /// Times the Q4_0 matmul on the model's shapes; run with `--ignored
    /// --nocapture`.
    #[test]
    #[ignore]
    fn q4_0_matmul_speed() {
        match super::Cuda::new() {
            Ok(gpu) => crate::gpu::tests::q4_0_matmul_speed(&gpu),
            Err(e) => eprintln!("skipping: {e}"),
        }
    }

    /// Times the Q8_0 matmul likewise.
    #[test]
    #[ignore]
    fn q8_0_matmul_speed() {
        match super::Cuda::new() {
            Ok(gpu) => crate::gpu::tests::q8_0_matmul_speed(&gpu),
            Err(e) => eprintln!("skipping: {e}"),
        }
    }

    /// Times the kernels at the model's shapes, for one token and for a
    /// batch of 64; run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn kernel_speed() {
        match super::Cuda::new() {
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
        match super::Cuda::new() {
            Ok(gpu) => crate::gpu::tests::attention_speed(&gpu, 256),
            Err(e) => eprintln!("skipping: {e}"),
        }
    }

    /// Times the cost of a dispatch; run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn dispatch_overhead() {
        match super::Cuda::new() {
            Ok(gpu) => crate::gpu::tests::dispatch_overhead(&gpu),
            Err(e) => eprintln!("skipping: {e}"),
        }
    }
}
