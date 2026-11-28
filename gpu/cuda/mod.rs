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
    fmt,
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

use crate::gpu::{self, CACHE_BLOCK, CONV_KERNEL, Device, Kept, Tensor, float, quant};

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
const TILE_BATCH: usize = 512;
/// Most matrix multiplications of a few tokens that go to the GPU as one
/// grid.
const TILE_JOBS: usize = 8;
/// Most weights of a matrix that is not quantized that are kept as halves
/// in tiles too: such a matrix is a small one.
const HALVES_WEIGHTS: usize = 1 << 24;
/// Bytes of a tile's pair of weights in halves, as [`tile_shared`] counts
/// them.
const HALVES_BYTES: usize = 4;
/// Most mixes of the same states the kernel for several takes.
const MIXES: usize = 4;
/// Most normalizations, and most clips, that go to the GPU as one grid.
const JOBS: usize = 4;

/// Most pairs of a token and an expert that go to the experts' tiles as
/// they are, a tile each, rather than sorted by expert first: a token's.
const EXPERT_DIRECT: usize = 32;

/// Tokens a block of the gate's kernels takes at once.
const GATE_TOKENS: usize = 8;
/// Most gates that go to the GPU as one grid.
const GATES: usize = 4;

/// Tiles a block of the tile kernels holds in shared memory at once: the
/// one it is multiplying and the ones on their way.
const TILE_STAGES: usize = 3;

/// The head the attention tile kernels are written for, lines of query
/// heads a block of the narrow and of the wide one takes, and the positions
/// they take at a step.
const ATTENTION_TILE_DIM: usize = 256;
const ATTENTION_TILE_LINES: [usize; 2] = [16, 64];
const ATTENTION_TILE_KEYS: usize = 32;

/// The head the tile kernels of attention with a bias by distance are
/// written for, and most elements of a head's projection for the bias
/// that they take, in fours.
const ATTENTION_REL_TILE_DIM: usize = 128;
const ATTENTION_TILE_REL: usize = 16;

/// How far into a sequence a token alone takes those kernels, in chunks of
/// no fewer positions than this many: nearer its start, it is attended to
/// in single precision.
const ATTENTION_TILE_DEPTH: usize = 2048;
const ATTENTION_TILE_TOKEN_CHUNK: usize = 64;

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
const KERNELS: [(&str, &str); 25] = [
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
    ("mix", include_str!("kernels/mix.cu")),
    ("gate", include_str!("kernels/gate.cu")),
    ("route", include_str!("kernels/route.cu")),
    ("short_conv", include_str!("kernels/short_conv.cu")),
    ("conv_store", include_str!("kernels/conv_store.cu")),
    ("clip", include_str!("kernels/clip.cu")),
    ("situ", include_str!("kernels/situ.cu")),
    ("expert_rows", include_str!("kernels/expert_rows.cu")),
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
    scales: CudaSlice<u8>,
    shape: Vec<usize>,
    format: Format,
    /// Weights that are not quantized, as halves in the tiles the tensor
    /// cores take: where there are tensor cores and the matrix is one a
    /// batch is multiplied by.
    halves: Option<CudaSlice<u8>>,
}

/// The formats weights come in, as [`Tensor`] has them.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    Bf16,
    Q4_0,
    Q8_0,
    Nvfp4,
}

/// The matmul kernels of a block-quantized format.
struct QuantKernels {
    /// A token in single precision, and a batch eight tokens at a time.
    row: CudaFunction,
    rows: CudaFunction,
    /// The tokens on the tensor cores, 16 and 64 at a time, where the GPU
    /// has them.
    tiles: Option<Tiles>,
    /// Several multiplications of up to 16 tokens in one grid, likewise.
    several: Option<CudaFunction>,
}

/// The narrow and the wide tile kernel of a format.
type Tiles = (CudaFunction, CudaFunction);

/// The experts' matmul kernels of a format: a pair of a token and an
/// expert at a time in single precision, and tiles of an expert's pairs
/// on the tensor cores, where the GPU has them.
struct ExpertKernels {
    rows: CudaFunction,
    tiles: Option<Tiles>,
    /// The tiles of a matrix that is a gate projection over an up
    /// projection, whose output is the SiTU activation of the two.
    situ: Option<Tiles>,
}

/// Where the pairs of a batch go on the tensor cores: a tile each, or in
/// the tiles written to the GPU, so many wide and so many narrow, of the
/// pairs in the order written beside them.
#[derive(Clone, Copy)]
enum ExpertPlan {
    Direct,
    Sorted {
        order: u64,
        tiles: u64,
        wide: usize,
        narrow: usize,
    },
}

/// What an expert's pairs of a batch are multiplied in: the pairs sorted
/// by expert, and the tiles of them, three words each, an expert, the
/// first of its pairs and how many, the wide kernel's and then the narrow
/// one's. The buffers are kept from a batch to the next.
#[derive(Default)]
struct Plan {
    order: Option<CudaSlice<u32>>,
    tiles: Option<CudaSlice<u32>>,
}

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
    normalizes_f16: CudaFunction,
    q4_0: QuantKernels,
    q8_0: QuantKernels,
    /// The tile kernels of weights in halves, where the GPU has the
    /// tensor cores.
    f16: Option<Tiles>,
    to_f16: CudaFunction,
    /// Attention on the tensor cores, narrow and wide, and the kernel that
    /// merges their chunks, where the GPU has them.
    attention_tiles: Option<([CudaFunction; 2], CudaFunction)>,
    /// The same with a bias by distance, a sink, a window and a ring, for
    /// heads of [`ATTENTION_REL_TILE_DIM`].
    attention_rel_tiles: Option<([CudaFunction; 2], CudaFunction)>,
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
    /// The experts' kernels over NVFP4 and over Q8_0 weights.
    expert_nvfp4: ExpertKernels,
    expert_q8_0: ExpertKernels,
    /// Several mixes of the same states: one, up to three, and four.
    mixes: [CudaFunction; 3],
    gate_down: CudaFunction,
    gate_down_few: CudaFunction,
    gate_norms: CudaFunction,
    gate_up: CudaFunction,
    route: CudaFunction,
    short_conv: CudaFunction,
    short_conv_state: CudaFunction,
    conv_store: CudaFunction,
    clip: CudaFunction,
    clips: CudaFunction,
    scale_rows: CudaFunction,
    softcap: CudaFunction,
    situ: CudaFunction,
    weighted_sum: CudaFunction,
    attention_rel: CudaFunction,
    attention_rel_combine: CudaFunction,
}

/// An error opening a CUDA device.
#[derive(Debug)]
pub enum CudaOpenError {
    /// No CUDA driver, NVRTC, or device: the GPU is not there, so the
    /// checks of it skip.
    Missing(String),
    /// The driver is there but a kernel would not compile or load, so the
    /// checks of it fail.
    Broken(String),
}

impl fmt::Display for CudaOpenError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CudaOpenError::Missing(msg) | CudaOpenError::Broken(msg) => f.write_str(msg),
        }
    }
}

impl Error for CudaOpenError {}

pub struct Cuda {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernels: Kernels,
    partials: CudaSlice<f32>,
    plan: RefCell<Plan>,
    /// What [`Device::gate`] keeps between its two kernels: what each
    /// token's input and its message come down to, and what normalizes
    /// each.
    gate_down: RefCell<Option<CudaSlice<f32>>>,
    name: String,
}

impl Weight {
    /// The weights as the tile kernels take them, and the bytes of a pair
    /// of them.
    fn tiled(&self) -> (&CudaSlice<u8>, usize) {
        match (self.format, &self.halves) {
            (Format::Bf16, Some(halves)) => (halves, HALVES_BYTES),
            (Format::Q8_0, _) => (&self.buf, 2),
            _ => (&self.buf, 1),
        }
    }
}

impl Cuda {
    /// Opens the first NVIDIA GPU, compiling the kernels for it.
    pub fn new() -> Result<Self, CudaOpenError> {
        Self::open(true)
    }

    /// Opens the GPU as one without tensor cores is opened, whatever it
    /// has, to check the kernels such a GPU runs.
    #[cfg(test)]
    fn without_tensor_cores() -> Result<Self, CudaOpenError> {
        Self::open(false)
    }

    /// Opens the first NVIDIA GPU, with the tile kernels if `tiles` and
    /// the GPU has the tensor cores for them.
    fn open(tiles: bool) -> Result<Self, CudaOpenError> {
        if !unsafe { sys::is_culib_present() } {
            return Err(CudaOpenError::Missing("no CUDA driver".into()));
        }
        if !unsafe { nvrtc::sys::is_culib_present() } {
            return Err(CudaOpenError::Missing(
                "no NVRTC, which the CUDA toolkit has, to compile the kernels".into(),
            ));
        }
        let ctx = CudaContext::new(0)
            .map_err(|e| CudaOpenError::Missing(format!("no CUDA device: {e}")))?;
        Self::device(ctx, tiles).map_err(|e| CudaOpenError::Broken(e.to_string()))
    }

    /// Opens a CUDA device that is present: compiles the kernels for it
    /// and loads them.
    fn device(ctx: Arc<CudaContext>, tiles: bool) -> Result<Self, Box<dyn Error>> {
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
            let several = match &tiles {
                Some(_) => {
                    let f = kernel(&format!("matmul_tiles{TILE_TOKENS_NARROW}_{format}"))?;
                    f.set_attribute(
                        CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        tile_shared(TILE_TOKENS_NARROW, bytes) as i32,
                    )?;
                    Some(f)
                }
                None => None,
            };
            Ok(QuantKernels {
                row: kernel(&format!("matmul_rows_{format}"))?,
                rows: kernel(&format!("matmul_rows{ROWS_TOKENS}_{format}"))?,
                tiles,
                several,
            })
        };
        let f16 = if tiles && arch >= TILE_CAPABILITY {
            let tile = |tokens: usize| -> Result<CudaFunction, Box<dyn Error>> {
                let f = kernel(&format!("matmul_tile{tokens}_f16"))?;
                f.set_attribute(
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    tile_shared(tokens, HALVES_BYTES) as i32,
                )?;
                Ok(f)
            };
            Some((tile(TILE_TOKENS_NARROW)?, tile(TILE_TOKENS)?))
        } else {
            None
        };
        let expert = |format: &str, bytes: usize| -> Result<ExpertKernels, Box<dyn Error>> {
            let tile = |name: &str, tokens: usize| -> Result<CudaFunction, Box<dyn Error>> {
                let f = kernel(&format!("{name}_tile{tokens}_{format}"))?;
                f.set_attribute(
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    expert_shared(tokens, bytes) as i32,
                )?;
                Ok(f)
            };
            let pair = |name: &str| -> Result<Option<Tiles>, Box<dyn Error>> {
                if tiles && arch >= TILE_CAPABILITY {
                    Ok(Some((tile(name, TILE_TOKENS_NARROW)?, tile(name, TILE_TOKENS)?)))
                } else {
                    Ok(None)
                }
            };
            Ok(ExpertKernels {
                rows: kernel(&format!("expert_rows_{format}"))?,
                tiles: pair("expert")?,
                situ: pair("expert_situ")?,
            })
        };
        let attention_tiles = if tiles && arch >= TILE_CAPABILITY {
            let tile = |lines: usize| -> Result<CudaFunction, Box<dyn Error>> {
                let f = kernel(&format!("attention_tile{lines}"))?;
                f.set_attribute(
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    attention_tile_shared(lines, ATTENTION_TILE_DIM) as i32,
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
        let attention_rel_tiles = if tiles && arch >= TILE_CAPABILITY {
            let tile = |lines: usize| -> Result<CudaFunction, Box<dyn Error>> {
                let f = kernel(&format!("attention_rel_tile{lines}"))?;
                f.set_attribute(
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    attention_tile_shared(lines, ATTENTION_REL_TILE_DIM) as i32,
                )?;
                Ok(f)
            };
            Some((
                [tile(ATTENTION_TILE_LINES[0])?, tile(ATTENTION_TILE_LINES[1])?],
                kernel("attention_rel_tile_combine")?,
            ))
        } else {
            None
        };
        let kernels = Kernels {
            attention_tiles,
            attention_rel_tiles,
            matmul_bf16: kernel("matmul_bf16")?,
            matmul_bf16_f16: kernel("matmul_bf16_f16")?,
            from_f16: kernel("from_f16")?,
            normalize_f16: kernel("normalize_f16")?,
            normalizes_f16: kernel("normalizes_f16")?,
            q4_0: quant("q4_0", 1)?,
            q8_0: quant("q8_0", 2)?,
            f16,
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
            expert_nvfp4: expert("nvfp4", 1)?,
            expert_q8_0: expert("q8_0", 2)?,
            mixes: [kernel("mixes1")?, kernel("mixes3")?, kernel("mixes4")?],
            gate_down: kernel("gate_down")?,
            gate_down_few: kernel("gate_down_few")?,
            gate_norms: kernel("gate_norms")?,
            gate_up: kernel("gate_up")?,
            route: kernel("route")?,
            short_conv: kernel("short_conv")?,
            short_conv_state: kernel("short_conv_state")?,
            conv_store: kernel("conv_store")?,
            clip: kernel("clip")?,
            clips: kernel("clips")?,
            scale_rows: kernel("scale_rows")?,
            softcap: kernel("softcap")?,
            situ: kernel("situ")?,
            weighted_sum: kernel("weighted_sum")?,
            attention_rel: kernel("attention_rel")?,
            attention_rel_combine: kernel("attention_rel_combine")?,
        };
        let partials = stream.alloc_zeros::<f32>(PARTIALS)?;
        Ok(Self {
            ctx,
            stream,
            kernels,
            partials,
            plan: RefCell::default(),
            gate_down: RefCell::default(),
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
            Format::Nvfp4 => None,
            Format::Bf16 => w.halves.as_ref().and(self.kernels.f16.as_ref()),
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
        let (weights, bytes) = w.tiled();
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
                &[out, self.ptr(weights), self.ptr(&w.scales), x, other],
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

    /// Moves a convolution's state on past a batch of the activations at
    /// `x`: a token alone has its state moved on by the threads that
    /// convolve it, a thread a channel.
    fn short_conv_state(&self, state: u64, x: u64, params: &ShortConvParams) {
        if params.n > 1 {
            self.launch(
                &self.kernels.short_conv_state,
                &[state, x],
                params,
                self.grid(params.channels as usize, 256),
                256,
            );
        }
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
                attention_tile_shared(lines, ATTENTION_TILE_DIM),
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

    /// Where each of the pairs of a token and an expert that `ids` has
    /// goes on the tensor cores: nowhere it is not already, for few
    /// enough that each is a tile of its own; otherwise the pairs are
    /// read back and sorted by expert, an expert's cut into wide tiles and
    /// the rest into narrow ones, and the order and the tiles written to
    /// the GPU, for the matmuls through the experts' matrices that follow.
    fn expert_plan(&self, ids: &Buffer, experts: usize) -> ExpertPlan {
        let pairs = ids.len;
        if pairs <= EXPERT_DIRECT {
            return ExpertPlan::Direct;
        }
        let picks: Vec<usize> = self.read(ids).iter().map(|&e| e as usize).collect();
        let mut first = vec![0; experts + 1];
        for &expert in &picks {
            first[expert + 1] += 1;
        }
        for expert in 0..experts {
            first[expert + 1] += first[expert];
        }
        let mut order = vec![0u32; pairs];
        let mut next = first.clone();
        for (pair, &expert) in picks.iter().enumerate() {
            order[next[expert]] = pair as u32;
            next[expert] += 1;
        }
        let (mut wides, mut narrows) = (Vec::new(), Vec::new());
        for expert in 0..experts {
            let (mut at, end) = (first[expert], first[expert + 1]);
            while at < end {
                let left = end - at;
                let (tiles, of) = if left > 2 * TILE_TOKENS_NARROW {
                    (&mut wides, left.min(TILE_TOKENS))
                } else {
                    (&mut narrows, left.min(TILE_TOKENS_NARROW))
                };
                tiles.extend([expert as u32, at as u32, of as u32]);
                at += of;
            }
        }
        let wide = wides.len() / 3;
        wides.extend(&narrows);
        // Nothing is running that reads the last batch's: the picks were
        // waited for.
        let mut plan = self.plan.borrow_mut();
        let Plan { order: on_gpu, tiles } = &mut *plan;
        for (slot, words) in [(&mut *on_gpu, &order), (&mut *tiles, &wides)] {
            match slot {
                Some(buf) if buf.len() >= words.len() => self
                    .stream
                    .memcpy_htod(words, &mut buf.slice_mut(..words.len()))
                    .expect("writing the experts' tiles"),
                _ => *slot = Some(self.stream.clone_htod(words).expect("out of GPU memory")),
            }
        }
        ExpertPlan::Sorted {
            order: self.ptr(on_gpu.as_ref().expect("written above")),
            tiles: self.ptr(tiles.as_ref().expect("written above")),
            wide,
            narrow: narrows.len() / 3,
        }
    }

    /// A matmul through the experts' matrices on the tensor cores, of the
    /// pairs `plan` places, each the expert `ids` has for it: `groups`
    /// groups of an expert's rows to a tile.
    #[allow(clippy::too_many_arguments)]
    fn expert_tiles(
        &self,
        (narrow, wide): &Tiles,
        out: u64,
        w: &Weight,
        scale: &Buffer,
        x: u64,
        ids: &Buffer,
        plan: ExpertPlan,
        groups: usize,
        params: ExpertParams,
    ) {
        let bytes = if w.format == Format::Q8_0 { 2 } else { 1 };
        let run = |kernel: &CudaFunction, tokens: usize, order: u64, plan: u64, count, direct| {
            let params = ExpertParams {
                direct: direct as u32,
                ..params
            };
            self.launch_shared(
                kernel,
                &[
                    out,
                    self.ptr(&w.buf),
                    self.ptr(&w.scales),
                    x,
                    order,
                    plan,
                    self.floats(ids),
                    self.floats(scale),
                ],
                &params,
                (groups as u32, count as u32),
                128,
                expert_shared(tokens, bytes),
            );
        };
        match plan {
            ExpertPlan::Direct => run(narrow, TILE_TOKENS_NARROW, 0, 0, ids.len, true),
            ExpertPlan::Sorted {
                order,
                tiles,
                wide: wides,
                narrow: narrows,
            } => {
                if wides > 0 {
                    run(wide, TILE_TOKENS, order, tiles, wides, false);
                }
                if narrows > 0 {
                    let plan = tiles + (wides * 3 * 4) as u64;
                    run(narrow, TILE_TOKENS_NARROW, order, plan, narrows, false);
                }
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

/// The shape of a short convolution of `x`, checked against its state,
/// its taps, and those of the heads' own.
fn short_conv_params(
    x: &Buffer,
    state: &Buffer,
    weight: &Buffer,
    dynamic: Option<(&Buffer, usize)>,
) -> ShortConvParams {
    let channels = weight.len - state.len;
    let taps = weight.len / channels;
    let n = x.len / channels;
    assert_eq!(x.len, n * channels);
    assert_eq!(state.len, (taps - 1) * channels);
    assert!(taps <= 9, "the kernel keeps up to eight tokens of state");
    if let Some((dynamic, heads)) = dynamic {
        assert!(dynamic.len == n * heads * taps && channels.is_multiple_of(heads));
    }
    ShortConvParams {
        n: n as u32,
        channels: channels as u32,
        taps: taps as u32,
        heads: dynamic.map_or(0, |(_, heads)| heads) as u32,
        add: 0,
    }
}

/// Bytes of shared memory a block of a tile kernel of `tokens` tokens
/// takes, for weights of `bytes` bytes to a pair: a stage holds the
/// tokens' halves for a tile's columns, the tile, and its scales, which
/// the halves have none of.
fn tile_shared(tokens: usize, bytes: usize) -> usize {
    let scales = if bytes == HALVES_BYTES { 0 } else { 2 * TILE_ROWS * 2 };
    let stage = tokens * TILE_COLS * 2 + TILE_ROWS * TILE_COLS / 2 * bytes + scales;
    TILE_STAGES * stage
}

/// The same for a block of an experts' tile kernel, which keeps the
/// numbers of its pairs and of their inputs' rows past the stages.
fn expert_shared(tokens: usize, bytes: usize) -> usize {
    tile_shared(tokens, bytes) + 2 * tokens * 4
}

/// Bytes of shared memory a block of an attention tile kernel of `lines`
/// lines takes: each line's query and its scores and weights for a step,
/// what it is scaled by, its maximum and its sum, and a step's keys and
/// values as halves.
fn attention_tile_shared(lines: usize, dim: usize) -> usize {
    let line = dim * 2 + ATTENTION_TILE_KEYS * (4 + 2) + 3 * 4;
    lines * line + 2 * ATTENTION_TILE_KEYS * dim * 2
}

/// Lays a block-quantized matrix out in tiles of [`TILE_ROWS`] rows by
/// [`TILE_COLS`] columns, the tiles of a group of rows one after another,
/// for a kernel to read a tile as one run: each row of a tile as `words`
/// packs its two blocks, and the tile's scales a block of columns at a
/// time. Rows and blocks past the matrix's are zeros with a scale of zero.
///
/// The scales are `per_tile` to a row of a tile, each for as many of its
/// columns as the others: two for the formats of blocks of 32, and four
/// for NVFP4, whose own blocks are 16, two to each block here.
fn tile<const W: usize, S: Copy + Default + Send + Sync>(
    quants: &[u8],
    scales: &[S],
    rows: usize,
    cols: usize,
    per_tile: usize,
    words: fn(&[u8], &[u8]) -> [u32; W],
) -> (Vec<u32>, Vec<S>) {
    let blocks = quant::blocks(cols);
    let per_block = quants.len() / (rows * blocks);
    let tiles = cols.div_ceil(TILE_COLS);
    let groups = rows.div_ceil(TILE_ROWS);
    let zeros = vec![0u8; per_block];
    let per_row = scales.len() / rows;
    let mut tiled = vec![0u32; groups * tiles * TILE_ROWS * W];
    let mut tiled_scales = vec![S::default(); groups * tiles * per_tile * TILE_ROWS];
    tiled
        .par_chunks_mut(tiles * TILE_ROWS * W)
        .zip(tiled_scales.par_chunks_mut(tiles * per_tile * TILE_ROWS))
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
                    for v in 0..per_tile {
                        if tile * per_tile + v < per_row {
                            tiled_scales[(tile * per_tile + v) * TILE_ROWS + row - first] =
                                scales[row * per_row + tile * per_tile + v];
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

/// NVFP4 weights, a row of blocks of 16 after another, as blocks of 32 laid
/// out as `Q4_0`'s are, for [`q4_0_words`]: a byte of a block of 32 has an
/// element of its first 16 in its low nibble and the one 16 after in its
/// high one, where a byte of NVFP4 has two of the same 16, eight apart.
fn nvfp4_blocks(quants: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; quants.len()];
    out.par_chunks_mut(16)
        .zip(quants.par_chunks(16))
        .for_each(|(out, pair)| {
            let nibble = |block: &[u8], i: usize| (block[i % 8] >> (i / 8 * 4)) & 0xf;
            for (i, byte) in out.iter_mut().enumerate() {
                *byte = nibble(&pair[..8], i) | nibble(&pair[8..], i) << 4;
            }
        });
    out
}

/// A row's two blocks of a tile of halves, as bytes: 32 words, the two a
/// lane takes of each of the tile's four runs of 16 columns side by side,
/// the run's elements `2i` and `2i + 1` and then `2i + 8` and `2i + 9`.
fn f16_words(a: &[u8], b: &[u8]) -> [u32; 32] {
    let mut words = [0; 32];
    for (v, block) in [a, b].into_iter().enumerate() {
        for (half, run) in block.chunks_exact(32).enumerate() {
            let pair = |e: usize| u32::from_le_bytes([run[2 * e], run[2 * e + 1], run[2 * e + 2], run[2 * e + 3]]);
            for i in 0..4 {
                let at = (i * 4 + v * 2 + half) * 2;
                words[at] = pair(2 * i);
                words[at + 1] = pair(2 * i + 8);
            }
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
        KERNELS
            .iter()
            .map(|(name, source)| format!("#line 1 \"kernels/{name}.cu\"\n{source}"))
            .collect::<Vec<_>>()
            .join("\n")
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
#[derive(Clone, Copy, Default)]
struct TileJob {
    out: u64,
    w: u64,
    scales: u64,
    x: u64,
    rows: u32,
    cols: u32,
    tiles: u32,
    groups: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TileJobs {
    job: [TileJob; TILE_JOBS],
    jobs: u32,
    count: u32,
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
struct AttentionRelParams {
    n_heads: u32,
    head_dim: u32,
    n_kv_heads: u32,
    pos: u32,
    first: u32,
    chunks: u32,
    chunk: u32,
    count: u32,
    base: u32,
    window: u32,
    ring: u32,
    n_rel: u32,
    extent: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NormJobs {
    out: [u64; JOBS],
    x: [u64; JOBS],
    weight: [u64; JOBS],
    dim: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ClipJobs {
    x: [u64; JOBS],
    scale: [u64; JOBS],
    first: [u32; JOBS],
    width: [u32; JOBS],
    dim: [u32; JOBS],
    threshold: [f32; JOBS],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MixesParams {
    out: [u64; MIXES],
    halves: [u64; MIXES],
    query: [u64; MIXES],
    norm: [u64; MIXES],
    state: [u64; 6],
    dim: u32,
    states: u32,
    mixes: u32,
    normed: u32,
    eps: f32,
    norm_eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GateJobs {
    x: [u64; GATES],
    message: [u64; GATES],
    down_input: [u64; GATES],
    down_message: [u64; GATES],
    gain: [u64; GATES],
    forget: [u64; GATES],
    n: u32,
    dim: u32,
    rank: u32,
    eps: f32,
    split: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RouteParams {
    experts: u32,
    k: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ShortConvParams {
    n: u32,
    channels: u32,
    taps: u32,
    heads: u32,
    add: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ConvStoreParams {
    n: u32,
    channels: u32,
    taps: u32,
    heads: u32,
    dim: u32,
    width: u32,
    kept: u32,
    pos: u32,
    rows: u32,
    value: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ClipParams {
    width: u32,
    dim: u32,
    threshold: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RowsParams {
    len: u32,
    width: u32,
    k: u32,
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
struct ExpertParams {
    rows: u32,
    cols: u32,
    tiles: u32,
    groups: u32,
    share: u32,
    direct: u32,
    beta: f32,
    linear_beta: f32,
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

// Plain structs of 32-bit fields, and of addresses, passed to the kernels
// by value.
unsafe impl DeviceRepr for MatmulParams {}
unsafe impl DeviceRepr for TileJobs {}
unsafe impl DeviceRepr for MixesParams {}
unsafe impl DeviceRepr for NormJobs {}
unsafe impl DeviceRepr for ClipJobs {}
unsafe impl DeviceRepr for TileParams {}
unsafe impl DeviceRepr for RmsnormParams {}
unsafe impl DeviceRepr for RopeParams {}
unsafe impl DeviceRepr for ConvParams {}
unsafe impl DeviceRepr for DeltaNetParams {}
unsafe impl DeviceRepr for AttentionParams {}
unsafe impl DeviceRepr for AttentionTileParams {}
unsafe impl DeviceRepr for StoreParams {}
unsafe impl DeviceRepr for LenParams {}
unsafe impl DeviceRepr for AttentionRelParams {}
unsafe impl DeviceRepr for GateJobs {}
unsafe impl DeviceRepr for RouteParams {}
unsafe impl DeviceRepr for ShortConvParams {}
unsafe impl DeviceRepr for ConvStoreParams {}
unsafe impl DeviceRepr for ClipParams {}
unsafe impl DeviceRepr for RowsParams {}
unsafe impl DeviceRepr for SituParams {}
unsafe impl DeviceRepr for ExpertParams {}

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

    fn takes_nvfp4(&self) -> bool {
        true
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
                // A half holds a bf16 weight exactly, down to magnitudes
                // of 2^-14 and to multiples of 2^-24 below.
                let halves = (self.kernels.f16.is_some()
                    && shape.len() == 2
                    && shape[1].is_multiple_of(quant::BLOCK)
                    && data.len() <= HALVES_WEIGHTS)
                    .then(|| {
                        let halves: Vec<u16> = data
                            .par_iter()
                            .map(|&w| float::to_f16(float::bf16(w)))
                            .collect();
                        let bytes = unsafe {
                            slice::from_raw_parts(halves.as_ptr().cast::<u8>(), 2 * halves.len())
                        };
                        let none = vec![0u8; shape[0] * quant::blocks(shape[1])];
                        let (tiled, _) = tile(bytes, &none, shape[0], shape[1], 1, f16_words);
                        self.upload_bytes(&tiled)
                    });
                Weight {
                    buf: self.upload_bytes(&data),
                    scales: self.stream.alloc_zeros(4).expect("out of GPU memory"),
                    shape,
                    format: Format::Bf16,
                    halves,
                }
            }
            Tensor::Q4_0 {
                shape,
                scales,
                quants,
            } => {
                assert_eq!(quants.len(), shape[0] * shape[1] / 2);
                assert_eq!(scales.len(), shape[0] * quant::blocks(shape[1]));
                let (quants, scales) = tile(&quants, &scales, shape[0], shape[1], 2, q4_0_words);
                Weight {
                    buf: self.upload_bytes(&quants),
                    scales: self.upload_bytes(&scales),
                    shape,
                    format: Format::Q4_0,
                    halves: None,
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
                let (quants, scales) = tile(bytes, &scales, shape[0], shape[1], 2, q8_0_words);
                Weight {
                    buf: self.upload_bytes(&quants),
                    scales: self.upload_bytes(&scales),
                    shape,
                    format: Format::Q8_0,
                    halves: None,
                }
            }
            Tensor::Nvfp4 {
                shape,
                scales,
                quants,
            } => {
                assert!(
                    shape[1].is_multiple_of(quant::BLOCK),
                    "NVFP4 weights come in rows of whole pairs of blocks"
                );
                assert_eq!(quants.len(), shape[0] * shape[1] / 2);
                assert_eq!(scales.len(), shape[0] * shape[1] / quant::NVFP4_BLOCK);
                let quants = nvfp4_blocks(&quants);
                let (quants, scales) = tile(&quants, &scales, shape[0], shape[1], 4, q4_0_words);
                Weight {
                    buf: self.upload_bytes(&quants),
                    scales: self.upload_bytes(&scales),
                    shape,
                    format: Format::Nvfp4,
                    halves: None,
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

    fn read_capped(&self, buf: &mut Buffer, cap: f32) -> Vec<f32> {
        let params = ClipParams {
            width: buf.len as u32,
            dim: 0,
            threshold: cap,
        };
        self.launch(
            &self.kernels.softcap,
            &[self.floats_mut(buf)],
            &params,
            self.grid(buf.len, 256),
            256,
        );
        self.read(buf)
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
        // Weights that are not quantized go through the tensor cores for
        // a batch, and for less a block to a row.
        if w.format == Format::Bf16 && (n < TILE_TOKENS_NARROW || self.tiles(w).is_none()) {
            return self.matmul_bf16(out, w, x, n);
        }
        assert!(
            w.format != Format::Nvfp4,
            "only the experts' matrices are NVFP4"
        );
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

    fn matmuls(&self, jobs: &mut [(&mut Buffer, &Weight, &Buffer)]) {
        // A few tokens through several matrices of one format are one
        // grid on the tensor cores: each matrix alone is too few blocks
        // to fill the GPU, and a grid waits for the one before.
        let format = jobs[0].1.format;
        let n = jobs[0].2.len / jobs[0].1.shape[1];
        let kernels = match format {
            Format::Q8_0 => &self.kernels.q8_0,
            _ => &self.kernels.q4_0,
        };
        let together = (2..=TILE_JOBS).contains(&jobs.len())
            && n <= TILE_TOKENS_NARROW
            && matches!(format, Format::Q4_0 | Format::Q8_0)
            && jobs
                .iter()
                .all(|(_, w, x)| w.format == format && x.len == n * w.shape[1]);
        let Some(kernel) = kernels.several.as_ref().filter(|_| together) else {
            for (out, w, x) in jobs {
                self.matmul(out, w, x);
            }
            return;
        };
        let mut params = TileJobs {
            job: [TileJob::default(); TILE_JOBS],
            jobs: jobs.len() as u32,
            count: n as u32,
        };
        let mut groups = 0;
        for (job, (out, w, x)) in params.job.iter_mut().zip(jobs.iter_mut()) {
            let (rows, cols) = (w.shape[0], w.shape[1]);
            assert_eq!(out.len, n * rows);
            let (weights, _) = w.tiled();
            *job = TileJob {
                out: self.floats_out(out),
                w: self.ptr(weights),
                scales: self.ptr(&w.scales),
                x: self.halves(x),
                rows: rows as u32,
                cols: cols as u32,
                tiles: cols.div_ceil(TILE_COLS) as u32,
                groups: rows.div_ceil(TILE_ROWS) as u32,
            };
            groups += job.groups;
        }
        let (_, bytes) = jobs[0].1.tiled();
        self.launch_shared(
            kernel,
            &[],
            &params,
            (1, groups),
            128,
            tile_shared(TILE_TOKENS_NARROW, bytes),
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

    fn copy_cache(&self, dst: &mut Cache, src: &Cache, len: usize) {
        assert!(len <= dst.len && len <= src.len && len.is_multiple_of(CACHE_BLOCK));
        if len == 0 {
            return;
        }
        self.stream
            .memcpy_dtod(&src.quants.slice(..len), &mut dst.quants.slice_mut(..len))
            .expect("copying a cache");
        let blocks = len / CACHE_BLOCK;
        self.stream
            .memcpy_dtod(
                &src.scales.slice(..blocks),
                &mut dst.scales.slice_mut(..blocks),
            )
            .expect("copying a cache");
    }

    fn mix(&self, out: &mut Buffer, states: &[&Buffer], query: &Buffer, eps: f32) {
        self.mixes(&mut [out], states, &[query], eps, None);
    }

    fn mixes(
        &self,
        outs: &mut [&mut Buffer],
        states: &[&Buffer],
        queries: &[&Buffer],
        eps: f32,
        norms: Option<(&[&Buffer], f32)>,
    ) {
        if outs.len() > MIXES {
            let (first, rest) = outs.split_at_mut(MIXES);
            let part = |from, to| norms.map(|(norms, eps)| (&norms[from..to], eps));
            self.mixes(first, states, &queries[..MIXES], eps, part(0, MIXES));
            return self.mixes(rest, states, &queries[MIXES..], eps, part(MIXES, queries.len()));
        }
        let dim = queries[0].len;
        let len = outs[0].len;
        assert!(
            !states.is_empty() && states.len() <= 6,
            "the kernel mixes up to six states"
        );
        assert!(
            dim.is_multiple_of(4) && dim <= 4096,
            "the kernel takes rows of up to 4096 in fours"
        );
        assert_eq!(outs.len(), queries.len());
        assert!(states.iter().all(|state| state.len == len) && len.is_multiple_of(dim));
        assert!(outs.iter().all(|out| out.len == len));
        assert!(queries.iter().all(|query| query.len == dim));
        let mut params = MixesParams {
            out: [0; MIXES],
            halves: [0; MIXES],
            query: [0; MIXES],
            norm: [0; MIXES],
            state: [0; 6],
            dim: dim as u32,
            states: states.len() as u32,
            mixes: outs.len() as u32,
            normed: norms.is_some() as u32,
            eps,
            norm_eps: norms.map_or(0.0, |(_, eps)| eps),
        };
        for (i, state) in states.iter().enumerate() {
            params.state[i] = self.floats(state);
        }
        // A mix that is normalized is a matrix multiplication's input,
        // which the tensor cores read as halves; one that is not goes
        // through a gate, which reads a batch's as both.
        let tiles = self.kernels.q4_0.tiles.is_some();
        for (j, out) in outs.iter_mut().enumerate() {
            params.query[j] = self.floats(queries[j]);
            if let Some((norms, _)) = norms {
                assert_eq!(norms[j].len, dim);
                params.norm[j] = self.floats(norms[j]);
            }
            if norms.is_none() || !tiles {
                params.out[j] = self.floats_out(out);
            }
            if tiles && (norms.is_some() || len > GATE_TOKENS * dim) {
                params.halves[j] = self.halves_of(out);
                out.form.set(match norms {
                    Some(_) => Form::Halves,
                    None => Form::Both,
                });
            }
        }
        let kernel = match outs.len() {
            1 => 0,
            2 | 3 => 1,
            _ => 2,
        };
        self.launch(
            &self.kernels.mixes[kernel],
            &[],
            &params,
            self.grid(len / dim, 1),
            256,
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
        self.gates(
            &mut [x],
            &[message],
            &[[down_input, down_message, gain, forget]],
            dim,
            rank,
            eps,
        );
    }

    fn gates(
        &self,
        xs: &mut [&mut Buffer],
        messages: &[&Buffer],
        weights: &[[&Weight; 4]],
        dim: usize,
        rank: usize,
        eps: f32,
    ) {
        if xs.len() > GATES {
            let (first, rest) = xs.split_at_mut(GATES);
            self.gates(first, &messages[..GATES], &weights[..GATES], dim, rank, eps);
            return self.gates(rest, &messages[GATES..], &weights[GATES..], dim, rank, eps);
        }
        assert!(xs.len() == messages.len() && xs.len() == weights.len());
        assert!(
            dim.is_multiple_of(8) && rank.is_multiple_of(8) && rank <= 64,
            "the kernel takes a gate of up to 64 dimensions, in eights"
        );
        let len = xs[0].len;
        let n = len / dim;
        let mut params = GateJobs {
            x: [0; GATES],
            message: [0; GATES],
            down_input: [0; GATES],
            down_message: [0; GATES],
            gain: [0; GATES],
            forget: [0; GATES],
            n: n as u32,
            dim: dim as u32,
            rank: rank as u32,
            eps,
            split: 0,
        };
        // Each gate's input and message as halves, where a batch's go
        // down through the tile kernels.
        let groups = n.div_ceil(GATE_TOKENS) as u32;
        let tiles = self
            .kernels
            .f16
            .as_ref()
            .filter(|_| groups > 1 && weights.iter().all(|w| w[0].halves.is_some() && w[1].halves.is_some()));
        let mut halves = [(0, 0); GATES];
        for (j, x) in xs.iter_mut().enumerate() {
            assert!(x.len == n * dim && messages[j].len == x.len);
            for (w, shape) in weights[j].iter().zip([[rank, dim], [rank, dim], [dim, rank], [dim, rank]])
            {
                assert!(
                    w.format == Format::Bf16 && w.shape == shape,
                    "a gate's weights are bf16"
                );
            }
            let [down_input, down_message, gain, forget] = weights[j];
            if tiles.is_some() {
                halves[j] = (self.halves(x), self.halves(messages[j]));
            }
            params.message[j] = self.floats(messages[j]);
            params.x[j] = self.floats_mut(x);
            params.down_input[j] = self.ptr(&down_input.buf);
            params.down_message[j] = self.ptr(&down_message.buf);
            params.gain[j] = self.ptr(&gain.buf);
            params.forget[j] = self.ptr(&forget.buf);
        }
        // The tokens' few dimensions, between the two kernels.
        let mut down = self.gate_down.borrow_mut();
        let floats = GATES * n * (2 * rank + 4);
        if down.as_ref().is_none_or(|down| down.len() < floats) {
            // Nothing may still be reading the one this replaces.
            self.stream.synchronize().expect("waiting for the GPU");
            *down = Some(
                self.stream
                    .alloc_zeros(floats)
                    .expect("out of GPU memory"),
            );
        }
        let down = self.ptr(down.as_ref().expect("allocated above"));
        let jobs = xs.len() as u32;
        if let Some(tiles) = tiles {
            // A batch's projections down are matrix multiplications on
            // the tensor cores, beside what normalizes each row.
            params.split = 1;
            let floats = |count: usize| (count * size_of::<f32>()) as u64;
            let norms = down + floats(2 * GATES * n * rank);
            self.launch(
                &self.kernels.gate_norms,
                &[norms],
                &params,
                (n as u32, jobs),
                256,
            );
            for (j, &(x, message)) in halves[..xs.len()].iter().enumerate() {
                let [down_input, down_message, ..] = weights[j];
                let out = down + floats(j * n * rank);
                self.matmul_tiles(tiles, out, down_input, x, 0, n, Products::Floats);
                let out = down + floats((GATES + j) * n * rank);
                self.matmul_tiles(tiles, out, down_message, message, 0, n, Products::Floats);
            }
        } else {
            // No more tokens than a block takes have a block to a row of
            // the projections down, rather than a warp.
            let (kernel, blocks) = match groups {
                1 => (&self.kernels.gate_down_few, rank),
                _ => (&self.kernels.gate_down, rank / 8),
            };
            self.launch(kernel, &[down], &params, (blocks as u32 * jobs, groups), 256);
        }
        self.launch(
            &self.kernels.gate_up,
            &[down],
            &params,
            (dim.div_ceil(256) as u32 * jobs, groups),
            256,
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
        let params = short_conv_params(x, state, weight, dynamic);
        assert_eq!(out.len, x.len);
        // Without taps of the heads' own, the kernel is given none, and
        // reads none.
        let taps = dynamic.map_or(0, |(dynamic, _)| self.floats(dynamic));
        let (x, state) = (self.floats(x), self.floats_mut(state));
        self.launch(
            &self.kernels.short_conv,
            &[self.floats_out(out), x, state, self.floats(weight), taps],
            &params,
            self.grid(out.len, 256),
            256,
        );
        self.short_conv_state(state, x, &params);
    }

    fn short_conv_add(
        &self,
        x: &mut Buffer,
        input: &Buffer,
        state: &mut Buffer,
        weight: &Buffer,
        _scratch: &mut Buffer,
    ) {
        let params = ShortConvParams {
            add: 1,
            ..short_conv_params(input, state, weight, None)
        };
        assert_eq!(x.len, input.len);
        let (input, state) = (self.floats(input), self.floats_mut(state));
        self.launch(
            &self.kernels.short_conv,
            &[self.floats_mut(x), input, state, self.floats(weight), 0],
            &params,
            self.grid(x.len, 256),
            256,
        );
        self.short_conv_state(state, input, &params);
    }

    fn short_conv_store(
        &self,
        cache: &mut Cache,
        pos: usize,
        rows: usize,
        n: usize,
        channels: usize,
        x: &Buffer,
        state: &mut Buffer,
        weight: &Buffer,
        dynamic: Option<(&Buffer, usize)>,
        scale: &Buffer,
        kept: Kept,
        scratch: &mut Buffer,
        wrapped: &mut Buffer,
    ) {
        let conv = short_conv_params(x, state, weight, dynamic);
        assert!(conv.n as usize == n && conv.channels as usize == channels);
        let (dim, kind, value) = match kept {
            Kept::Clipped { dim, threshold } => (dim, 0, threshold),
            Kept::Normed { eps } => (scale.len, 1, eps),
        };
        // A block is a head, and a warp of it a block of the cache.
        if !dim.is_multiple_of(CACHE_BLOCK) || dim > 1024 || !channels.is_multiple_of(dim) {
            return gpu::short_conv_store(
                self, cache, pos, rows, n, channels, x, state, weight, dynamic, scale, kept,
                scratch, wrapped,
            );
        }
        assert!(n <= rows && rows * channels <= cache.len);
        assert!(scale.len.is_multiple_of(dim) && channels.is_multiple_of(scale.len));
        let params = ConvStoreParams {
            n: conv.n,
            channels: conv.channels,
            taps: conv.taps,
            heads: conv.heads,
            dim: dim as u32,
            width: scale.len as u32,
            kept: kind,
            pos: (pos % rows) as u32,
            rows: rows as u32,
            value,
        };
        let taps = dynamic.map_or(0, |(dynamic, _)| self.floats(dynamic));
        let (x, state) = (self.floats(x), self.floats_mut(state));
        self.launch(
            &self.kernels.conv_store,
            &[
                self.ptr(&cache.quants),
                self.ptr(&cache.scales),
                x,
                state,
                self.floats(weight),
                taps,
                self.floats(scale),
            ],
            &params,
            self.grid(n * channels, dim),
            dim as u32,
        );
        self.short_conv_state(state, x, &conv);
    }

    fn clip(&self, x: &mut Buffer, scale: &Buffer, dim: usize, threshold: f32) {
        assert!(x.len.is_multiple_of(scale.len) && scale.len.is_multiple_of(dim));
        let params = ClipParams {
            width: scale.len as u32,
            dim: dim as u32,
            threshold,
        };
        // A warp per head.
        self.launch(
            &self.kernels.clip,
            &[self.floats_mut(x), self.floats(scale)],
            &params,
            self.grid(x.len / dim, 1),
            32,
        );
    }

    fn clips(&self, jobs: &mut [(&mut Buffer, &Buffer, usize, f32)]) {
        // One grid for a few tokens, whose kernels cost more to start
        // than to run; a batch's are faster each on its own.
        let few = jobs
            .iter()
            .all(|(x, scale, ..)| x.len <= TILE_TOKENS_NARROW * scale.len);
        if jobs.len() > JOBS || !few {
            for (x, scale, dim, threshold) in jobs {
                self.clip(x, scale, *dim, *threshold);
            }
            return;
        }
        // A warp per head, the heads of one job after another's.
        let mut params = ClipJobs {
            x: [0; JOBS],
            scale: [0; JOBS],
            first: [0; JOBS],
            width: [1; JOBS],
            dim: [1; JOBS],
            threshold: [0.0; JOBS],
        };
        let mut heads = 0;
        for (j, (x, scale, dim, threshold)) in jobs.iter_mut().enumerate() {
            assert!(x.len.is_multiple_of(scale.len) && scale.len.is_multiple_of(*dim));
            params.x[j] = self.floats_mut(x);
            params.scale[j] = self.floats(scale);
            params.first[j] = heads;
            params.width[j] = scale.len as u32;
            params.dim[j] = *dim as u32;
            params.threshold[j] = *threshold;
            heads += (x.len / *dim) as u32;
        }
        for first in &mut params.first[jobs.len()..] {
            *first = heads;
        }
        self.launch(&self.kernels.clips, &[], &params, (heads, 1), 32);
    }

    fn normalizes(
        &self,
        outs: &mut [&mut Buffer],
        xs: &[&Buffer],
        weights: &[&Buffer],
        len: usize,
        eps: f32,
    ) {
        let dim = weights[0].len;
        // Without tensor cores nothing reads halves, and a batch's
        // normalizations are faster each on its own.
        if self.kernels.q4_0.tiles.is_none()
            || len > TILE_TOKENS_NARROW * dim
            || outs.len() > JOBS
            || weights.iter().any(|weight| weight.len != dim)
        {
            for (j, out) in outs.iter_mut().enumerate() {
                self.normalize(out, xs[j], weights[j], len, eps);
            }
            return;
        }
        assert!(len.is_multiple_of(dim));
        let mut params = NormJobs {
            out: [0; JOBS],
            x: [0; JOBS],
            weight: [0; JOBS],
            dim: dim as u32,
            eps,
        };
        for (j, out) in outs.iter_mut().enumerate() {
            assert!(out.len == len && len <= xs[j].len);
            params.x[j] = self.floats(xs[j]);
            params.weight[j] = self.floats(weights[j]);
            params.out[j] = self.halves_of(out);
            out.form.set(Form::Halves);
        }
        self.launch(
            &self.kernels.normalizes_f16,
            &[],
            &params,
            ((len / dim) as u32, outs.len() as u32),
            256,
        );
    }

    fn scale_rows(&self, x: &mut Buffer, factors: &Buffer) {
        assert!(x.len.is_multiple_of(factors.len));
        let params = RowsParams {
            len: x.len as u32,
            width: (x.len / factors.len) as u32,
            k: 0,
        };
        self.launch(
            &self.kernels.scale_rows,
            &[self.floats_mut(x), self.floats(factors)],
            &params,
            self.grid(x.len, 256),
            256,
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
        out.form.set(Form::Floats);
        let window = window.min(u32::MAX as usize);
        let params = AttentionRelParams {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            n_kv_heads: n_kv_heads as u32,
            pos: pos as u32,
            first: 0,
            chunks: 0,
            chunk: CHUNK as u32,
            count: n as u32,
            base: 0,
            window: window as u32,
            ring: rows as u32,
            n_rel: n_rel as u32,
            extent: (bias.len / n_rel) as u32,
        };
        let buffers = [
            self.ptr(&out.buf),
            self.floats(q),
            self.ptr(&k_cache.quants),
            self.ptr(&k_cache.scales),
            self.ptr(&v_cache.quants),
            self.ptr(&v_cache.scales),
            self.ptr(&self.partials),
            self.floats(rel),
            self.floats(bias),
            self.floats(sinks),
        ];
        if let Some((tiles, combine)) = &self.kernels.attention_rel_tiles
            && head_dim == ATTENTION_REL_TILE_DIM
            && n_rel.is_multiple_of(4)
            && n_rel <= ATTENTION_TILE_REL
            && (n > 1 || pos + 1 >= ATTENTION_TILE_DEPTH)
        {
            // As `attention_tiles` has it: the positions from where the
            // first token's window starts are cut into chunks, so that
            // there are blocks enough to fill the GPU.
            let group = n_heads / n_kv_heads;
            let narrow = ATTENTION_TILE_LINES[0] / group;
            let (kernel, lines, threads) = if n <= 2 * narrow {
                (&tiles[0], ATTENTION_TILE_LINES[0], 128)
            } else {
                (&tiles[1], ATTENTION_TILE_LINES[1], 256)
            };
            let tokens = lines / group;
            let base = (pos + 1).saturating_sub(window);
            let span = pos + n - base;
            // A token alone has few blocks, each a chain of steps one
            // after another: shorter chunks than a batch's make it more.
            let least = if n == 1 {
                ATTENTION_TILE_TOKEN_CHUNK
            } else {
                ATTENTION_TILE_CHUNK
            };
            let chunks = ATTENTION_TILE_BLOCKS
                .div_ceil(n.div_ceil(tokens) * n_kv_heads)
                .min(span / least)
                .max(1);
            let chunk = span.div_ceil(chunks).next_multiple_of(ATTENTION_TILE_KEYS);
            let chunks = span.div_ceil(chunk);
            let per_token = n_heads * chunks * (head_dim + 2);
            let per_dispatch = PARTIALS / per_token / tokens * tokens;
            assert!(
                per_dispatch > 0,
                "attention over {span} positions needs more partials than {PARTIALS}"
            );
            for first in (0..n).step_by(per_dispatch) {
                let count = per_dispatch.min(n - first);
                let params = AttentionRelParams {
                    first: first as u32,
                    chunks: chunks as u32,
                    chunk: chunk as u32,
                    count: count as u32,
                    base: base as u32,
                    ..params
                };
                self.launch_shared(
                    kernel,
                    &buffers,
                    &params,
                    self.grid(count.div_ceil(tokens) * n_kv_heads * chunks, 1),
                    threads,
                    attention_tile_shared(lines, head_dim),
                );
                if chunks > 1 {
                    self.launch(
                        combine,
                        &[buffers[0], buffers[6], buffers[9]],
                        &params,
                        self.grid(count * n_heads, 1),
                        head_dim as u32,
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
            let params = AttentionRelParams {
                first: first as u32,
                chunks: chunks as u32,
                count: tokens as u32,
                ..params
            };
            self.launch(
                &self.kernels.attention_rel,
                &buffers,
                &params,
                self.grid(tokens * n_kv_heads * chunks, 1),
                256,
            );
            if chunks > 1 {
                self.launch(
                    &self.kernels.attention_rel_combine,
                    &[buffers[0], buffers[6], buffers[9]],
                    &params,
                    self.grid(tokens * n_heads, 1),
                    256,
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
        let params = RouteParams {
            experts: experts as u32,
            k: k as u32,
            scale,
        };
        // A warp per token.
        self.launch(
            &self.kernels.route,
            &[
                self.floats_out(ids),
                self.floats_out(weights),
                self.floats(logits),
                self.floats(bias),
            ],
            &params,
            self.grid(n, 1),
            32,
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
        let kernels = match w.format {
            Format::Nvfp4 => &self.kernels.expert_nvfp4,
            Format::Q8_0 => &self.kernels.expert_q8_0,
            Format::Bf16 | Format::Q4_0 => panic!("the experts' matrices are NVFP4 or Q8_0"),
        };
        let params = ExpertParams {
            rows: rows as u32,
            cols: cols as u32,
            tiles: cols.div_ceil(TILE_COLS) as u32,
            groups: (rows / TILE_ROWS) as u32,
            share: if shared { k as u32 } else { 1 },
            direct: 0,
            beta: 0.0,
            linear_beta: 0.0,
        };
        // On the tensor cores where each expert's rows are whole groups
        // of the tiles', which the experts' matrices were tiled in as one.
        if let Some(tiles) = &kernels.tiles
            && rows.is_multiple_of(TILE_ROWS)
        {
            let x = self.halves(x);
            let out = self.floats_out(out);
            let plan = self.expert_plan(ids, scale.len);
            return self.expert_tiles(tiles, out, w, scale, x, ids, plan, rows / TILE_ROWS, params);
        }
        self.launch(
            &kernels.rows,
            &[
                self.floats_out(out),
                self.ptr(&w.buf),
                self.ptr(&w.scales),
                self.floats(x),
                self.floats(ids),
                self.floats(scale),
            ],
            &params,
            (pairs as u32, rows.div_ceil(TILE_ROWS) as u32),
            256,
        );
    }

    fn expert_ffn(
        &self,
        out: &mut Buffer,
        gate_up: &Weight,
        gate_up_scale: &Buffer,
        down: &Weight,
        down_scale: &Buffer,
        x: &Buffer,
        ids: &Buffer,
        k: usize,
        width: usize,
        beta: f32,
        linear_beta: f32,
        scratch: &mut Buffer,
        act: &mut Buffer,
    ) {
        let experts = gate_up_scale.len;
        let rows = down.shape[0] / experts;
        let cols = gate_up.shape[1];
        let pairs = ids.len;
        let kernels = |w: &Weight| match w.format {
            Format::Nvfp4 => Some(&self.kernels.expert_nvfp4),
            Format::Q8_0 => Some(&self.kernels.expert_q8_0),
            Format::Bf16 | Format::Q4_0 => None,
        };
        // On the tensor cores, where each expert's projections are whole
        // groups of the tiles' rows: the pairs are placed once for both
        // matmuls, and the first leaves the activation, as halves.
        let (Some(situ), Some(tiles)) = (
            kernels(gate_up).and_then(|kernels| kernels.situ.as_ref()),
            kernels(down).and_then(|kernels| kernels.tiles.as_ref()),
        ) else {
            return gpu::expert_ffn(
                self, out, gate_up, gate_up_scale, down, down_scale, x, ids, k, width, beta,
                linear_beta, scratch, act,
            );
        };
        if !width.is_multiple_of(TILE_ROWS) || !rows.is_multiple_of(TILE_ROWS) {
            return gpu::expert_ffn(
                self, out, gate_up, gate_up_scale, down, down_scale, x, ids, k, width, beta,
                linear_beta, scratch, act,
            );
        }
        assert!(gate_up.shape[0] == experts * 2 * width && down_scale.len == experts);
        assert_eq!(down.shape[1], width);
        assert!(x.len == pairs / k * cols && act.len == pairs * width);
        assert_eq!(out.len, pairs * rows);
        let plan = self.expert_plan(ids, experts);
        let x = self.halves(x);
        let halves = self.halves_of(act);
        act.form.set(Form::Halves);
        let params = ExpertParams {
            rows: width as u32,
            cols: cols as u32,
            tiles: cols.div_ceil(TILE_COLS) as u32,
            groups: (2 * width / TILE_ROWS) as u32,
            share: k as u32,
            direct: 0,
            beta,
            linear_beta,
        };
        self.expert_tiles(
            situ,
            halves,
            gate_up,
            gate_up_scale,
            x,
            ids,
            plan,
            width / TILE_ROWS,
            params,
        );
        let params = ExpertParams {
            rows: rows as u32,
            cols: width as u32,
            tiles: width.div_ceil(TILE_COLS) as u32,
            groups: (rows / TILE_ROWS) as u32,
            share: 1,
            direct: 0,
            beta: 0.0,
            linear_beta: 0.0,
        };
        let out = self.floats_out(out);
        self.expert_tiles(
            tiles,
            out,
            down,
            down_scale,
            halves,
            ids,
            plan,
            rows / TILE_ROWS,
            params,
        );
    }

    fn situ(&self, out: &mut Buffer, gate_up: &Buffer, width: usize, beta: f32, linear_beta: f32) {
        assert!(gate_up.len == 2 * out.len && out.len.is_multiple_of(width));
        let params = SituParams {
            len: out.len as u32,
            width: width as u32,
            beta,
            linear_beta,
        };
        // As halves too where the tensor cores are, for the experts'
        // matmul it is for.
        self.launch(
            &self.kernels.situ,
            &[
                self.floats_out(out),
                self.halves_too(out),
                self.floats(gate_up),
            ],
            &params,
            self.grid(out.len, 256),
            256,
        );
    }

    fn weighted_sum(&self, out: &mut Buffer, y: &Buffer, weights: &Buffer) {
        let width = y.len / weights.len;
        assert!(y.len == weights.len * width && out.len.is_multiple_of(width));
        let params = RowsParams {
            len: out.len as u32,
            width: width as u32,
            k: (weights.len / (out.len / width)) as u32,
        };
        self.launch(
            &self.kernels.weighted_sum,
            &[self.floats_out(out), self.floats(y), self.floats(weights)],
            &params,
            self.grid(out.len, 256),
            256,
        );
    }
}

#[cfg(test)]
mod tests {
    // With tensor cores, every matmul through block-quantized weights is
    // in half precision; without them, none is.
    check_cuda!(
        super::Cuda::new();
        write_and_read_round_trip,
        alloc_is_zeroed_and_copy_moves_ranges,
        matmul_matches_cpu,
        matmuls_match_cpu,
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
        normalizes_and_clips_match_cpu,
        swiglu_matches_cpu,
        matmul_add_matches_cpu,
        conv_matches_cpu,
        delta_net_matches_cpu,
        expert_matmul_matches_cpu,
        expert_ffn_matches_cpu,
        mix_matches_cpu,
        mixes_match_cpu,
        short_conv_matches_cpu,
        short_conv_fused_matches_cpu,
        attention_rel_matches_cpu,
        route_matches_cpu,
        rows_and_gates_match_cpu,
        block_conv_matches_cpu,
        read_capped_matches_cpu,
        gate_matches_cpu,
        copy_cache_copies_a_prefix,
        resize_keeps_capacity
    );

    mod without_tensor_cores {
        check_cuda!(
            super::super::Cuda::without_tensor_cores();
            q4_0_matmul_matches_cpu,
            q8_0_matmul_matches_cpu,
            normalize_matches_cpu,
            swiglu_matches_cpu,
            matmul_add_matches_cpu,
            expert_matmul_matches_cpu,
            expert_ffn_matches_cpu,
            attention_rel_matches_cpu,
            mixes_match_cpu,
            short_conv_fused_matches_cpu,
            gate_matches_cpu,
            rows_and_gates_match_cpu
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

    /// Times the experts' matmul at Inkling's shapes.
    #[test]
    #[ignore]
    fn expert_matmul_speed() {
        match super::Cuda::new() {
            Ok(gpu) => crate::gpu::tests::expert_matmul_speed(&gpu),
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
