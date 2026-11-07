//! The model on an NVIDIA GPU, through CUDA: every operation is a
//! hand-written kernel in `kernels/`, compiled from CUDA C by NVRTC when
//! the device opens, for the GPU's own architecture.
//!
//! Operations go to one stream in the order they are called, which runs
//! them in that order, so nothing needs a barrier; reading a buffer back
//! waits for the stream. The driver and NVRTC are loaded when the device
//! opens, so a build needs no CUDA, and a machine without them gets an
//! error from [`Cuda::new`] rather than a device.

use std::{error::Error, slice, sync::Arc};

use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchConfig,
        PushKernelArg, sys,
    },
    nvrtc::{self, CompileError, CompileOptions, Ptx},
};

use crate::gpu::{CACHE_BLOCK, CONV_KERNEL, Device, Tensor, quant};

/// Tokens a block of the bf16 and block-quantized matmul kernels takes at
/// once, and rows a block of the block-quantized ones does, a warp each.
const MATMUL_TOKENS: usize = 8;
const QUANT_ROWS: usize = 8;

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
const KERNELS: [(&str, &str); 15] = [
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
    ("matmul_q8_0", include_str!("kernels/matmul_q8_0.cu")),
    ("matmul_q4_0", include_str!("kernels/matmul_q4_0.cu")),
];

/// A vector of f32 activations in device memory.
pub struct Buffer {
    buf: CudaSlice<f32>,
    len: usize,
    cap: usize,
}

/// A cache of activations in device memory, in blocks of 8-bit integers,
/// and the blocks' scales as f16 bits.
pub struct Cache {
    quants: CudaSlice<i8>,
    scales: CudaSlice<u16>,
    len: usize,
}

/// A weight matrix in device memory: its weights, and for the
/// block-quantized formats their scales in a buffer of their own.
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

struct Kernels {
    matmul_bf16: CudaFunction,
    matmul_q4_0: CudaFunction,
    matmul_q8_0: CudaFunction,
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
        let ptx = compile(major * 10 + minor).map_err(|e| format!("{name}: {e}"))?;
        let module = ctx.load_module(ptx)?;
        let stream = ctx.default_stream();
        let kernel = |name: &str| module.load_function(name);
        let kernels = Kernels {
            matmul_bf16: kernel("matmul_bf16")?,
            matmul_q4_0: kernel("matmul_q4_0")?,
            matmul_q8_0: kernel("matmul_q8_0")?,
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
        assert!(
            grid.1 as usize <= MAX_GRID_Y,
            "{} rows of blocks is more than the GPU allows",
            grid.1
        );
        let config = LaunchConfig {
            grid_dim: (grid.0, grid.1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
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
}

/// Compiles the kernels to PTX for a GPU of compute capability `cc`, as
/// `major * 10 + minor`: for the newest architecture NVRTC knows that is
/// no newer than the GPU's, since PTX runs on the architectures after its
/// own.
fn compile(cc: i32) -> Result<Ptx, Box<dyn Error>> {
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
    let source = KERNELS.map(|(_, source)| source).join("\n");
    let mut log = String::new();
    for &(_, arch) in ARCHS.iter().filter(|&&(known, _)| known <= cc) {
        let options = CompileOptions {
            arch: Some(arch),
            name: Some("dwim.cu".into()),
            ..Default::default()
        };
        match nvrtc::compile_ptx_with_opts(&source, options) {
            Ok(ptx) => return Ok(ptx),
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

// Plain structs of 32-bit fields, passed to the kernels by value.
unsafe impl DeviceRepr for MatmulParams {}
unsafe impl DeviceRepr for RmsnormParams {}
unsafe impl DeviceRepr for RopeParams {}
unsafe impl DeviceRepr for ConvParams {}
unsafe impl DeviceRepr for DeltaNetParams {}
unsafe impl DeviceRepr for AttentionParams {}
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
        let buf = self
            .stream
            .alloc_zeros::<f32>(len.max(1))
            .expect("out of GPU memory");
        Buffer { buf, len, cap: len }
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
        self.stream
            .clone_dtoh(&buf.buf.slice(..buf.len))
            .expect("reading a buffer")
    }

    fn write(&self, buf: &mut Buffer, data: &[f32]) {
        assert_eq!(buf.len, data.len());
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
                self.ptr(&src.buf),
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
        // One block of 64 threads per row of bf16 weights, or of a warp per
        // row for eight rows of block-quantized weights, across the grid,
        // and the tokens eight at a time, each group of them a row of the
        // grid. The block-quantized kernels take the scales after the
        // quants.
        let (kernel, buffers, per_block, threads): (_, &[u64], _, _) = match w.format {
            Format::Bf16 => (
                &self.kernels.matmul_bf16,
                &[self.ptr(&out.buf), self.ptr(&w.buf), self.ptr(&x.buf)],
                1,
                64,
            ),
            Format::Q4_0 => (
                &self.kernels.matmul_q4_0,
                &[
                    self.ptr(&out.buf),
                    self.ptr(&w.buf),
                    self.ptr(&w.scales),
                    self.ptr(&x.buf),
                ],
                QUANT_ROWS,
                256,
            ),
            Format::Q8_0 => (
                &self.kernels.matmul_q8_0,
                &[
                    self.ptr(&out.buf),
                    self.ptr(&w.buf),
                    self.ptr(&w.scales),
                    self.ptr(&x.buf),
                ],
                QUANT_ROWS,
                256,
            ),
        };
        let count = rows.div_ceil(per_block);
        let width = count.min(MAX_GRID_X);
        let grid = (
            width as u32,
            (count.div_ceil(width) * n.div_ceil(MATMUL_TOKENS)) as u32,
        );
        let params = MatmulParams {
            rows: rows as u32,
            cols: cols as u32,
            n: n as u32,
            stride: width as u32,
            splits: 1,
        };
        self.launch(kernel, buffers, &params, grid, threads);
    }

    fn add(&self, x: &mut Buffer, y: &Buffer) {
        assert_eq!(x.len, y.len);
        let params = LenParams { len: x.len as u32 };
        let grid = self.grid(x.len, 256);
        self.launch(
            &self.kernels.add,
            &[self.ptr(&x.buf), self.ptr(&y.buf)],
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
            &[self.ptr(&x.buf), self.ptr(&weight.buf)],
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
            &[self.ptr(&x.buf)],
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
            &[self.ptr(&x.buf), self.ptr(&table.buf)],
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
                    self.ptr(&q.buf),
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
            &[self.ptr(&gate.buf), self.ptr(&up.buf)],
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
            &[self.ptr(&x.buf), self.ptr(&gate.buf)],
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
                self.ptr(&q.buf),
                self.ptr(&k.buf),
                self.ptr(&v.buf),
                self.ptr(&state_out.buf),
                self.ptr(&x.buf),
                self.ptr(&state.buf),
                self.ptr(&weight.buf),
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
        let grid = self.grid(n_v_heads, 1);
        self.launch(
            &self.kernels.delta_net,
            &[
                self.ptr(&out.buf),
                self.ptr(&q.buf),
                self.ptr(&k.buf),
                self.ptr(&v.buf),
                self.ptr(&gates.buf),
                self.ptr(&decay.buf),
                self.ptr(&state.buf),
            ],
            &params,
            grid,
            256,
        );
    }
}

#[cfg(test)]
mod tests {
    check_against_cpu!(super::Cuda::new());

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
            Ok(gpu) => crate::gpu::tests::attention_speed(&gpu, 64),
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
