//! The GPU of the platform, through whichever of its APIs opens it: CUDA
//! when an NVIDIA driver is present and Vulkan otherwise, or Metal on a
//! Mac. The device, and its buffers, weights, and caches, are the opened
//! API's own, wrapped; every operation unwraps them and goes straight
//! through.

use std::{error::Error, fmt, str::FromStr};

#[cfg(target_vendor = "apple")]
use crate::gpu::metal;
use crate::gpu::{Device, Tensor};
#[cfg(not(target_vendor = "apple"))]
use crate::gpu::{cuda, vulkan};

/// An API a GPU is driven through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Api {
    Cuda,
    Vulkan,
    Metal,
}

impl Api {
    pub fn name(self) -> &'static str {
        match self {
            Api::Cuda => "CUDA",
            Api::Vulkan => "Vulkan",
            Api::Metal => "Metal",
        }
    }
}

impl fmt::Display for Api {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Api {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "cuda" => Ok(Api::Cuda),
            "vulkan" => Ok(Api::Vulkan),
            "metal" => Ok(Api::Metal),
            _ => Err(format!(
                "unknown API '{s}' (expected cuda, vulkan, or metal)"
            )),
        }
    }
}

/// The GPU, through the API that opened it.
// One is opened for a run, so the variants' sizes do not matter.
#[allow(clippy::large_enum_variant)]
pub enum Gpu {
    #[cfg(not(target_vendor = "apple"))]
    Cuda(cuda::Cuda),
    #[cfg(not(target_vendor = "apple"))]
    Vulkan(vulkan::Vulkan),
    #[cfg(target_vendor = "apple")]
    Metal(metal::Metal),
}

/// A vector of f32 activations in the GPU's memory.
pub enum Buffer {
    #[cfg(not(target_vendor = "apple"))]
    Cuda(cuda::Buffer),
    #[cfg(not(target_vendor = "apple"))]
    Vulkan(vulkan::Buffer),
    #[cfg(target_vendor = "apple")]
    Metal(metal::Buffer),
}

/// A weight matrix in the GPU's memory.
pub enum Weight {
    #[cfg(not(target_vendor = "apple"))]
    Cuda(cuda::Weight),
    #[cfg(not(target_vendor = "apple"))]
    Vulkan(vulkan::Weight),
    #[cfg(target_vendor = "apple")]
    Metal(metal::Weight),
}

/// A key or value cache in the GPU's memory.
pub enum Cache {
    #[cfg(not(target_vendor = "apple"))]
    Cuda(cuda::Cache),
    #[cfg(not(target_vendor = "apple"))]
    Vulkan(vulkan::Cache),
    #[cfg(target_vendor = "apple")]
    Metal(metal::Cache),
}

/// The buffer, weight, or cache of the API `$api` that `$x` wraps.
macro_rules! unwrap {
    ($api:ident, [$buffer:expr]) => {
        match $buffer {
            Buffer::$api(buffer) => buffer,
            #[allow(unreachable_patterns)]
            _ => unreachable!("a buffer of another API"),
        }
    };
    ($api:ident, {$weight:expr}) => {
        match $weight {
            Weight::$api(weight) => weight,
            #[allow(unreachable_patterns)]
            _ => unreachable!("a weight of another API"),
        }
    };
    ($api:ident, ($cache:expr)) => {
        match $cache {
            Cache::$api(cache) => cache,
            #[allow(unreachable_patterns)]
            _ => unreachable!("a cache of another API"),
        }
    };
    ($api:ident, $other:expr) => {
        $other
    };
}

/// Calls the opened API's `$method` with the arguments, unwrapping the
/// buffers among them, written as `[buffer]`, the weights, written as
/// `{weight}`, and the caches, written as `(cache)`; and wraps what it
/// returns as `$wrap`, when given.
macro_rules! dispatch {
    ($gpu:expr, $method:ident($($arg:tt),*)) => {
        match $gpu {
            #[cfg(not(target_vendor = "apple"))]
            Gpu::Cuda(gpu) => gpu.$method($(unwrap!(Cuda, $arg)),*),
            #[cfg(not(target_vendor = "apple"))]
            Gpu::Vulkan(gpu) => gpu.$method($(unwrap!(Vulkan, $arg)),*),
            #[cfg(target_vendor = "apple")]
            Gpu::Metal(gpu) => gpu.$method($(unwrap!(Metal, $arg)),*),
        }
    };
    ($gpu:expr, $method:ident($($arg:tt),*) => $wrap:ident) => {
        match $gpu {
            #[cfg(not(target_vendor = "apple"))]
            Gpu::Cuda(gpu) => $wrap::Cuda(gpu.$method($(unwrap!(Cuda, $arg)),*)),
            #[cfg(not(target_vendor = "apple"))]
            Gpu::Vulkan(gpu) => $wrap::Vulkan(gpu.$method($(unwrap!(Vulkan, $arg)),*)),
            #[cfg(target_vendor = "apple")]
            Gpu::Metal(gpu) => $wrap::Metal(gpu.$method($(unwrap!(Metal, $arg)),*)),
        }
    };
}

impl Gpu {
    /// Opens the GPU through `api`, or, given none, through CUDA when an
    /// NVIDIA driver is present and through the platform's API otherwise.
    pub fn open(api: Option<Api>) -> Result<Self, Box<dyn Error>> {
        match api {
            #[cfg(not(target_vendor = "apple"))]
            Some(Api::Cuda) => Ok(Gpu::Cuda(cuda::Cuda::new()?)),
            #[cfg(not(target_vendor = "apple"))]
            Some(Api::Vulkan) => Ok(Gpu::Vulkan(vulkan::Vulkan::new()?)),
            #[cfg(target_vendor = "apple")]
            Some(Api::Metal) => Ok(Gpu::Metal(metal::Metal::new()?)),
            Some(api) => Err(format!("{api} is not available on this platform").into()),
            #[cfg(not(target_vendor = "apple"))]
            None => match cuda::Cuda::new() {
                Ok(gpu) => Ok(Gpu::Cuda(gpu)),
                Err(cuda) => vulkan::Vulkan::new()
                    .map(Gpu::Vulkan)
                    .map_err(|vulkan| format!("CUDA: {cuda}; Vulkan: {vulkan}").into()),
            },
            #[cfg(target_vendor = "apple")]
            None => Ok(Gpu::Metal(metal::Metal::new()?)),
        }
    }

    /// Opens the GPU through whichever API does: see [`Gpu::open`].
    pub fn new() -> Result<Self, Box<dyn Error>> {
        Self::open(None)
    }

    /// The API the GPU was opened through.
    pub fn api(&self) -> Api {
        match self {
            #[cfg(not(target_vendor = "apple"))]
            Gpu::Cuda(_) => Api::Cuda,
            #[cfg(not(target_vendor = "apple"))]
            Gpu::Vulkan(_) => Api::Vulkan,
            #[cfg(target_vendor = "apple")]
            Gpu::Metal(_) => Api::Metal,
        }
    }
}

impl Device for Gpu {
    type Buffer = Buffer;
    type Weight = Weight;
    type Cache = Cache;

    fn name(&self) -> String {
        dispatch!(self, name())
    }

    fn memory(&self) -> Option<u64> {
        dispatch!(self, memory())
    }

    fn upload(&self, tensor: Tensor) -> Weight {
        dispatch!(self, upload(tensor) => Weight)
    }

    fn alloc(&self, len: usize) -> Buffer {
        dispatch!(self, alloc(len) => Buffer)
    }

    fn alloc_cache(&self, len: usize) -> Cache {
        dispatch!(self, alloc_cache(len) => Cache)
    }

    fn resize(&self, buf: &mut Buffer, len: usize) {
        dispatch!(self, resize([buf], len))
    }

    fn read(&self, buf: &Buffer) -> Vec<f32> {
        dispatch!(self, read([buf]))
    }

    fn write(&self, buf: &mut Buffer, data: &[f32]) {
        dispatch!(self, write([buf], data))
    }

    fn copy(
        &self,
        dst: &mut Buffer,
        dst_offset: usize,
        src: &Buffer,
        src_offset: usize,
        len: usize,
    ) {
        dispatch!(self, copy([dst], dst_offset, [src], src_offset, len))
    }

    fn store(&self, cache: &mut Cache, offset: usize, src: &Buffer) {
        dispatch!(self, store((cache), offset, [src]))
    }

    fn read_cache(&self, cache: &Cache, len: usize) -> Vec<u8> {
        dispatch!(self, read_cache((cache), len))
    }

    fn write_cache(&self, cache: &mut Cache, data: &[u8]) {
        dispatch!(self, write_cache((cache), data))
    }

    fn matmul(&self, out: &mut Buffer, w: &Weight, x: &Buffer) {
        dispatch!(self, matmul([out], { w }, [x]))
    }

    fn add(&self, x: &mut Buffer, y: &Buffer) {
        dispatch!(self, add([x], [y]))
    }

    fn rmsnorm(&self, x: &mut Buffer, weight: &Buffer, eps: f32) {
        dispatch!(self, rmsnorm([x], [weight], eps))
    }

    fn l2norm(&self, x: &mut Buffer, dim: usize, eps: f32) {
        dispatch!(self, l2norm([x], dim, eps))
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
        dispatch!(self, rope([x], [table], pos, n_heads, head_dim, rot_dim))
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
        dispatch!(
            self,
            attention(
                [out],
                [q],
                (k_cache),
                (v_cache),
                pos,
                n_heads,
                head_dim,
                n_kv_heads
            )
        )
    }

    fn silu_mul(&self, gate: &mut Buffer, up: &Buffer) {
        dispatch!(self, silu_mul([gate], [up]))
    }

    fn sigmoid_mul(&self, x: &mut Buffer, gate: &Buffer) {
        dispatch!(self, sigmoid_mul([x], [gate]))
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
        dispatch!(
            self,
            conv([q], [k], [v], [state_out], [x], [state], [weight])
        )
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
        dispatch!(
            self,
            delta_net(
                [out],
                [q],
                [k],
                [v],
                [gates],
                [decay],
                [state],
                n_k_heads,
                n_v_heads,
                head_dim
            )
        )
    }
}
