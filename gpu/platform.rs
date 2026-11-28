//! The GPU of the platform, through whichever of its APIs opens it: CUDA
//! when an NVIDIA driver is present and Vulkan otherwise, or Metal on a
//! Mac. The device, and its buffers, weights, and caches, are the opened
//! API's own, wrapped; every operation unwraps them and goes straight
//! through.

use std::error::Error;

#[cfg(target_vendor = "apple")]
use crate::gpu::metal;
#[cfg(not(target_vendor = "apple"))]
use crate::gpu::{cuda, vulkan};
use crate::{
    gpu::{Device, Kept, Tensor},
    model::Api,
};

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

    fn batch(&self) -> usize {
        dispatch!(self, batch())
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

    fn read_capped(&self, buf: &mut Buffer, cap: f32) -> Vec<f32> {
        dispatch!(self, read_capped([buf], cap))
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

    fn matmuls(&self, jobs: &mut [(&mut Buffer, &Weight, &Buffer)]) {
        // The jobs are unwrapped one by one, which `dispatch!` cannot.
        macro_rules! matmuls {
            ($api:ident, $gpu:expr) => {{
                let mut jobs: Vec<_> = jobs
                    .iter_mut()
                    .map(|(out, w, x)| {
                        (
                            unwrap!($api, [&mut **out]),
                            unwrap!($api, { *w }),
                            unwrap!($api, [*x]),
                        )
                    })
                    .collect();
                $gpu.matmuls(&mut jobs)
            }};
        }
        match self {
            #[cfg(not(target_vendor = "apple"))]
            Gpu::Cuda(gpu) => matmuls!(Cuda, gpu),
            #[cfg(not(target_vendor = "apple"))]
            Gpu::Vulkan(gpu) => matmuls!(Vulkan, gpu),
            #[cfg(target_vendor = "apple")]
            Gpu::Metal(gpu) => matmuls!(Metal, gpu),
        }
    }

    fn matmul_add(&self, x: &mut Buffer, w: &Weight, input: &Buffer, scratch: &mut Buffer) {
        dispatch!(self, matmul_add([x], { w }, [input], [scratch]))
    }

    fn swiglu(
        &self,
        out: &mut Buffer,
        gate: &Weight,
        up: &Weight,
        x: &Buffer,
        scratch: &mut Buffer,
    ) {
        dispatch!(self, swiglu([out], { gate }, { up }, [x], [scratch]))
    }

    fn normalize(&self, out: &mut Buffer, x: &Buffer, weight: &Buffer, len: usize, eps: f32) {
        dispatch!(self, normalize([out], [x], [weight], len, eps))
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

    fn takes_nvfp4(&self) -> bool {
        dispatch!(self, takes_nvfp4())
    }

    fn copy_cache(&self, dst: &mut Cache, src: &Cache, len: usize) {
        dispatch!(self, copy_cache((dst), (src), len))
    }

    fn mix(&self, out: &mut Buffer, states: &[&Buffer], query: &Buffer, eps: f32) {
        // The states are unwrapped one by one, which `dispatch!` cannot.
        macro_rules! mix {
            ($api:ident, $gpu:expr) => {{
                let states: Vec<_> = states.iter().map(|state| unwrap!($api, [state])).collect();
                $gpu.mix(unwrap!($api, [out]), &states, unwrap!($api, [query]), eps)
            }};
        }
        match self {
            #[cfg(not(target_vendor = "apple"))]
            Gpu::Cuda(gpu) => mix!(Cuda, gpu),
            #[cfg(not(target_vendor = "apple"))]
            Gpu::Vulkan(gpu) => mix!(Vulkan, gpu),
            #[cfg(target_vendor = "apple")]
            Gpu::Metal(gpu) => mix!(Metal, gpu),
        }
    }

    fn mixes(
        &self,
        outs: &mut [&mut Buffer],
        states: &[&Buffer],
        queries: &[&Buffer],
        eps: f32,
        norms: Option<(&[&Buffer], f32)>,
    ) {
        // The buffers are unwrapped one by one, which `dispatch!` cannot.
        macro_rules! mixes {
            ($api:ident, $gpu:expr) => {{
                let mut outs: Vec<_> = outs
                    .iter_mut()
                    .map(|out| unwrap!($api, [&mut **out]))
                    .collect();
                let states: Vec<_> = states.iter().map(|state| unwrap!($api, [state])).collect();
                let queries: Vec<_> = queries.iter().map(|query| unwrap!($api, [query])).collect();
                let norms = norms.map(|(norms, eps)| {
                    let norms: Vec<_> = norms.iter().map(|norm| unwrap!($api, [norm])).collect();
                    (norms, eps)
                });
                let norms = norms.as_ref().map(|(norms, eps)| (norms.as_slice(), *eps));
                $gpu.mixes(&mut outs, &states, &queries, eps, norms)
            }};
        }
        match self {
            #[cfg(not(target_vendor = "apple"))]
            Gpu::Cuda(gpu) => mixes!(Cuda, gpu),
            #[cfg(not(target_vendor = "apple"))]
            Gpu::Vulkan(gpu) => mixes!(Vulkan, gpu),
            #[cfg(target_vendor = "apple")]
            Gpu::Metal(gpu) => mixes!(Metal, gpu),
        }
    }

    fn gated_update(&self, x: &mut Buffer, forget: &Buffer, gain: &Buffer, message: &Buffer) {
        dispatch!(self, gated_update([x], [forget], [gain], [message]))
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
        dispatch!(
            self,
            gate(
                [x],
                [message],
                { down_input },
                { down_message },
                { gain },
                { forget },
                dim,
                rank,
                eps
            )
        )
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
        // The buffers and weights are unwrapped one by one, which
        // `dispatch!` cannot.
        macro_rules! gates {
            ($api:ident, $gpu:expr) => {{
                let mut xs: Vec<_> = xs.iter_mut().map(|x| unwrap!($api, [&mut **x])).collect();
                let messages: Vec<_> = messages
                    .iter()
                    .map(|message| unwrap!($api, [message]))
                    .collect();
                let weights: Vec<_> = weights
                    .iter()
                    .map(|weights| weights.map(|weight| unwrap!($api, { weight })))
                    .collect();
                $gpu.gates(&mut xs, &messages, &weights, dim, rank, eps)
            }};
        }
        match self {
            #[cfg(not(target_vendor = "apple"))]
            Gpu::Cuda(gpu) => gates!(Cuda, gpu),
            #[cfg(not(target_vendor = "apple"))]
            Gpu::Vulkan(gpu) => gates!(Vulkan, gpu),
            #[cfg(target_vendor = "apple")]
            Gpu::Metal(gpu) => gates!(Metal, gpu),
        }
    }

    fn short_conv(
        &self,
        out: &mut Buffer,
        x: &Buffer,
        state: &mut Buffer,
        weight: &Buffer,
        dynamic: Option<(&Buffer, usize)>,
    ) {
        // The heads' taps are unwrapped where there are any, which
        // `dispatch!` cannot.
        macro_rules! short_conv {
            ($api:ident, $gpu:expr) => {{
                let dynamic = dynamic.map(|(taps, heads)| (unwrap!($api, [taps]), heads));
                $gpu.short_conv(
                    unwrap!($api, [out]),
                    unwrap!($api, [x]),
                    unwrap!($api, [state]),
                    unwrap!($api, [weight]),
                    dynamic,
                )
            }};
        }
        match self {
            #[cfg(not(target_vendor = "apple"))]
            Gpu::Cuda(gpu) => short_conv!(Cuda, gpu),
            #[cfg(not(target_vendor = "apple"))]
            Gpu::Vulkan(gpu) => short_conv!(Vulkan, gpu),
            #[cfg(target_vendor = "apple")]
            Gpu::Metal(gpu) => short_conv!(Metal, gpu),
        }
    }

    fn short_conv_add(
        &self,
        x: &mut Buffer,
        input: &Buffer,
        state: &mut Buffer,
        weight: &Buffer,
        scratch: &mut Buffer,
    ) {
        dispatch!(
            self,
            short_conv_add([x], [input], [state], [weight], [scratch])
        )
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
        // The heads' taps are unwrapped where there are any, which
        // `dispatch!` cannot.
        macro_rules! short_conv_store {
            ($api:ident, $gpu:expr) => {{
                let dynamic = dynamic.map(|(taps, heads)| (unwrap!($api, [taps]), heads));
                $gpu.short_conv_store(
                    unwrap!($api, (cache)),
                    pos,
                    rows,
                    n,
                    channels,
                    unwrap!($api, [x]),
                    unwrap!($api, [state]),
                    unwrap!($api, [weight]),
                    dynamic,
                    unwrap!($api, [scale]),
                    kept,
                    unwrap!($api, [scratch]),
                    unwrap!($api, [wrapped]),
                )
            }};
        }
        match self {
            #[cfg(not(target_vendor = "apple"))]
            Gpu::Cuda(gpu) => short_conv_store!(Cuda, gpu),
            #[cfg(not(target_vendor = "apple"))]
            Gpu::Vulkan(gpu) => short_conv_store!(Vulkan, gpu),
            #[cfg(target_vendor = "apple")]
            Gpu::Metal(gpu) => short_conv_store!(Metal, gpu),
        }
    }

    fn clip(&self, x: &mut Buffer, scale: &Buffer, dim: usize, threshold: f32) {
        dispatch!(self, clip([x], [scale], dim, threshold))
    }

    fn clips(&self, jobs: &mut [(&mut Buffer, &Buffer, usize, f32)]) {
        // The jobs are unwrapped one by one, which `dispatch!` cannot.
        macro_rules! clips {
            ($api:ident, $gpu:expr) => {{
                let mut jobs: Vec<_> = jobs
                    .iter_mut()
                    .map(|(x, scale, dim, threshold)| {
                        (
                            unwrap!($api, [&mut **x]),
                            unwrap!($api, [*scale]),
                            *dim,
                            *threshold,
                        )
                    })
                    .collect();
                $gpu.clips(&mut jobs)
            }};
        }
        match self {
            #[cfg(not(target_vendor = "apple"))]
            Gpu::Cuda(gpu) => clips!(Cuda, gpu),
            #[cfg(not(target_vendor = "apple"))]
            Gpu::Vulkan(gpu) => clips!(Vulkan, gpu),
            #[cfg(target_vendor = "apple")]
            Gpu::Metal(gpu) => clips!(Metal, gpu),
        }
    }

    fn normalizes(
        &self,
        outs: &mut [&mut Buffer],
        xs: &[&Buffer],
        weights: &[&Buffer],
        len: usize,
        eps: f32,
    ) {
        // The buffers are unwrapped one by one, which `dispatch!` cannot.
        macro_rules! normalizes {
            ($api:ident, $gpu:expr) => {{
                let mut outs: Vec<_> = outs
                    .iter_mut()
                    .map(|out| unwrap!($api, [&mut **out]))
                    .collect();
                let xs: Vec<_> = xs.iter().map(|x| unwrap!($api, [x])).collect();
                let weights: Vec<_> = weights
                    .iter()
                    .map(|weight| unwrap!($api, [weight]))
                    .collect();
                $gpu.normalizes(&mut outs, &xs, &weights, len, eps)
            }};
        }
        match self {
            #[cfg(not(target_vendor = "apple"))]
            Gpu::Cuda(gpu) => normalizes!(Cuda, gpu),
            #[cfg(not(target_vendor = "apple"))]
            Gpu::Vulkan(gpu) => normalizes!(Vulkan, gpu),
            #[cfg(target_vendor = "apple")]
            Gpu::Metal(gpu) => normalizes!(Metal, gpu),
        }
    }

    fn scale_rows(&self, x: &mut Buffer, factors: &Buffer) {
        dispatch!(self, scale_rows([x], [factors]))
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
        dispatch!(
            self,
            block_conv([out], [x], [taps], from, [base], kernel, group)
        )
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
        dispatch!(
            self,
            attention_rel(
                [out],
                [q],
                (k_cache),
                (v_cache),
                [rel],
                [bias],
                [sinks],
                pos,
                n_heads,
                head_dim,
                n_kv_heads,
                window,
                rows
            )
        )
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
        dispatch!(self, route([ids], [weights], [logits], [bias], k, scale))
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
        dispatch!(self, expert_matmul([out], { w }, [scale], [x], [ids], k))
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
        dispatch!(
            self,
            expert_ffn(
                [out],
                { gate_up },
                [gate_up_scale],
                { down },
                [down_scale],
                [x],
                [ids],
                k,
                width,
                beta,
                linear_beta,
                [scratch],
                [act]
            )
        )
    }

    fn situ(&self, out: &mut Buffer, gate_up: &Buffer, width: usize, beta: f32, linear_beta: f32) {
        dispatch!(self, situ([out], [gate_up], width, beta, linear_beta))
    }

    fn weighted_sum(&self, out: &mut Buffer, y: &Buffer, weights: &Buffer) {
        dispatch!(self, weighted_sum([out], [y], [weights]))
    }
}
