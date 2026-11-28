//! Devices a model runs on: the operations a transformer's forward pass is
//! built from, and their implementations. [`Cpu`] is the reference; the
//! `cuda`, `metal`, and [`vulkan`] devices run the same operations on a GPU
//! with hand-written compute kernels, and [`Gpu`] is whichever of them
//! opens.

#[cfg(test)]
#[macro_use]
mod tests;

pub mod cpu;
pub mod float;
pub mod platform;
pub mod quant;
pub mod vulkan;

pub use cpu::Cpu;
pub use platform::Gpu;
pub use vulkan::Vulkan;

// The one place the platform decides which devices are built: Metal on
// Apple's alone, and CUDA elsewhere. [`Gpu`] is whichever of them opens.
cfg_select! {
    target_vendor = "apple" => {
        pub mod metal;

        pub use metal::Metal;
    }
    _ => {
        pub mod cuda;

        pub use cuda::Cuda;
    }
}

/// Taps of the causal convolution in linear attention.
pub const CONV_KERNEL: usize = 4;

/// Activations a block of a key or value cache holds: 8-bit integers
/// sharing an f16 scale, as [`quant::quantize`] makes them.
pub const CACHE_BLOCK: usize = 32;

/// A weight matrix on the host, of shape `[rows, cols]`: raw bf16 bits, or
/// rows of 4-bit or 8-bit blocks with their scales apart, as [`quant`] lays
/// them out. A matrix of NVFP4 is one a device may not take: see
/// [`Device::takes`].
pub enum Tensor {
    Bf16 {
        shape: Vec<usize>,
        data: Vec<u16>,
    },
    Q4_0 {
        shape: Vec<usize>,
        scales: Vec<u16>,
        quants: Vec<u8>,
    },
    Q8_0 {
        shape: Vec<usize>,
        scales: Vec<u16>,
        quants: Vec<i8>,
    },
    Nvfp4 {
        shape: Vec<usize>,
        scales: Vec<u8>,
        quants: Vec<u8>,
    },
}

impl Tensor {
    pub fn shape(&self) -> &[usize] {
        match self {
            Tensor::Bf16 { shape, .. }
            | Tensor::Q4_0 { shape, .. }
            | Tensor::Q8_0 { shape, .. }
            | Tensor::Nvfp4 { shape, .. } => shape,
        }
    }

    /// The matrix made of the given rows of this one, in the given order: a
    /// row may be taken more than once or not at all.
    pub fn select_rows(&self, rows: &[usize]) -> Tensor {
        let cols = self.shape()[1];
        let shape = vec![rows.len(), cols];
        fn take<T: Copy>(data: &[T], per_row: usize, rows: &[usize]) -> Vec<T> {
            let mut out = Vec::with_capacity(rows.len() * per_row);
            for &r in rows {
                out.extend_from_slice(&data[r * per_row..][..per_row]);
            }
            out
        }
        match self {
            Tensor::Bf16 { data, .. } => Tensor::Bf16 {
                shape,
                data: take(data, cols, rows),
            },
            Tensor::Q4_0 { scales, quants, .. } => Tensor::Q4_0 {
                shape,
                scales: take(scales, quant::blocks(cols), rows),
                quants: take(quants, cols / 2, rows),
            },
            Tensor::Q8_0 { scales, quants, .. } => Tensor::Q8_0 {
                shape,
                scales: take(scales, quant::blocks(cols), rows),
                quants: take(quants, cols, rows),
            },
            Tensor::Nvfp4 { scales, quants, .. } => Tensor::Nvfp4 {
                shape,
                scales: take(scales, cols / quant::NVFP4_BLOCK, rows),
                quants: take(quants, cols / 2, rows),
            },
        }
    }

    /// The matrix with its columns taken in groups of `width`, in the given
    /// order of groups: a group may be taken more than once or not at all.
    /// The groups must be whole blocks of the format.
    pub fn select_column_groups(&self, width: usize, groups: &[usize]) -> Tensor {
        let (rows, cols) = (self.shape()[0], self.shape()[1]);
        assert!(cols.is_multiple_of(width));
        let shape = vec![rows, groups.len() * width];
        // Takes groups of `per_group` elements out of each row of
        // `per_row`.
        fn take<T: Copy>(data: &[T], per_row: usize, per_group: usize, groups: &[usize]) -> Vec<T> {
            let mut out = Vec::with_capacity(data.len() / per_row * groups.len() * per_group);
            for row in data.chunks_exact(per_row) {
                for &g in groups {
                    out.extend_from_slice(&row[g * per_group..][..per_group]);
                }
            }
            out
        }
        match self {
            Tensor::Bf16 { data, .. } => Tensor::Bf16 {
                shape,
                data: take(data, cols, width, groups),
            },
            Tensor::Q4_0 { scales, quants, .. } => Tensor::Q4_0 {
                shape,
                scales: take(scales, quant::blocks(cols), quant::blocks(width), groups),
                quants: take(quants, cols / 2, width / 2, groups),
            },
            Tensor::Q8_0 { scales, quants, .. } => Tensor::Q8_0 {
                shape,
                scales: take(scales, quant::blocks(cols), quant::blocks(width), groups),
                quants: take(quants, cols, width, groups),
            },
            Tensor::Nvfp4 { scales, quants, .. } => Tensor::Nvfp4 {
                shape,
                scales: take(
                    scales,
                    cols / quant::NVFP4_BLOCK,
                    width / quant::NVFP4_BLOCK,
                    groups,
                ),
                quants: take(quants, cols / 2, width / 2, groups),
            },
        }
    }

    /// The weights, reconstructed, row by row.
    pub fn dequantize(&self) -> Vec<f32> {
        let (rows, cols) = (self.shape()[0], self.shape()[1]);
        let mut out = vec![0.0; rows * cols];
        for (r, out) in out.chunks_exact_mut(cols).enumerate() {
            match self {
                Tensor::Bf16 { data, .. } => {
                    for (o, &w) in out.iter_mut().zip(&data[r * cols..][..cols]) {
                        *o = float::bf16(w);
                    }
                }
                Tensor::Q4_0 { scales, quants, .. } => {
                    let blocks = quant::blocks(cols);
                    quant::dequantize_q4_0_row(
                        &scales[r * blocks..][..blocks],
                        &quants[r * cols / 2..][..cols / 2],
                        out,
                    );
                }
                Tensor::Q8_0 { scales, quants, .. } => {
                    let blocks = quant::blocks(cols);
                    quant::dequantize_q8_0_row(
                        &scales[r * blocks..][..blocks],
                        &quants[r * cols..][..cols],
                        out,
                    );
                }
                Tensor::Nvfp4 { scales, quants, .. } => {
                    let blocks = cols / quant::NVFP4_BLOCK;
                    quant::dequantize_nvfp4_row(
                        &scales[r * blocks..][..blocks],
                        &quants[r * cols / 2..][..cols / 2],
                        out,
                    );
                }
            }
        }
        out
    }
}

/// The operations a transformer's forward pass is built from.
///
/// A device owns the memory the model computes in: weights are uploaded to it
/// once, activations live in buffers allocated on it, and every operation runs
/// on it. Operations may run lazily; reading a buffer back waits for them.
/// What becomes of a head of a convolution's output before it is kept in
/// a cache.
#[derive(Clone, Copy)]
pub enum Kept {
    /// Scaled, and then down to a length of `threshold` if it is longer,
    /// a head being `dim` long.
    Clipped { dim: usize, threshold: f32 },
    /// Divided by its root mean square, and scaled.
    Normed { eps: f32 },
}

pub trait Device {
    /// A vector of f32 activations in device memory. Operations that work
    /// on rows of activations, one per token, take a buffer's length as a
    /// whole number of rows.
    type Buffer;
    /// A weight matrix in device memory, in any of [`Tensor`]'s formats.
    type Weight;
    /// A key or value cache in device memory: activations stored in blocks
    /// of [`CACHE_BLOCK`] as [`quantize`] makes them, 8.5 bits each, for
    /// about half the memory of half-precision floats.
    type Cache;

    /// What the device is called: the GPU's name, or `cpu`.
    fn name(&self) -> String;

    /// Bytes of memory free on the device for a model, if it can tell:
    /// what it can allocate without taking memory others are using. On a
    /// GPU that shares the host's memory, that is the host's memory free.
    fn memory(&self) -> Option<u64>;

    /// Most tokens a forward pass should run through the device at once.
    /// Running a batch of tokens together reads each weight once for the
    /// whole batch, rather than once per token, and the activations are
    /// allocated for this many. A batch is one submission to the device,
    /// which a driver may abandon as hung if it runs for seconds, so
    /// batches are kept short.
    fn batch(&self) -> usize {
        64
    }

    /// Whether the device takes weights in NVFP4, and runs
    /// [`expert_matmul`](Self::expert_matmul) over them. The other
    /// formats every device takes.
    fn takes_nvfp4(&self) -> bool {
        false
    }

    /// Copies a weight matrix into device memory.
    fn upload(&self, tensor: Tensor) -> Self::Weight;

    /// Allocates a zeroed buffer of `len` activations.
    fn alloc(&self, len: usize) -> Self::Buffer;

    /// Allocates a zeroed cache of `len` activations, a whole number of
    /// blocks.
    fn alloc_cache(&self, len: usize) -> Self::Cache;

    /// Sets a buffer's length, up to the length it was allocated with, so
    /// that it holds a batch of some other number of tokens. The
    /// activations it gains are undefined.
    fn resize(&self, buf: &mut Self::Buffer, len: usize);

    /// Copies a buffer out of device memory.
    fn read(&self, buf: &Self::Buffer) -> Vec<f32>;

    /// A buffer copied out of device memory with each activation capped
    /// softly: `cap * tanh(x / cap)`, what a model's logits go through
    /// where it has a soft cap on them. The buffer is left undefined.
    fn read_capped(&self, buf: &mut Self::Buffer, cap: f32) -> Vec<f32> {
        let mut host = self.read(buf);
        for x in &mut host {
            *x = cap * (*x / cap).tanh();
        }
        host
    }

    /// Copies `data` into a buffer of the same length.
    fn write(&self, buf: &mut Self::Buffer, data: &[f32]);

    /// `dst[dst_offset..][..len] = src[src_offset..][..len]`
    fn copy(
        &self,
        dst: &mut Self::Buffer,
        dst_offset: usize,
        src: &Self::Buffer,
        src_offset: usize,
        len: usize,
    );

    /// `cache[offset..][..src.len()] = src`: stores activations in a cache,
    /// quantized a block at a time as [`quantize`] does. The offset and the
    /// length are whole blocks.
    fn store(&self, cache: &mut Self::Cache, offset: usize, src: &Self::Buffer);

    /// The first `len` activations of a cache, a whole number of blocks, as
    /// the bytes they are stored in: their 8-bit integers, then their
    /// blocks' scales as little-endian f16 bits, [`cache_bytes`] in all.
    fn read_cache(&self, cache: &Self::Cache, len: usize) -> Vec<u8>;

    /// Sets the first activations of a cache to `data`, bytes as
    /// [`read_cache`](Self::read_cache) gives them.
    fn write_cache(&self, cache: &mut Self::Cache, data: &[u8]);

    /// `dst[..len] = src[..len]`, of two caches, a whole number of blocks.
    fn copy_cache(&self, dst: &mut Self::Cache, src: &Self::Cache, len: usize) {
        self.write_cache(dst, &self.read_cache(src, len));
    }

    /// `out[t] = w · x[t]` for each row `x[t]` of `x`, for a weight matrix
    /// `w` of shape `[rows, cols]`: `x` holds `cols` activations per token,
    /// and `out` `rows` per token.
    fn matmul(&self, out: &mut Self::Buffer, w: &Self::Weight, x: &Self::Buffer);

    /// Several matrix multiplications, each an output, a weight and an
    /// input as [`Self::matmul`] takes them: for a device that runs them
    /// faster together than one after another.
    fn matmuls(&self, jobs: &mut [(&mut Self::Buffer, &Self::Weight, &Self::Buffer)]) {
        for (out, w, x) in jobs {
            self.matmul(out, w, x);
        }
    }

    /// `x += w · input`, a row of `x` for each of `input`: a matrix
    /// multiplication whose result goes to the residual stream. `scratch`
    /// is as long as `x`, and is left undefined.
    fn matmul_add(
        &self,
        x: &mut Self::Buffer,
        w: &Self::Weight,
        input: &Self::Buffer,
        scratch: &mut Self::Buffer,
    ) {
        self.matmul(scratch, w, input);
        self.add(x, scratch);
    }

    /// `out = silu(gate · x) * (up · x)` for each row of `x`: the SwiGLU
    /// feed-forward network up to its last matrix. `scratch` is as long as
    /// `out`, and is left undefined. A device may keep the result in half
    /// precision, for the matrix multiplication it is for.
    fn swiglu(
        &self,
        out: &mut Self::Buffer,
        gate: &Self::Weight,
        up: &Self::Weight,
        x: &Self::Buffer,
        scratch: &mut Self::Buffer,
    ) {
        self.matmul(out, gate, x);
        self.matmul(scratch, up, x);
        self.silu_mul(out, scratch);
    }

    /// `x += y`
    fn add(&self, x: &mut Self::Buffer, y: &Self::Buffer);

    /// `out = rmsnorm(x) * weight` for the first `len` activations of `x`,
    /// which is how long `out` is: [`rmsnorm`](Self::rmsnorm) into another
    /// buffer, which is what a matrix multiplication's input goes through.
    /// A device may keep the result in half precision, for the matrix
    /// multiplications it is for.
    fn normalize(
        &self,
        out: &mut Self::Buffer,
        x: &Self::Buffer,
        weight: &Self::Buffer,
        len: usize,
        eps: f32,
    ) {
        self.copy(out, 0, x, 0, len);
        self.rmsnorm(out, weight, eps);
    }

    /// Several normalizations as [`Self::normalize`] is one, `outs[j]` of
    /// `xs[j]` by `weights[j]`: the inputs of a layer's queries, keys and
    /// values.
    fn normalizes(
        &self,
        outs: &mut [&mut Self::Buffer],
        xs: &[&Self::Buffer],
        weights: &[&Self::Buffer],
        len: usize,
        eps: f32,
    ) {
        for (j, out) in outs.iter_mut().enumerate() {
            self.normalize(out, xs[j], weights[j], len, eps);
        }
    }

    /// Normalizes each `weight.len()`-long row of `x` by its root mean square,
    /// then scales it by `weight`.
    fn rmsnorm(&self, x: &mut Self::Buffer, weight: &Self::Buffer, eps: f32);

    /// Normalizes each `dim`-long row of `x` to unit length.
    fn l2norm(&self, x: &mut Self::Buffer, dim: usize, eps: f32);

    /// Rotates the first `rot_dim` elements of each `head_dim`-long head of
    /// `x`, which holds `n_heads` heads per token, to encode the token's
    /// position (rotary position embeddings): `pos` for the first token,
    /// `pos + 1` for the next, and so on. Element `i` of a head at position
    /// `p` pairs with element `i + rot_dim / 2`, rotated by the angle whose
    /// cosine and sine are at `table[(p * rot_dim / 2 + i) * 2..][..2]`.
    #[allow(clippy::too_many_arguments)]
    fn rope(
        &self,
        x: &mut Self::Buffer,
        table: &Self::Buffer,
        pos: usize,
        n_heads: usize,
        head_dim: usize,
        rot_dim: usize,
    );

    /// Causal self-attention for the tokens in `q`, the first at position
    /// `pos`: each of a token's `n_heads` heads, each `head_dim` long,
    /// attends over the key and value caches up to and including the token's
    /// own position, and writes its result to `out`. The caches hold
    /// `n_kv_heads` heads per position, and must already hold the tokens'
    /// own keys and values.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        out: &mut Self::Buffer,
        q: &Self::Buffer,
        k_cache: &Self::Cache,
        v_cache: &Self::Cache,
        pos: usize,
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
    );

    /// `gate = silu(gate) * up`: the SwiGLU activation.
    fn silu_mul(&self, gate: &mut Self::Buffer, up: &Self::Buffer);

    /// `x *= sigmoid(gate)`
    fn sigmoid_mul(&self, x: &mut Self::Buffer, gate: &Self::Buffer);

    /// The causal convolution of linear attention: each channel of `x`,
    /// which holds a row of channels per token, is convolved over the last
    /// [`CONV_KERNEL`] tokens with its taps from `weight`, which holds that
    /// many per channel, and passed through SiLU. `state` holds the rows of
    /// the `CONV_KERNEL - 1` tokens before the batch, and `state_out` gets
    /// the rows of the last `CONV_KERNEL - 1` tokens of it. The channels of
    /// the result are split by token into `q`, `k`, and `v`, in that order,
    /// as wide as each is per token.
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &self,
        q: &mut Self::Buffer,
        k: &mut Self::Buffer,
        v: &mut Self::Buffer,
        state_out: &mut Self::Buffer,
        x: &Self::Buffer,
        state: &Self::Buffer,
        weight: &Self::Buffer,
    );

    /// The gated delta rule of linear attention, one token after another.
    /// Each of the `n_v_heads` heads keeps a `head_dim` by `head_dim` state
    /// in `state`, which decays by `exp(a * softplus(alpha + dt_bias))` per
    /// token, forgets what it holds for the token's key in proportion to
    /// `sigmoid(beta)`, and stores the token's value in its place; the
    /// head's output is what the state holds for the token's query, scaled
    /// by the inverse square root of `head_dim`. `q` and `k` hold
    /// `n_k_heads` heads per token, each shared by the value heads in turn;
    /// `gates` holds `alpha` for every value head then `beta` for every one
    /// per token; and `decay` holds `a` for every value head then `dt_bias`
    /// for every one.
    #[allow(clippy::too_many_arguments)]
    fn delta_net(
        &self,
        out: &mut Self::Buffer,
        q: &Self::Buffer,
        k: &Self::Buffer,
        v: &Self::Buffer,
        gates: &Self::Buffer,
        decay: &Self::Buffer,
        state: &mut Self::Buffer,
        n_k_heads: usize,
        n_v_heads: usize,
        head_dim: usize,
    );

    /// Attention over the depth of the model: for each token, `out` is a
    /// mix of the token's rows of `states`, the residual stream as it was
    /// at several depths, each normalized by its root mean square. A
    /// state's share is the softmax, over the states, of its normalized
    /// row's dot product with `query`, which is as wide as a row, over the
    /// square root of the width.
    ///
    /// The operations from here on are a model's own, and a device that
    /// has no kernel for one runs it on the host, with the CPU's.
    fn mix(
        &self,
        out: &mut Self::Buffer,
        states: &[&Self::Buffer],
        query: &Self::Buffer,
        eps: f32,
    ) {
        let states: Vec<Vec<f32>> = states.iter().map(|s| self.read(s)).collect();
        let mut host = self.read(out);
        Cpu.mix(
            &mut host,
            &states.iter().collect::<Vec<_>>(),
            &self.read(query),
            eps,
        );
        self.write(out, &host);
    }

    /// `x = x - tanh(forget) * x + tanh(gain) * message`: what a gate
    /// lets into the residual stream of a message, and what it has the
    /// stream forget.
    fn gated_update(
        &self,
        x: &mut Self::Buffer,
        forget: &Self::Buffer,
        gain: &Self::Buffer,
        message: &Self::Buffer,
    ) {
        let mut host = self.read(x);
        Cpu.gated_update(
            &mut host,
            &self.read(forget),
            &self.read(gain),
            &self.read(message),
        );
        self.write(x, &host);
    }

    /// A gate on a message, for each row of `x` and of `message`: both
    /// rows, each normalized by its root mean square, go through
    /// `down_input` and `down_message` to a few dimensions, the two
    /// results are added, and from the sum `gain` and `forget` give what
    /// [`gated_update`](Self::gated_update) takes, which changes `x`.
    #[allow(clippy::too_many_arguments)]
    fn gate(
        &self,
        x: &mut Self::Buffer,
        message: &Self::Buffer,
        down_input: &Self::Weight,
        down_message: &Self::Weight,
        gain: &Self::Weight,
        forget: &Self::Weight,
        dim: usize,
        rank: usize,
        eps: f32,
    ) {
        let rows = self.read(x).len() / dim;
        let ones = {
            let mut ones = self.alloc(dim);
            self.write(&mut ones, &vec![1.0; dim]);
            ones
        };
        let (mut input, mut normed) = (self.alloc(rows * dim), self.alloc(rows * dim));
        self.normalize(&mut input, x, &ones, rows * dim, eps);
        self.normalize(&mut normed, message, &ones, rows * dim, eps);
        let (mut down, mut from_message) = (self.alloc(rows * rank), self.alloc(rows * rank));
        self.matmul(&mut down, down_input, &input);
        self.matmul(&mut from_message, down_message, &normed);
        self.add(&mut down, &from_message);
        self.matmul(&mut input, gain, &down);
        self.matmul(&mut normed, forget, &down);
        self.gated_update(x, &normed, &input, message);
    }

    /// Several gates as [`Self::gate`] is one: `xs[j]` gets `messages[j]`
    /// through the weights `weights[j]`, the projections down of the
    /// input and of the message and then the gain and what is forgotten,
    /// as `gate` takes them.
    fn gates(
        &self,
        xs: &mut [&mut Self::Buffer],
        messages: &[&Self::Buffer],
        weights: &[[&Self::Weight; 4]],
        dim: usize,
        rank: usize,
        eps: f32,
    ) {
        for (j, x) in xs.iter_mut().enumerate() {
            let [down_input, down_message, gain, forget] = weights[j];
            self.gate(
                x,
                messages[j],
                down_input,
                down_message,
                gain,
                forget,
                dim,
                rank,
                eps,
            );
        }
    }

    /// A short causal convolution with a residual: each channel of `x`,
    /// which holds a row of channels per token, is convolved over the
    /// token and the `taps - 1` before it with its taps from `weight`,
    /// which holds `taps` per channel, the last for the token itself. With
    /// `dynamic`, a row per token of `taps` more for each of its heads,
    /// which the channels are split evenly between, the token `lag` back
    /// is also taken times the head's tap `lag`. The sum is added to the
    /// channel, and the whole divided by the square root of the taps and
    /// one. `state` holds the rows of the `taps - 1` tokens before the
    /// batch, and is left holding the last `taps - 1` of it.
    fn short_conv(
        &self,
        out: &mut Self::Buffer,
        x: &Self::Buffer,
        state: &mut Self::Buffer,
        weight: &Self::Buffer,
        dynamic: Option<(&Self::Buffer, usize)>,
    ) {
        let mut host = self.read(out);
        let mut host_state = self.read(state);
        let taps = dynamic.map(|(taps, heads)| (self.read(taps), heads));
        Cpu.short_conv(
            &mut host,
            &self.read(x),
            &mut host_state,
            &self.read(weight),
            taps.as_ref().map(|(taps, heads)| (taps, *heads)),
        );
        self.write(out, &host);
        self.write(state, &host_state);
    }

    /// Several mixes of the same states as [`Self::mix`] gives, `outs[j]`
    /// with `queries[j]`; and, with `norms`, each then normalized in place
    /// as [`Self::rmsnorm`] does, by `norms[j]` and with the epsilon
    /// beside them: a matrix multiplication's input, which a device may
    /// keep in half precision.
    fn mixes(
        &self,
        outs: &mut [&mut Self::Buffer],
        states: &[&Self::Buffer],
        queries: &[&Self::Buffer],
        eps: f32,
        norms: Option<(&[&Self::Buffer], f32)>,
    ) {
        for (j, out) in outs.iter_mut().enumerate() {
            self.mix(out, states, queries[j], eps);
            if let Some((norms, eps)) = norms {
                self.rmsnorm(out, norms[j], eps);
            }
        }
    }

    /// The feed-forward networks of each token's experts: `out` gets what
    /// [`Self::expert_matmul`] gives through `down` of the
    /// [`Self::situ`] of what it gives through `gate_up` of the token's
    /// row of `x`, a row per expert of a token; the activation is `width`
    /// wide. `scratch` is as long as the first matmul's output and `act`
    /// as the activation, and both are left undefined.
    #[allow(clippy::too_many_arguments)]
    fn expert_ffn(
        &self,
        out: &mut Self::Buffer,
        gate_up: &Self::Weight,
        gate_up_scale: &Self::Buffer,
        down: &Self::Weight,
        down_scale: &Self::Buffer,
        x: &Self::Buffer,
        ids: &Self::Buffer,
        k: usize,
        width: usize,
        beta: f32,
        linear_beta: f32,
        scratch: &mut Self::Buffer,
        act: &mut Self::Buffer,
    ) {
        expert_ffn(
            self,
            out,
            gate_up,
            gate_up_scale,
            down,
            down_scale,
            x,
            ids,
            k,
            width,
            beta,
            linear_beta,
            scratch,
            act,
        );
    }

    /// `x += ` the convolution of `input` that [`Self::short_conv`] gives
    /// without taps of the heads' own: a convolution whose output joins
    /// the residual stream. `scratch` is as long as `x`, and is left
    /// undefined.
    fn short_conv_add(
        &self,
        x: &mut Self::Buffer,
        input: &Self::Buffer,
        state: &mut Self::Buffer,
        weight: &Self::Buffer,
        scratch: &mut Self::Buffer,
    ) {
        self.short_conv(scratch, input, state, weight, None);
        self.add(x, scratch);
    }

    /// The convolution of `x` that [`Self::short_conv`] gives, each head
    /// of it clipped or normalized as `kept` says, with `scale` what
    /// [`Self::clip`] scales by or the weight of [`Self::rmsnorm`], and
    /// stored in `cache` as [`Self::store`] does: the row of the token at
    /// position `p` in row `p % rows`, the first of the `n` at `pos`, each
    /// `channels` wide. What a token's keys and values go through on
    /// their way to the caches. `scratch` is as long as `x` and `wrapped`
    /// may be, and both are left undefined.
    #[allow(clippy::too_many_arguments)]
    fn short_conv_store(
        &self,
        cache: &mut Self::Cache,
        pos: usize,
        rows: usize,
        n: usize,
        channels: usize,
        x: &Self::Buffer,
        state: &mut Self::Buffer,
        weight: &Self::Buffer,
        dynamic: Option<(&Self::Buffer, usize)>,
        scale: &Self::Buffer,
        kept: Kept,
        scratch: &mut Self::Buffer,
        wrapped: &mut Self::Buffer,
    ) {
        short_conv_store(
            self, cache, pos, rows, n, channels, x, state, weight, dynamic, scale, kept, scratch,
            wrapped,
        );
    }

    /// Scales each `scale.len()`-long row of `x` by `scale`, then each
    /// `dim`-long head of it down to a length of `threshold`, if it is
    /// longer.
    fn clip(&self, x: &mut Self::Buffer, scale: &Self::Buffer, dim: usize, threshold: f32) {
        let mut host = self.read(x);
        Cpu.clip(&mut host, &self.read(scale), dim, threshold);
        self.write(x, &host);
    }

    /// Several clips as [`Self::clip`] is one: each the activations, what
    /// scales them, a head's width and the threshold.
    fn clips(&self, jobs: &mut [(&mut Self::Buffer, &Self::Buffer, usize, f32)]) {
        for (x, scale, dim, threshold) in jobs {
            self.clip(x, scale, *dim, *threshold);
        }
    }

    /// Scales each of the `factors.len()` rows of `x` by its factor.
    fn scale_rows(&self, x: &mut Self::Buffer, factors: &Self::Buffer) {
        let mut host = self.read(x);
        Cpu.scale_rows(&mut host, &self.read(factors));
        self.write(x, &host);
    }

    /// A convolution over the rows of `x`, a block of them, with taps of
    /// each row's own: a channel of a row is the sum, over the row and the
    /// `kernel - 1` before it, of the channel that far back times a tap,
    /// which is the one `base` has for the channel and the distance, a row
    /// of channels per distance, plus the one the row has in `taps` for
    /// the distance and the channel's group, of `group` channels. A row of
    /// `taps` holds its own from `from` on, a group's for each distance, a
    /// distance after another. The rows before the block count as zeros.
    #[allow(clippy::too_many_arguments)]
    fn block_conv(
        &self,
        out: &mut Self::Buffer,
        x: &Self::Buffer,
        taps: &Self::Buffer,
        from: usize,
        base: &Self::Buffer,
        kernel: usize,
        group: usize,
    ) {
        let mut host = self.read(out);
        Cpu.block_conv(
            &mut host,
            &self.read(x),
            &self.read(taps),
            from,
            &self.read(base),
            kernel,
            group,
        );
        self.write(out, &host);
    }

    /// Causal self-attention as [`attention`](Self::attention) is, over
    /// the keys no more than `window - 1` positions back, with a bias by
    /// distance and a sink. The caches hold `rows` positions, position `p`
    /// in row `p % rows`: as many as a sequence is long, or a ring of the
    /// latest, no fewer than the window and the batch reach back. A key `d` positions back has added to its
    /// score the dot product of the head's row of `rel`, which holds a row
    /// per head per token, with row `d` of `bias`, a row as wide per
    /// distance; a key further back than `bias` has rows gets none. The
    /// softmax is taken over the keys and a sink that has no value, whose
    /// score is the head's of `sinks`, which holds one per head per token:
    /// what the sink takes, no key gets.
    #[allow(clippy::too_many_arguments)]
    fn attention_rel(
        &self,
        out: &mut Self::Buffer,
        q: &Self::Buffer,
        k_cache: &Self::Cache,
        v_cache: &Self::Cache,
        rel: &Self::Buffer,
        bias: &Self::Buffer,
        sinks: &Self::Buffer,
        pos: usize,
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
        window: usize,
        rows: usize,
    ) {
        let q = self.read(q);
        let n = q.len() / (n_heads * head_dim);
        let kv_dim = n_kv_heads * head_dim;
        let len = (pos + n).min(rows) * kv_dim;
        let cache = |device: &Self::Cache| {
            let mut host = Cpu.alloc_cache(len);
            Cpu.write_cache(&mut host, &self.read_cache(device, len));
            host
        };
        let mut host = self.read(out);
        Cpu.attention_rel(
            &mut host,
            &q,
            &cache(k_cache),
            &cache(v_cache),
            &self.read(rel),
            &self.read(bias),
            &self.read(sinks),
            pos,
            n_heads,
            head_dim,
            n_kv_heads,
            window,
            rows,
        );
        self.write(out, &host);
    }

    /// Picks each token's experts: the `k` of them whose score is highest,
    /// the score of each being the sigmoid of its logit, of which `logits`
    /// holds a row per token, plus its `bias`. `ids` gets the experts'
    /// numbers, a row of `k` per token, the highest score first, and
    /// `weights` what each one's output counts for: its sigmoid over the
    /// sum of the `k`, times `scale`.
    fn route(
        &self,
        ids: &mut Self::Buffer,
        weights: &mut Self::Buffer,
        logits: &Self::Buffer,
        bias: &Self::Buffer,
        k: usize,
        scale: f32,
    ) {
        let (mut host_ids, mut host_weights) = (self.read(ids), self.read(weights));
        Cpu.route(
            &mut host_ids,
            &mut host_weights,
            &self.read(logits),
            &self.read(bias),
            k,
            scale,
        );
        self.write(ids, &host_ids);
        self.write(weights, &host_weights);
    }

    /// A matrix multiplication by each token's experts: `w` holds the
    /// matrices of all the experts, one after another, as many as `scale`
    /// has scales, and `ids` a row of `k` experts per token, as
    /// [`route`](Self::route) picks them. `out` gets a row per expert of a
    /// token: the expert's matrix times the token's row of `x`, or times
    /// the expert's own row of it where `x` holds one per expert, scaled
    /// by the expert's scale.
    fn expert_matmul(
        &self,
        out: &mut Self::Buffer,
        w: &Self::Weight,
        scale: &Self::Buffer,
        x: &Self::Buffer,
        ids: &Self::Buffer,
        k: usize,
    ) {
        let _ = (out, w, scale, x, ids, k);
        unimplemented!("{} does not run mixtures of experts", self.name());
    }

    /// The SiTU activation of a feed-forward network: each row of
    /// `gate_up` is a gate then an up projection, each `width` wide, as a
    /// row of `out` is, which gets
    /// `beta * tanh(gate / beta) * sigmoid(gate) * linear_beta * tanh(up / linear_beta)`.
    fn situ(
        &self,
        out: &mut Self::Buffer,
        gate_up: &Self::Buffer,
        width: usize,
        beta: f32,
        linear_beta: f32,
    ) {
        let mut host = self.read(out);
        Cpu.situ(&mut host, &self.read(gate_up), width, beta, linear_beta);
        self.write(out, &host);
    }

    /// `out[t] = sum over e of weights[t][e] * y[t][e]`: the outputs of a
    /// token's experts, of which `y` holds a row each, summed by what each
    /// counts for.
    fn weighted_sum(&self, out: &mut Self::Buffer, y: &Self::Buffer, weights: &Self::Buffer) {
        let mut host = self.read(out);
        Cpu.weighted_sum(&mut host, &self.read(y), &self.read(weights));
        self.write(out, &host);
    }
}

/// The bytes a cache holds for `len` activations, a whole number of blocks.
pub fn cache_bytes(len: usize) -> usize {
    assert!(len.is_multiple_of(CACHE_BLOCK));
    len + len / CACHE_BLOCK * 2
}

/// [`Device::short_conv_store`] an operation at a time: the convolution,
/// what `kept` does to it, and the rows stored, in two parts where the
/// batch runs past the end of the cache's rows, the second from its start.
#[allow(clippy::too_many_arguments)]
pub fn short_conv_store<D: Device + ?Sized>(
    d: &D,
    cache: &mut D::Cache,
    pos: usize,
    rows: usize,
    n: usize,
    channels: usize,
    x: &D::Buffer,
    state: &mut D::Buffer,
    weight: &D::Buffer,
    dynamic: Option<(&D::Buffer, usize)>,
    scale: &D::Buffer,
    kept: Kept,
    scratch: &mut D::Buffer,
    wrapped: &mut D::Buffer,
) {
    d.short_conv(scratch, x, state, weight, dynamic);
    match kept {
        Kept::Clipped { dim, threshold } => d.clip(scratch, scale, dim, threshold),
        Kept::Normed { eps } => d.rmsnorm(scratch, scale, eps),
    }
    let at = pos % rows;
    let first = n.min(rows - at);
    if first < n {
        let rest = (n - first) * channels;
        d.resize(wrapped, rest);
        d.copy(wrapped, 0, scratch, first * channels, rest);
        d.store(cache, 0, wrapped);
        d.resize(scratch, first * channels);
    }
    d.store(cache, at * channels, scratch);
    d.resize(scratch, n * channels);
}

/// [`Device::expert_ffn`] an operation at a time.
#[allow(clippy::too_many_arguments)]
pub fn expert_ffn<D: Device + ?Sized>(
    d: &D,
    out: &mut D::Buffer,
    gate_up: &D::Weight,
    gate_up_scale: &D::Buffer,
    down: &D::Weight,
    down_scale: &D::Buffer,
    x: &D::Buffer,
    ids: &D::Buffer,
    k: usize,
    width: usize,
    beta: f32,
    linear_beta: f32,
    scratch: &mut D::Buffer,
    act: &mut D::Buffer,
) {
    d.expert_matmul(scratch, gate_up, gate_up_scale, x, ids, k);
    d.situ(act, scratch, width, beta, linear_beta);
    d.expert_matmul(out, down, down_scale, act, ids, k);
}
