//! Checks of a GPU device's operations against the CPU reference.

use crate::gpu::{CACHE_BLOCK, CONV_KERNEL, Cpu, Device, Tensor, cache_bytes, quant};

/// Defines a test for each check, run on the device `$open` returns, and
/// skipped if it returns an error.
macro_rules! check_against_cpu {
    ($open:expr) => {
        check_against_cpu!(
            $open;
            write_and_read_round_trip,
            alloc_is_zeroed_and_copy_moves_ranges,
            matmul_matches_cpu,
            q4_0_matmul_matches_cpu,
            q8_0_matmul_matches_cpu,
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
            expert_matmul_matches_cpu,
            mix_matches_cpu,
            short_conv_matches_cpu,
            attention_rel_matches_cpu,
            route_matches_cpu,
            rows_and_gates_match_cpu,
            gate_matches_cpu,
            copy_cache_copies_a_prefix,
            resize_keeps_capacity
        );
    };
    ($open:expr; $($check:ident),*) => {
        $(
            #[test]
            fn $check() {
                match $open {
                    Ok(gpu) => crate::gpu::tests::$check(&gpu),
                    Err(e) => eprintln!("skipping: {e}"),
                }
            }
        )*
    };
}

/// Deterministic pseudo-random values in [-1, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn floats(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }

    fn bf16(&mut self, shape: &[usize]) -> Tensor {
        let n = shape.iter().product();
        Tensor::Bf16 {
            shape: shape.to_vec(),
            data: (0..n)
                .map(|_| (self.next().to_bits() >> 16) as u16)
                .collect(),
        }
    }

    fn q4_0(&mut self, shape: &[usize]) -> Tensor {
        let (rows, cols) = (shape[0], shape[1]);
        let blocks = quant::blocks(cols);
        let (mut scales, mut quants) = (vec![0; rows * blocks], vec![0; rows * cols / 2]);
        for (scales, quants) in scales
            .chunks_exact_mut(blocks)
            .zip(quants.chunks_exact_mut(cols / 2))
        {
            quant::quantize_q4_0_row(&self.floats(cols), scales, quants);
        }
        Tensor::Q4_0 {
            shape: shape.to_vec(),
            scales,
            quants,
        }
    }

    fn q8_0(&mut self, shape: &[usize]) -> Tensor {
        let (rows, cols) = (shape[0], shape[1]);
        let blocks = quant::blocks(cols);
        let (mut scales, mut quants) = (vec![0; rows * blocks], vec![0; rows * cols]);
        for (scales, quants) in scales
            .chunks_exact_mut(blocks)
            .zip(quants.chunks_exact_mut(cols))
        {
            quant::quantize_q8_0_row(&self.floats(cols), scales, quants);
        }
        Tensor::Q8_0 {
            shape: shape.to_vec(),
            scales,
            quants,
        }
    }
}

impl Rng {
    /// A matrix of NVFP4: every value there is, under scales from an
    /// eighth to eight.
    fn nvfp4(&mut self, shape: &[usize]) -> Tensor {
        let n = shape[0] * shape[1];
        let mut byte = |below: f32| ((self.next() + 1.0) * 0.5 * below) as u8;
        Tensor::Nvfp4 {
            shape: shape.to_vec(),
            scales: (0..n / quant::NVFP4_BLOCK)
                .map(|_| (4 << 3) + byte(48.0))
                .collect(),
            quants: (0..n / 2).map(|_| byte(256.0)).collect(),
        }
    }
}

fn clone(t: &Tensor) -> Tensor {
    match t {
        Tensor::Bf16 { shape, data } => Tensor::Bf16 {
            shape: shape.clone(),
            data: data.clone(),
        },
        Tensor::Q4_0 {
            shape,
            scales,
            quants,
        } => Tensor::Q4_0 {
            shape: shape.clone(),
            scales: scales.clone(),
            quants: quants.clone(),
        },
        Tensor::Q8_0 {
            shape,
            scales,
            quants,
        } => Tensor::Q8_0 {
            shape: shape.clone(),
            scales: scales.clone(),
            quants: quants.clone(),
        },
        Tensor::Nvfp4 {
            shape,
            scales,
            quants,
        } => Tensor::Nvfp4 {
            shape: shape.clone(),
            scales: scales.clone(),
            quants: quants.clone(),
        },
    }
}

fn close(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!(
            (a - b).abs() <= tolerance * (1.0 + a.abs().max(b.abs())),
            "element {i}: {a} vs {b}"
        );
    }
}

fn buffer<D: Device>(gpu: &D, data: &[f32]) -> D::Buffer {
    let mut buf = gpu.alloc(data.len());
    gpu.write(&mut buf, data);
    buf
}

/// A cache holding `data`, stored in two parts split at `split`, as the
/// prompt and then the tokens after it are.
fn cache<D: Device>(gpu: &D, data: &[f32], split: usize) -> D::Cache {
    let mut cache = gpu.alloc_cache(data.len());
    for (offset, part) in [(0, &data[..split]), (split, &data[split..])] {
        if !part.is_empty() {
            gpu.store(&mut cache, offset, &buffer(gpu, part));
        }
    }
    cache
}

pub fn write_and_read_round_trip<D: Device>(gpu: &D) {
    let data = Rng(1).floats(100_003);
    let buf = buffer(gpu, &data);
    assert_eq!(gpu.read(&buf), data);
}

pub fn alloc_is_zeroed_and_copy_moves_ranges<D: Device>(gpu: &D) {
    let zero = gpu.alloc(1000);
    assert!(gpu.read(&zero).iter().all(|&v| v == 0.0));
    let data = Rng(2).floats(50);
    let src = buffer(gpu, &data);
    let mut dst = gpu.alloc(100);
    gpu.copy(&mut dst, 30, &src, 10, 20);
    let mut want = vec![0.0; 100];
    Cpu.copy(&mut want, 30, &data, 10, 20);
    assert_eq!(gpu.read(&dst), want);
}

pub fn matmul_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(4);
    for (rows, cols, n) in [(1, 8, 1), (200, 192, 1), (77, 1032, 3), (70_000, 8, 2)] {
        let w = rng.bf16(&[rows, cols]);
        let x = rng.floats(n * cols);
        let mut want = vec![0.0; n * rows];
        Cpu.matmul(&mut want, &clone(&w), &x);
        let weight = gpu.upload(w);
        let x = buffer(gpu, &x);
        let mut out = gpu.alloc(n * rows);
        gpu.matmul(&mut out, &weight, &x);
        close(&gpu.read(&out), &want, 1e-4);
    }
}

/// Checks a batch's results against the CPU's as a device that multiplies a
/// batch in half precision, as the tile kernels do, can meet them: to an
/// error of a thousandth of the outputs' typical magnitude in root mean
/// square and a hundredth at worst, which is a tenth of what eight-bit
/// activations would cost.
fn close_batch(got: &[f32], want: &[f32], what: &str) {
    let typical = (want.iter().map(|v| (v * v) as f64).sum::<f64>() / want.len() as f64).sqrt();
    let (mut worst, mut sum) = (0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want) {
        let e = (g - w).abs() as f64;
        worst = worst.max(e);
        sum += e * e;
    }
    let rms = (sum / got.len() as f64).sqrt();
    assert!(
        rms <= 1e-3 * typical && worst <= 1e-2 * typical,
        "{what}: rms {rms:.2e}, worst {worst:.2e}, typical {typical:.2e}"
    );
}

/// Single tokens and batches, at the edges of the groups of eight tokens
/// the row kernels take, at the model's shapes, and at the edges of the
/// tiles of the tile kernels, which a batch of more than a few tokens goes
/// through where the matrix is a whole number of their steps of 128
/// columns, through `quantized` matrices: a single token is held to the
/// usual error, unless the device multiplies it `in_halves` too, and a
/// batch to a batch's.
fn block_matmul_matches_cpu<D: Device>(
    gpu: &D,
    quantized: &mut dyn FnMut(&[usize]) -> Tensor,
    in_halves: bool,
) {
    for (rows, cols, n) in [
        (1, 32, 1),
        (200, 5120, 1),
        (70_000, 32, 2),
        (77, 1056, 3),
        (200, 5120, 8),
        (64, 160, 9),
        (64, 128, 16),
        (100, 256, 47),
        (1030, 1152, 33),
        (300, 1152, 130),
        (2500, 5120, 64),
    ] {
        let w = quantized(&[rows, cols]);
        let x = Rng(rows as u64 + n as u64).floats(n * cols);
        let mut want = vec![0.0; n * rows];
        Cpu.matmul(&mut want, &clone(&w), &x);
        let weight = gpu.upload(w);
        let mut out = gpu.alloc(n * rows);
        gpu.matmul(&mut out, &weight, &buffer(gpu, &x));
        let got = gpu.read(&out);
        if n > 1 || in_halves {
            close_batch(&got, &want, &format!("{rows}x{cols} n={n}"));
        } else {
            close(&got, &want, 1e-4);
        }
    }
}

pub fn q4_0_matmul_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(40);
    block_matmul_matches_cpu(gpu, &mut |shape| rng.q4_0(shape), false);
}

pub fn q8_0_matmul_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(80);
    block_matmul_matches_cpu(gpu, &mut |shape| rng.q8_0(shape), false);
}

/// The Q4_0 matmul of a device that multiplies single tokens in half
/// precision as it does batches. Only CUDA does, which is not built on
/// Apple's platforms.
#[cfg(not(target_vendor = "apple"))]
pub fn q4_0_matmul_in_halves_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(40);
    block_matmul_matches_cpu(gpu, &mut |shape| rng.q4_0(shape), true);
}

/// The Q8_0 matmul likewise.
#[cfg(not(target_vendor = "apple"))]
pub fn q8_0_matmul_in_halves_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(80);
    block_matmul_matches_cpu(gpu, &mut |shape| rng.q8_0(shape), true);
}

/// Times the Q4_0 matmul likewise.
pub fn q4_0_matmul_speed<D: Device>(gpu: &D) {
    let mut rng = Rng(41);
    matmul_speed(gpu, &mut |shape| rng.q4_0(shape), |cols| {
        cols / 2 + 2 * quant::blocks(cols)
    });
}

/// Times the Q8_0 matmul likewise.
pub fn q8_0_matmul_speed<D: Device>(gpu: &D) {
    let mut rng = Rng(81);
    matmul_speed(gpu, &mut |shape| rng.q8_0(shape), |cols| {
        cols + 2 * quant::blocks(cols)
    });
}

/// Times a matmul through `quantized` matrices, whose rows take
/// `row_bytes` of their columns, on the model's shapes.
fn matmul_speed<D: Device>(
    gpu: &D,
    quantized: &mut dyn FnMut(&[usize]) -> Tensor,
    row_bytes: impl Fn(usize) -> usize,
) {
    let mut rng = Rng(21);
    let (rows, cols) = (17408, 5120);
    let weight = gpu.upload(quantized(&[rows, cols]));
    // Reading a buffer back waits for the device: a small one, so that
    // the copy is not what is timed.
    let sync = gpu.alloc(1);
    for n in [1, 2, 4, 8, 12, 16, 32, 64, 96, 128, 256] {
        let x = buffer(gpu, &rng.floats(n * cols));
        let mut out = gpu.alloc(n * rows);
        // Warm up until the clocks are up.
        for _ in 0..(1600 / n).max(40) {
            gpu.matmul(&mut out, &weight, &x);
        }
        gpu.read(&sync);
        let runs = (800 / n).max(20);
        let start = std::time::Instant::now();
        for _ in 0..runs {
            gpu.matmul(&mut out, &weight, &x);
        }
        gpu.read(&sync);
        let each = start.elapsed().as_secs_f64() / runs as f64;
        let flops = 2.0 * (rows * cols * n) as f64 / each;
        eprintln!(
            "{rows}x{cols} n={n:<3} {:8.3} ms  {:6.3} ms/token  {:6.2} TFLOP/s",
            each * 1e3,
            each * 1e3 / n as f64,
            flops / 1e12
        );
    }
    // The model's other shapes, for a single token: the ones with fewer
    // rows take the other kernels.
    for (rows, cols) in [(5120, 17408), (6144, 5120), (10240, 5120), (1024, 5120)] {
        let weight = gpu.upload(quantized(&[rows, cols]));
        let x = buffer(gpu, &rng.floats(cols));
        let mut out = gpu.alloc(rows);
        for _ in 0..2000 {
            gpu.matmul(&mut out, &weight, &x);
        }
        gpu.read(&out);
        let runs = 2000;
        let start = std::time::Instant::now();
        for _ in 0..runs {
            gpu.matmul(&mut out, &weight, &x);
        }
        gpu.read(&out);
        let each = start.elapsed().as_secs_f64() / runs as f64;
        let bytes = rows * row_bytes(cols);
        eprintln!(
            "{rows}x{cols} n=1   {:8.3} ms  {:6.0} GB/s",
            each * 1e3,
            bytes as f64 / each / 1e9
        );
    }
}

/// Times attention for a batch of `n` tokens at the model's heads, at
/// depths of context, and prints each depth's time, its rate, and how
/// much of the arithmetic it does a second: a page is read at a depth
/// three times slower than at the start, which the arithmetic of attention
/// does not account for.
pub fn attention_speed<D: Device>(gpu: &D, n: usize) {
    let mut rng = Rng(29);
    let (heads, head_dim, kv_heads) = (24, 256, 4);
    let kv_dim = kv_heads * head_dim;
    let q = buffer(gpu, &rng.floats(n * heads * head_dim));
    let mut out = gpu.alloc(n * heads * head_dim);
    // Reading it waits for the runs before, which is all the wait is for.
    let sync = gpu.alloc(1);
    for pos in [1024, 8192, 32768, 65536] {
        let k = cache(gpu, &rng.floats((pos + n) * kv_dim), 0);
        let v = cache(gpu, &rng.floats((pos + n) * kv_dim), 0);
        let mut run = || gpu.attention(&mut out, &q, &k, &v, pos, heads, head_dim, kv_heads);
        // Warm up until the clocks are up.
        for _ in 0..3 {
            run();
        }
        gpu.read(&sync);
        let runs = (40_000 / pos).max(if n == 1 { 40 } else { 3 });
        let start = std::time::Instant::now();
        for _ in 0..runs {
            run();
        }
        gpu.read(&sync);
        let each = start.elapsed().as_secs_f64() / runs as f64;
        // Each token's heads score and weight the positions up to its own,
        // a multiply and an add per element of the head each.
        let positions: usize = (0..n).map(|t| pos + t + 1).sum();
        let flops = (4 * positions * heads * head_dim) as f64;
        // The caches it reads: every position's keys and values, a byte an
        // element and a half float a block, once for the batch.
        let bytes = 2 * (pos + n) * cache_bytes(kv_dim);
        eprintln!(
            "attention n={n} at {pos:<6} {:8.2} ms  {:6.3} ms/token  {:5.2} TFLOP/s  {:5.0} GB/s",
            each * 1e3,
            each * 1e3 / n as f64,
            flops / each / 1e12,
            bytes as f64 / each / 1e9
        );
    }
}

/// Times a run of small elementwise dispatches, for the cost of a dispatch
/// itself: adds of a token's width of activations, back to back.
pub fn dispatch_overhead<D: Device>(gpu: &D) {
    let mut rng = Rng(22);
    let mut x = buffer(gpu, &rng.floats(5120));
    let y = buffer(gpu, &rng.floats(5120));
    for _ in 0..2000 {
        gpu.add(&mut x, &y);
    }
    gpu.read(&x);
    let runs = 4000;
    let start = std::time::Instant::now();
    for _ in 0..runs {
        gpu.add(&mut x, &y);
    }
    gpu.read(&x);
    eprintln!(
        "add of 5120: {:.2} us a dispatch",
        start.elapsed().as_secs_f64() * 1e6 / runs as f64
    );
}

/// Times every kernel at the model's shapes for a batch of `n` tokens, the
/// matmuls other than the largest included, and prints each one's cost
/// and, times its calls a batch, its share of a token.
pub fn kernel_speed<D: Device>(gpu: &D, n: usize) {
    let mut rng = Rng(23);
    let (hidden, inter, heads, head_dim, rot_dim, kv_heads) = (5120, 17408, 24, 256, 64, 4);
    let (k_heads, v_heads, state_dim) = (16, 48, 128);
    let (k_dim, v_dim) = (k_heads * state_dim, v_heads * state_dim);
    let (q_dim, kv_dim) = (heads * head_dim, kv_heads * head_dim);
    let channels = 2 * k_dim + v_dim;
    let context = 1024;
    let sync = gpu.alloc(1);
    let mut total = 0.0;
    let mut time = |name: &str, calls: usize, op: &mut dyn FnMut()| {
        for _ in 0..(200 / n).max(20) {
            op();
        }
        gpu.read(&sync);
        let runs = (500 / n).max(20);
        let start = std::time::Instant::now();
        for _ in 0..runs {
            op();
        }
        gpu.read(&sync);
        let each = start.elapsed().as_secs_f64() / runs as f64;
        let per_token = each * calls as f64 / n as f64;
        total += per_token;
        eprintln!(
            "{name:<28} {:7.1} us  x{calls:<3} {:6.2} ms/token",
            each * 1e6,
            per_token * 1e3
        );
    };
    let (mut h, mut i, mut qd, mut vd, mut kd) = (
        buffer(gpu, &rng.floats(n * hidden)),
        buffer(gpu, &rng.floats(n * inter)),
        buffer(gpu, &rng.floats(n * q_dim)),
        buffer(gpu, &rng.floats(n * v_dim)),
        buffer(gpu, &rng.floats(n * kv_dim)),
    );
    let (h2, i2, vd2) = (
        buffer(gpu, &rng.floats(n * hidden)),
        buffer(gpu, &rng.floats(n * inter)),
        buffer(gpu, &rng.floats(n * v_dim)),
    );
    let (norm_h, norm_head, norm_state) = (
        buffer(gpu, &rng.floats(hidden)),
        buffer(gpu, &rng.floats(head_dim)),
        buffer(gpu, &rng.floats(state_dim)),
    );
    time("rmsnorm 5120", 48, &mut || {
        gpu.rmsnorm(&mut h, &norm_h, 1e-6)
    });
    time("rmsnorm 6144 by 256", 16, &mut || {
        gpu.rmsnorm(&mut qd, &norm_head, 1e-6)
    });
    time("rmsnorm 1024 by 256", 16, &mut || {
        gpu.rmsnorm(&mut kd, &norm_head, 1e-6)
    });
    time("rmsnorm 6144 by 128", 48, &mut || {
        gpu.rmsnorm(&mut vd, &norm_state, 1e-6)
    });
    time("add 5120", 128, &mut || gpu.add(&mut h, &h2));
    time("copy 5120", 48, &mut || {
        gpu.copy(&mut h, 0, &h2, 0, n * hidden)
    });
    time("silu_mul 17408", 64, &mut || gpu.silu_mul(&mut i, &i2));
    time("silu_mul 6144", 48, &mut || gpu.silu_mul(&mut vd, &vd2));
    time("sigmoid_mul 6144", 16, &mut || {
        gpu.sigmoid_mul(&mut qd, &vd2)
    });
    let mut kq = buffer(gpu, &rng.floats(n * k_dim));
    time("l2norm 2048 by 128", 96, &mut || {
        gpu.l2norm(&mut kq, state_dim, 1e-6)
    });
    let table = buffer(gpu, &rng.floats((context + n) * rot_dim));
    time("rope 24 heads", 16, &mut || {
        gpu.rope(&mut qd, &table, 500, heads, head_dim, rot_dim)
    });
    time("rope 4 heads", 16, &mut || {
        gpu.rope(&mut kd, &table, 500, kv_heads, head_dim, rot_dim)
    });
    let mut k_cache = cache(gpu, &rng.floats((context + n) * kv_dim), 0);
    let v_cache = cache(gpu, &rng.floats((context + n) * kv_dim), 0);
    time("store 1024", 32, &mut || {
        gpu.store(&mut k_cache, 500 * kv_dim, &kd)
    });
    let mut att = gpu.alloc(n * q_dim);
    time("attention at 1024", 16, &mut || {
        gpu.attention(
            &mut att,
            &qd,
            &k_cache,
            &v_cache,
            context - 1,
            heads,
            head_dim,
            kv_heads,
        )
    });
    let long = 4 * context;
    let k_long = cache(gpu, &rng.floats((long + n) * kv_dim), 0);
    let v_long = cache(gpu, &rng.floats((long + n) * kv_dim), 0);
    time("attention at 4096", 16, &mut || {
        gpu.attention(
            &mut att,
            &qd,
            &k_long,
            &v_long,
            long - 1,
            heads,
            head_dim,
            kv_heads,
        )
    });
    let (mut cq, mut ck, mut cv) = (
        gpu.alloc(n * k_dim),
        gpu.alloc(n * k_dim),
        gpu.alloc(n * v_dim),
    );
    let (xc, cstate, cweight) = (
        buffer(gpu, &rng.floats(n * channels)),
        buffer(gpu, &rng.floats((CONV_KERNEL - 1) * channels)),
        buffer(gpu, &rng.floats(channels * CONV_KERNEL)),
    );
    let mut cstate_out = gpu.alloc((CONV_KERNEL - 1) * channels);
    time("conv 10240", 48, &mut || {
        gpu.conv(
            &mut cq,
            &mut ck,
            &mut cv,
            &mut cstate_out,
            &xc,
            &cstate,
            &cweight,
        )
    });
    let gates = buffer(gpu, &rng.floats(n * 2 * v_heads));
    let decay: Vec<f32> = (0..2 * v_heads)
        .map(|i| {
            if i < v_heads {
                -rng.next().abs()
            } else {
                4.0 * rng.next()
            }
        })
        .collect();
    let decay = buffer(gpu, &decay);
    let mut dstate = buffer(gpu, &rng.floats(v_heads * state_dim * state_dim));
    let mut dout = gpu.alloc(n * v_dim);
    time("delta_net 48 heads", 48, &mut || {
        gpu.delta_net(
            &mut dout,
            &cq,
            &ck,
            &cv,
            &gates,
            &decay,
            &mut dstate,
            k_heads,
            v_heads,
            state_dim,
        )
    });
    let ab = gpu.upload(rng.bf16(&[2 * v_heads, hidden]));
    let mut gout = gpu.alloc(n * 2 * v_heads);
    time("matmul bf16 96x5120", 48, &mut || {
        gpu.matmul(&mut gout, &ab, &h)
    });
    for (rows, cols, calls) in [
        (5120, 17408, 64),
        (6144, 5120, 80),
        (10240, 5120, 48),
        (1024, 5120, 32),
    ] {
        let weight = gpu.upload(rng.q4_0(&[rows, cols]));
        let x = buffer(gpu, &rng.floats(n * cols));
        let mut out = gpu.alloc(n * rows);
        time(&format!("matmul {rows}x{cols}"), calls, &mut || {
            gpu.matmul(&mut out, &weight, &x)
        });
    }
    eprintln!("{:<28} {:22} {:6.2} ms/token", "total", "", total * 1e3);
}

pub fn rmsnorm_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(5);
    for (dim, rows) in [(128, 5), (1024, 3), (8, 1)] {
        let w = rng.floats(dim);
        let x = rng.floats(dim * rows);
        let mut want = x.clone();
        Cpu.rmsnorm(&mut want, &w, 1e-6);
        let weight = buffer(gpu, &w);
        let mut buf = buffer(gpu, &x);
        gpu.rmsnorm(&mut buf, &weight, 1e-6);
        close(&gpu.read(&buf), &want, 1e-5);
    }
}

pub fn l2norm_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(15);
    for (dim, rows) in [(128, 5), (1024, 3), (8, 1)] {
        let x = rng.floats(dim * rows);
        let mut want = x.clone();
        Cpu.l2norm(&mut want, dim, 1e-6);
        let mut buf = buffer(gpu, &x);
        gpu.l2norm(&mut buf, dim, 1e-6);
        close(&gpu.read(&buf), &want, 1e-5);
    }
}

pub fn rope_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(6);
    for (n_heads, head_dim, rot_dim, n, pos) in [(4, 16, 16, 3, 7), (3, 256, 64, 2, 5)] {
        let table = rng.floats((pos + n) * rot_dim);
        let x = rng.floats(n * n_heads * head_dim);
        let mut want = x.clone();
        Cpu.rope(&mut want, &table, pos, n_heads, head_dim, rot_dim);
        let table = buffer(gpu, &table);
        let mut buf = buffer(gpu, &x);
        gpu.rope(&mut buf, &table, pos, n_heads, head_dim, rot_dim);
        close(&gpu.read(&buf), &want, 1e-5);
    }
}

pub fn attention_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(7);
    // Each case ends with the magnitude of the caches. The last three are
    // single tokens deep into a context. The one before them has more
    // tokens of the model's heads than the kernel on Apple's matrix
    // accelerators packs the queries of at once, so the batch goes through
    // two dispatches there, over three chunks; the one before it is the
    // model's heads for a batch whose positions cross what a device may
    // take in one piece; the one before that, with as many query heads to a
    // key/value head as the kernels take, attends over more positions than
    // one dispatch of either backend has partials for, so the batch goes
    // through several; the two before that have caches small enough that
    // their blocks' scales are subnormal f16s, which a driver may flush to
    // zero.
    let cases = [
        (4, 32, 2, 3, 5, 1.0),
        (16, 128, 8, 2, 300, 1.0),
        (2, 128, 1, 1, 0, 1.0),
        (24, 256, 4, 3, 1000, 1.0),
        (3, 96, 1, 2, 9, 1.0),
        (2, 128, 1, 1, 0, 1e-3),
        (24, 256, 4, 3, 1000, 1e-3),
        (32, 128, 4, 64, 32704, 1.0),
        (24, 256, 4, 64, 470, 1.0),
        (24, 256, 4, 700, 470, 1.0),
        // A token at a time deep into a context, which a device may take
        // on other kernels than a token near the start: the model's heads,
        // twice, and heads of 128 two to a key/value head.
        (24, 256, 4, 1, 5000, 1.0),
        (24, 256, 4, 1, 40000, 1.0),
        (16, 128, 8, 1, 3000, 1.0),
    ];
    for (n_heads, head_dim, n_kv_heads, n, pos, size) in cases {
        let kv_dim = n_kv_heads * head_dim;
        let q = rng.floats(n * n_heads * head_dim);
        let mut scaled = |n| -> Vec<f32> { rng.floats(n).iter().map(|x| x * size).collect() };
        let k_cache = scaled((pos + n) * kv_dim);
        let v_cache = scaled((pos + n) * kv_dim);
        let mut want = vec![0.0; q.len()];
        let (cpu_k, cpu_v) = (
            cache(&Cpu, &k_cache, pos * kv_dim),
            cache(&Cpu, &v_cache, pos * kv_dim),
        );
        Cpu.attention(
            &mut want, &q, &cpu_k, &cpu_v, pos, n_heads, head_dim, n_kv_heads,
        );
        let q = buffer(gpu, &q);
        let (k, v) = (
            cache(gpu, &k_cache, pos * kv_dim),
            cache(gpu, &v_cache, pos * kv_dim),
        );
        let mut out = gpu.alloc(want.len());
        gpu.attention(&mut out, &q, &k, &v, pos, n_heads, head_dim, n_kv_heads);
        // The caches hold the same bits, which `store_quantizes_like_cpu`
        // checks; only the order of the sums differs. The outputs are as
        // small as the values, so they are compared at the values' size.
        let unscale = |x: Vec<f32>| -> Vec<f32> { x.iter().map(|x| x / size).collect() };
        close(&unscale(gpu.read(&out)), &unscale(want), 1e-4);
    }
}

pub fn store_quantizes_like_cpu<D: Device>(gpu: &D) {
    // Random blocks, and blocks at the edges: ties between multiples of the
    // scale, magnitudes past what an f16 scale reaches, scales that are
    // subnormal f16s or round to zero, and a block of zeros.
    let mut data = Rng(9).floats(64 * CACHE_BLOCK);
    let edges: [&[f32]; 7] = [
        &[127.0, 0.5, 1.5, -2.5, 126.5, -0.0, 3.5000002, -3.4999998],
        &[1e6, -1e6, 65504.0, 65520.0, 0.1],
        &[1e-3, -3e-7, 2e-4, 1e-9],
        &[1e-6, 7e-7, -3e-7],
        &[1e-9, -1e-12],
        &[0.0],
        &[0.3, 0.3 / 127.0 * 64.5, -0.3 / 127.0 * 0.5],
    ];
    for (block, edge) in data.chunks_exact_mut(CACHE_BLOCK).zip(edges) {
        block.fill(0.0);
        block[..edge.len()].copy_from_slice(edge);
    }
    let len = data.len();
    // Split at an odd block, so that a store starts part way into what a
    // device may keep together.
    let split = 11 * CACHE_BLOCK;
    let want = Cpu.read_cache(&cache(&Cpu, &data, split), len);
    assert_eq!(gpu.read_cache(&cache(gpu, &data, split), len), want);
}

pub fn cache_round_trips<D: Device>(gpu: &D) {
    let len = 32 * CACHE_BLOCK;
    let data = Rng(10).floats(len);
    let mut c = cache(gpu, &data, 7 * CACHE_BLOCK);
    let blocks: Vec<_> = data
        .chunks_exact(CACHE_BLOCK)
        .map(quant::quantize)
        .collect();
    let bytes = |blocks: &[(u16, [i8; CACHE_BLOCK])]| -> Vec<u8> {
        let quants = blocks.iter().flat_map(|(_, q)| q.map(|q| q as u8));
        let scales = blocks.iter().flat_map(|(s, _)| s.to_le_bytes());
        quants.chain(scales).collect()
    };
    assert_eq!(gpu.read_cache(&c, len), bytes(&blocks));
    assert_eq!(gpu.read_cache(&c, 3 * CACHE_BLOCK), bytes(&blocks[..3]));
    let mut written = blocks.clone();
    for (i, (scale, q)) in written[..13].iter_mut().enumerate() {
        *scale = i as u16 * 3;
        *q = std::array::from_fn(|j| (i * 7 + j) as i8);
    }
    gpu.write_cache(&mut c, &bytes(&written[..13]));
    assert_eq!(gpu.read_cache(&c, len), bytes(&written));
}

pub fn elementwise_match_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(8);
    let a = rng.floats(3001);
    let b = rng.floats(3001);
    let mut want = a.clone();
    Cpu.silu_mul(&mut want, &b);
    let mut gate = buffer(gpu, &a);
    let up = buffer(gpu, &b);
    gpu.silu_mul(&mut gate, &up);
    close(&gpu.read(&gate), &want, 1e-6);

    let mut want = a.clone();
    Cpu.sigmoid_mul(&mut want, &b);
    let mut x = buffer(gpu, &a);
    gpu.sigmoid_mul(&mut x, &up);
    close(&gpu.read(&x), &want, 1e-6);

    let mut want = a.clone();
    Cpu.add(&mut want, &b);
    let mut x = buffer(gpu, &a);
    gpu.add(&mut x, &up);
    close(&gpu.read(&x), &want, 1e-6);
}

/// The result may be held in half precision, and is what the operations
/// after it take, whichever way the device holds it.
pub fn normalize_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(21);
    for (dim, rows) in [(128, 5), (1024, 3), (8, 1)] {
        let w = rng.floats(dim);
        let x = rng.floats(dim * (rows + 1));
        let y = rng.floats(dim * rows);
        let mut want = vec![0.0; dim * rows];
        Cpu.normalize(&mut want, &x, &w, dim * rows, 1e-6);
        let mut out = gpu.alloc(dim * rows);
        gpu.normalize(
            &mut out,
            &buffer(gpu, &x),
            &buffer(gpu, &w),
            dim * rows,
            1e-6,
        );
        close(&gpu.read(&out), &want, 1e-3);
        Cpu.add(&mut want, &y);
        gpu.add(&mut out, &buffer(gpu, &y));
        close(&gpu.read(&out), &want, 1e-3);
        let mut part = gpu.alloc(dim);
        gpu.copy(&mut part, 0, &out, dim * (rows - 1), dim);
        close(&gpu.read(&part), &want[dim * (rows - 1)..], 1e-3);
    }
}

/// Shapes at the edges of the tile kernels' tiles and of their batches of
/// tokens, with an odd number of rows among them.
const FUSED_SHAPES: [(usize, usize, usize); 6] = [
    (64, 128, 16),
    (200, 5120, 1),
    (77, 1056, 3),
    (64, 160, 9),
    (301, 1152, 130),
    (2500, 512, 70),
];

pub fn swiglu_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(22);
    for (rows, cols, n) in FUSED_SHAPES {
        let (gate, up) = (rng.q4_0(&[rows, cols]), rng.q4_0(&[rows, cols]));
        let x = rng.floats(n * cols);
        let mut want = vec![0.0; n * rows];
        Cpu.swiglu(
            &mut want,
            &clone(&gate),
            &clone(&up),
            &x,
            &mut vec![0.0; n * rows],
        );
        let (gate, up) = (gpu.upload(gate), gpu.upload(up));
        let mut out = gpu.alloc(n * rows);
        let mut scratch = gpu.alloc(n * rows);
        gpu.swiglu(&mut out, &gate, &up, &buffer(gpu, &x), &mut scratch);
        close_batch(&gpu.read(&out), &want, &format!("{rows}x{cols} n={n}"));
    }
}

pub fn matmul_add_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(23);
    for (rows, cols, n) in FUSED_SHAPES {
        for w in [
            rng.q4_0(&[rows, cols]),
            rng.q8_0(&[rows, cols]),
            rng.bf16(&[rows, cols]),
        ] {
            let input = rng.floats(n * cols);
            let x = rng.floats(n * rows);
            let mut want = x.clone();
            Cpu.matmul_add(&mut want, &clone(&w), &input, &mut vec![0.0; n * rows]);
            let mut out = buffer(gpu, &x);
            let mut scratch = gpu.alloc(n * rows);
            gpu.matmul_add(&mut out, &gpu.upload(w), &buffer(gpu, &input), &mut scratch);
            // What was added, which is what a batch's error is a part of.
            let added =
                |sum: &[f32]| -> Vec<f32> { sum.iter().zip(&x).map(|(s, x)| s - x).collect() };
            close_batch(
                &added(&gpu.read(&out)),
                &added(&want),
                &format!("{rows}x{cols} n={n}"),
            );
        }
    }
}

pub fn conv_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(11);
    let (q_dim, k_dim, v_dim) = (16, 16, 48);
    let channels = q_dim + k_dim + v_dim;
    for n in [1, 2, 7] {
        let x = rng.floats(n * channels);
        let state = rng.floats((CONV_KERNEL - 1) * channels);
        let weight = rng.floats(channels * CONV_KERNEL);
        let (mut want_q, mut want_k, mut want_v) = (
            vec![0.0; n * q_dim],
            vec![0.0; n * k_dim],
            vec![0.0; n * v_dim],
        );
        let mut want_state = vec![0.0; state.len()];
        Cpu.conv(
            &mut want_q,
            &mut want_k,
            &mut want_v,
            &mut want_state,
            &x,
            &state,
            &weight,
        );
        let (mut q, mut k, mut v) = (
            gpu.alloc(n * q_dim),
            gpu.alloc(n * k_dim),
            gpu.alloc(n * v_dim),
        );
        let mut state_out = gpu.alloc(state.len());
        gpu.conv(
            &mut q,
            &mut k,
            &mut v,
            &mut state_out,
            &buffer(gpu, &x),
            &buffer(gpu, &state),
            &buffer(gpu, &weight),
        );
        close(&gpu.read(&q), &want_q, 1e-5);
        close(&gpu.read(&k), &want_k, 1e-5);
        close(&gpu.read(&v), &want_v, 1e-5);
        close(&gpu.read(&state_out), &want_state, 1e-6);
    }
}

pub fn delta_net_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(12);
    let (n_k_heads, n_v_heads, head_dim) = (2, 6, 128);
    for n in [1, 5] {
        let q = rng.floats(n * n_k_heads * head_dim);
        let k = rng.floats(n * n_k_heads * head_dim);
        let v = rng.floats(n * n_v_heads * head_dim);
        let gates = rng.floats(n * 2 * n_v_heads);
        let decay: Vec<f32> = (0..2 * n_v_heads)
            .map(|i| {
                if i < n_v_heads {
                    -rng.next().abs()
                } else {
                    4.0 * rng.next()
                }
            })
            .collect();
        let state = rng.floats(n_v_heads * head_dim * head_dim);
        let mut want = vec![0.0; v.len()];
        let mut want_state = state.clone();
        Cpu.delta_net(
            &mut want,
            &q,
            &k,
            &v,
            &gates,
            &decay,
            &mut want_state,
            n_k_heads,
            n_v_heads,
            head_dim,
        );
        let mut out = gpu.alloc(v.len());
        let mut gpu_state = buffer(gpu, &state);
        gpu.delta_net(
            &mut out,
            &buffer(gpu, &q),
            &buffer(gpu, &k),
            &buffer(gpu, &v),
            &buffer(gpu, &gates),
            &buffer(gpu, &decay),
            &mut gpu_state,
            n_k_heads,
            n_v_heads,
            head_dim,
        );
        close(&gpu.read(&out), &want, 1e-4);
        close(&gpu.read(&gpu_state), &want_state, 1e-4);
    }
}

pub fn resize_keeps_capacity<D: Device>(gpu: &D) {
    let mut buf = gpu.alloc(64);
    gpu.resize(&mut buf, 16);
    assert_eq!(gpu.read(&buf).len(), 16);
    gpu.resize(&mut buf, 64);
    assert_eq!(gpu.read(&buf).len(), 64);
}

/// The experts' matrix multiplication, where the device runs it: over
/// NVFP4 and Q8_0 weights, with an input row per token and one per expert
/// of a token; for a few tokens, with an expert picked twice and one never
/// picked, and for a batch, which a device may take an expert at a time,
/// with one expert picked by more tokens than a tile of them holds.
pub fn expert_matmul_matches_cpu<D: Device>(gpu: &D) {
    if !gpu.takes_nvfp4() {
        return;
    }
    let mut rng = Rng(2304);
    let (experts, k) = (5, 3);
    let few: Vec<f32> = [0, 4, 2, 2, 0, 1, 4, 4, 1, 1, 2, 0]
        .iter()
        .map(|&e| e as f32)
        .collect();
    // Every token of the batch picks expert 2, most pick 0, and none 3.
    let batch: Vec<f32> = (0..70)
        .flat_map(|t| [2, if t % 4 == 0 { 4 } else { 0 }, 1 + 3 * (t % 2)])
        .map(|e| e as f32)
        .collect();
    let scale = rng.floats(experts);
    for ids in [few, batch] {
        let n = ids.len() / k;
        for (rows, cols) in [(40, 64), (192, 2304), (2304, 1152)] {
            for nvfp4 in [true, false] {
                for shared in [true, false] {
                    let w = if nvfp4 {
                        rng.nvfp4(&[experts * rows, cols])
                    } else {
                        rng.q8_0(&[experts * rows, cols])
                    };
                    let x = rng.floats(if shared { n } else { n * k } * cols);
                    let mut want = vec![0.0; n * k * rows];
                    let host = clone(&w);
                    Cpu.expert_matmul(&mut want, &host, &scale, &x, &ids, k);
                    let mut out = gpu.alloc(n * k * rows);
                    let on_gpu = buffer(gpu, &ids);
                    let (w, scales) = (gpu.upload(w), buffer(gpu, &scale));
                    gpu.expert_matmul(&mut out, &w, &scales, &buffer(gpu, &x), &on_gpu, k);
                    let what = format!("{rows}x{cols} n={n} nvfp4={nvfp4} shared={shared}");
                    close_batch(&gpu.read(&out), &want, &what);
                    // And again with the same picks, as a layer's second
                    // multiplication is.
                    let x = rng.floats(if shared { n } else { n * k } * cols);
                    Cpu.expert_matmul(&mut want, &host, &scale, &x, &ids, k);
                    gpu.expert_matmul(&mut out, &w, &scales, &buffer(gpu, &x), &on_gpu, k);
                    close_batch(&gpu.read(&out), &want, &format!("{what} again"));
                }
            }
        }
    }
}

/// The mix over the depth, of one state and of as many as a model keeps,
/// of the model's width and of a narrow one.
pub fn mix_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(60);
    for (dim, count, n) in [(2304, 5, 3), (64, 1, 2), (2304, 2, 64), (128, 6, 1)] {
        // States of different sizes, so that normalizing them matters.
        let states: Vec<Vec<f32>> = (0..count)
            .map(|s| {
                let size = 0.1 + 3.0 * s as f32;
                rng.floats(n * dim).iter().map(|x| x * size).collect()
            })
            .collect();
        let query: Vec<f32> = rng.floats(dim).iter().map(|x| x * 4.0).collect();
        let mut want = vec![0.0; n * dim];
        Cpu.mix(&mut want, &states.iter().collect::<Vec<_>>(), &query, 1e-6);
        let on_gpu: Vec<D::Buffer> = states.iter().map(|state| buffer(gpu, state)).collect();
        let mut out = gpu.alloc(n * dim);
        gpu.mix(
            &mut out,
            &on_gpu.iter().collect::<Vec<_>>(),
            &buffer(gpu, &query),
            1e-6,
        );
        close(&gpu.read(&out), &want, 1e-4);
    }
}

/// The short convolution, over batches of every size against its taps,
/// with and without taps of the heads' own, its state carried from a batch
/// to the next.
pub fn short_conv_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(61);
    for (channels, taps, heads) in [(1024, 4, 0), (1024, 4, 8), (2304, 4, 0), (48, 3, 4)] {
        let weight = rng.floats(taps * channels);
        let mut cpu_state = rng.floats((taps - 1) * channels);
        let mut state = buffer(gpu, &cpu_state);
        let on_gpu = buffer(gpu, &weight);
        for n in [1, 1, 2, 3, 5, 64, 1] {
            let x = rng.floats(n * channels);
            let dynamic = rng.floats(n * heads.max(1) * taps);
            let mut want = vec![0.0; n * channels];
            Cpu.short_conv(
                &mut want,
                &x,
                &mut cpu_state,
                &weight,
                (heads > 0).then_some((&dynamic, heads)),
            );
            let mut out = gpu.alloc(n * channels);
            let dynamic = buffer(gpu, &dynamic);
            gpu.short_conv(
                &mut out,
                &buffer(gpu, &x),
                &mut state,
                &on_gpu,
                (heads > 0).then_some((&dynamic, heads)),
            );
            close(&gpu.read(&out), &want, 1e-5);
            close(&gpu.read(&state), &cpu_state, 0.0);
        }
    }
}

/// Attention with a bias by distance and a sink: over everything and over
/// a window, in caches as long as the sequence and in rings the positions
/// have gone round, of one chunk of positions and of several, a batch and
/// a token at a time.
pub fn attention_rel_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(62);
    // Heads, their width, key/value heads, tokens, the position of the
    // first, the window, the rows of the caches, and the distances the
    // bias has rows for.
    let cases = [
        (4, 32, 2, 3, 5, usize::MAX, 8, 4),
        (24, 128, 8, 2, 300, usize::MAX, 302, 1024),
        (24, 128, 8, 1, 0, usize::MAX, 16, 1024),
        (24, 128, 8, 64, 2000, 1024, 1088, 1024),
        (24, 128, 8, 1, 5000, 1024, 1088, 1024),
        (24, 128, 8, 5, 40, 16, 24, 16),
        (24, 128, 8, 3, 1500, usize::MAX, 1503, 100),
        // A token deep into a context, which a device may take on other
        // kernels than a token near the start.
        (24, 128, 8, 1, 3000, usize::MAX, 3001, 1024),
        (24, 128, 8, 1, 40000, usize::MAX, 40001, 1024),
        (6, 64, 2, 64, 100, 7, 71, 7),
    ];
    let n_rel = 16;
    for (n_heads, head_dim, n_kv_heads, n, pos, window, rows, extent) in cases {
        let kv_dim = n_kv_heads * head_dim;
        let q: Vec<f32> = rng.floats(n * n_heads * head_dim);
        let rel = rng.floats(n * n_heads * n_rel);
        let bias = rng.floats(extent * n_rel);
        // Sinks from far below the keys' scores to above them.
        let sinks: Vec<f32> = rng.floats(n * n_heads).iter().map(|x| x * 6.0).collect();
        // What the caches hold once the batch's keys are stored: the
        // latest positions, each in its row.
        let held = (pos + n).min(rows);
        let (k_cache, v_cache) = (rng.floats(held * kv_dim), rng.floats(held * kv_dim));
        let mut want = vec![0.0; q.len()];
        Cpu.attention_rel(
            &mut want,
            &q,
            &cache(&Cpu, &k_cache, 0),
            &cache(&Cpu, &v_cache, 0),
            &rel,
            &bias,
            &sinks,
            pos,
            n_heads,
            head_dim,
            n_kv_heads,
            window,
            rows,
        );
        let mut out = gpu.alloc(want.len());
        gpu.attention_rel(
            &mut out,
            &buffer(gpu, &q),
            &cache(gpu, &k_cache, 0),
            &cache(gpu, &v_cache, 0),
            &buffer(gpu, &rel),
            &buffer(gpu, &bias),
            &buffer(gpu, &sinks),
            pos,
            n_heads,
            head_dim,
            n_kv_heads,
            window,
            rows,
        );
        // A batch of heads of 128 may go through kernels that hold the
        // queries, the keys and the softmax's weights as half floats, and
        // is held to a batch's error.
        if (n > 1 || pos >= 2048) && head_dim == 128 {
            close_batch(&gpu.read(&out), &want, &format!("{n} tokens at {pos}"));
        } else {
            close(&gpu.read(&out), &want, 1e-4);
        }
    }
}

/// The router: the experts picked are the same, and what each counts for.
pub fn route_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(63);
    for (experts, k, n) in [(128, 8, 5), (4, 2, 1), (256, 32, 3), (16, 16, 2)] {
        let logits: Vec<f32> = rng.floats(n * experts).iter().map(|x| x * 4.0).collect();
        let bias: Vec<f32> = rng.floats(experts).iter().map(|x| x * 0.2).collect();
        let (mut want_ids, mut want_weights) = (vec![0.0; n * k], vec![0.0; n * k]);
        Cpu.route(&mut want_ids, &mut want_weights, &logits, &bias, k, 8.0);
        let (mut ids, mut weights) = (gpu.alloc(n * k), gpu.alloc(n * k));
        gpu.route(
            &mut ids,
            &mut weights,
            &buffer(gpu, &logits),
            &buffer(gpu, &bias),
            k,
            8.0,
        );
        assert_eq!(gpu.read(&ids), want_ids);
        close(&gpu.read(&weights), &want_weights, 1e-5);
    }
}

/// The operations a thread an element takes: the gate on a message, the
/// clip, the scaling of rows, the SiTU activation, and the experts' sum.
pub fn rows_and_gates_match_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(64);
    let mut wide =
        |n: usize, size: f32| -> Vec<f32> { rng.floats(n).iter().map(|x| x * size).collect() };

    let len = 3 * 2304;
    let (x, forget, gain, message) = (
        wide(len, 2.0),
        wide(len, 3.0),
        wide(len, 3.0),
        wide(len, 2.0),
    );
    let mut want = x.clone();
    Cpu.gated_update(&mut want, &forget, &gain, &message);
    let mut got = buffer(gpu, &x);
    gpu.gated_update(
        &mut got,
        &buffer(gpu, &forget),
        &buffer(gpu, &gain),
        &buffer(gpu, &message),
    );
    close(&gpu.read(&got), &want, 1e-5);

    // Heads longer than the threshold and shorter, of the queries' width
    // and of the bias's.
    for (heads, dim, n, threshold) in [(24, 128, 3, 22.6), (24, 16, 5, 8.0), (8, 128, 2, 0.5)] {
        let x = wide(n * heads * dim, 6.0);
        let scale = wide(heads * dim, 1.5);
        let mut want = x.clone();
        Cpu.clip(&mut want, &scale, dim, threshold);
        let mut got = buffer(gpu, &x);
        gpu.clip(&mut got, &buffer(gpu, &scale), dim, threshold);
        close(&gpu.read(&got), &want, 1e-5);
    }

    let (x, factors) = (wide(5 * 3072, 1.0), wide(5, 2.0));
    let mut want = x.clone();
    Cpu.scale_rows(&mut want, &factors);
    let mut got = buffer(gpu, &x);
    gpu.scale_rows(&mut got, &buffer(gpu, &factors));
    close(&gpu.read(&got), &want, 0.0);

    // Gates and up projections past where the activation's tanh bends.
    let (width, rows) = (1152, 16);
    let gate_up = wide(rows * 2 * width, 40.0);
    let mut want = vec![0.0; rows * width];
    Cpu.situ(&mut want, &gate_up, width, 4.0, 25.0);
    let mut got = gpu.alloc(rows * width);
    gpu.situ(&mut got, &buffer(gpu, &gate_up), width, 4.0, 25.0);
    close(&gpu.read(&got), &want, 1e-4);

    let (width, k, n) = (2304, 8, 3);
    let (y, weights) = (wide(n * k * width, 1.0), wide(n * k, 2.0));
    let mut want = vec![0.0; n * width];
    Cpu.weighted_sum(&mut want, &y, &weights);
    let mut got = gpu.alloc(n * width);
    gpu.weighted_sum(&mut got, &buffer(gpu, &y), &buffer(gpu, &weights));
    close(&gpu.read(&got), &want, 1e-5);
}

/// A copy of the first of a cache's activations takes those, and leaves
/// the rest of the other as it was.
pub fn copy_cache_copies_a_prefix<D: Device>(gpu: &D) {
    let len = 34 * CACHE_BLOCK;
    let (from, to) = (Rng(65).floats(len), Rng(66).floats(len));
    let src = cache(gpu, &from, 0);
    let mut dst = cache(gpu, &to, 0);
    let part = 21 * CACHE_BLOCK;
    gpu.copy_cache(&mut dst, &src, part);
    let (copied, kept) = (
        gpu.read_cache(&src, len),
        gpu.read_cache(&cache(gpu, &to, 0), len),
    );
    let got = gpu.read_cache(&dst, len);
    // The quants, then the scales, two bytes a block.
    assert_eq!(got[..part], copied[..part]);
    assert_eq!(got[part..len], kept[part..len]);
    let (blocks, scales) = (part / CACHE_BLOCK * 2, len);
    assert_eq!(got[scales..][..blocks], copied[scales..][..blocks]);
    assert_eq!(got[scales + blocks..], kept[scales + blocks..]);
}

/// The gate on a message, of the model's width and rank and of narrow
/// ones, a token and a batch.
pub fn gate_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(67);
    for (dim, rank, n) in [(2304, 64, 3), (64, 8, 1), (2304, 64, 64), (128, 16, 5)] {
        let x: Vec<f32> = rng.floats(n * dim).iter().map(|x| x * 3.0).collect();
        let message = rng.floats(n * dim);
        // Weights small enough that the gate is not shut or open all the
        // way, where what is left of `x` is the difference of two numbers
        // almost the same.
        let mut small = |rows: usize, cols: usize| Tensor::Bf16 {
            shape: vec![rows, cols],
            data: rng
                .floats(rows * cols)
                .iter()
                .map(|w| crate::gpu::float::to_bf16(w / (cols as f32).sqrt()))
                .collect(),
        };
        let weights = [
            small(rank, dim),
            small(rank, dim),
            small(dim, rank),
            small(dim, rank),
        ];
        let mut want = x.clone();
        let [a, b, c, d] = &weights;
        Cpu.gate(
            &mut want,
            &message,
            &clone(a),
            &clone(b),
            &clone(c),
            &clone(d),
            dim,
            rank,
            1e-6,
        );
        let [a, b, c, d] = weights.map(|w| gpu.upload(w));
        let mut got = buffer(gpu, &x);
        gpu.gate(
            &mut got,
            &buffer(gpu, &message),
            &a,
            &b,
            &c,
            &d,
            dim,
            rank,
            1e-6,
        );
        close(&gpu.read(&got), &want, 1e-4);
    }
}
