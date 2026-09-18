//! Checks of a GPU device's operations against the CPU reference.

use crate::{Cpu, Device, Tensor};

/// Defines a test for each check, run on the device `$open` returns, and
/// skipped if it returns an error.
macro_rules! check_against_cpu {
    ($open:expr) => {
        check_against_cpu!(
            $open;
            write_and_read_round_trip,
            alloc_is_zeroed_and_copy_moves_ranges,
            embed_matches_cpu,
            matmul_matches_cpu,
            rmsnorm_matches_cpu,
            rope_matches_cpu,
            attention_matches_cpu,
            store_rounds_like_cpu,
            elementwise_match_cpu,
            resize_keeps_capacity
        );
    };
    ($open:expr; $($check:ident),*) => {
        $(
            #[test]
            fn $check() {
                match $open {
                    Ok(gpu) => crate::tests::$check(&gpu),
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
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn floats(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }

    fn tensor(&mut self, shape: &[usize]) -> Tensor {
        let n = shape.iter().product();
        Tensor {
            shape: shape.to_vec(),
            data: (0..n).map(|_| (self.next().to_bits() >> 16) as u16).collect(),
        }
    }
}

fn bf16_tensor(t: &Tensor) -> Tensor {
    Tensor {
        shape: t.shape.clone(),
        data: t.data.clone(),
    }
}

fn close(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!((a - b).abs() <= tolerance * (1.0 + a.abs().max(b.abs())), "element {i}: {a} vs {b}");
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

pub fn embed_matches_cpu<D: Device>(gpu: &D) {
    let table = Rng(3).tensor(&[50, 24]);
    let tokens = [3, 49, 0, 17];
    let mut want = vec![0.0; tokens.len() * 24];
    Cpu.embed(&mut want, &bf16_tensor(&table), &tokens);
    let weight = gpu.upload(table);
    let mut out = gpu.alloc(tokens.len() * 24);
    gpu.embed(&mut out, &weight, &tokens);
    assert_eq!(gpu.read(&out), want);
}

pub fn matmul_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(4);
    for (rows, cols, n) in [(1, 8, 1), (200, 192, 1), (77, 1032, 3), (70_000, 8, 2)] {
        let w = rng.tensor(&[rows, cols]);
        let x = rng.floats(n * cols);
        let mut want = vec![0.0; n * rows];
        Cpu.matmul(&mut want, &bf16_tensor(&w), &x);
        let weight = gpu.upload(w);
        let x = buffer(gpu, &x);
        let mut out = gpu.alloc(n * rows);
        gpu.matmul(&mut out, &weight, &x);
        close(&gpu.read(&out), &want, 1e-4);
    }
}

pub fn rmsnorm_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(5);
    for (dim, rows) in [(128, 5), (1024, 3), (8, 1)] {
        let w = rng.tensor(&[dim]);
        let x = rng.floats(dim * rows);
        let mut want = x.clone();
        Cpu.rmsnorm(&mut want, &bf16_tensor(&w), 1e-6);
        let weight = gpu.upload(w);
        let mut buf = buffer(gpu, &x);
        gpu.rmsnorm(&mut buf, &weight, 1e-6);
        close(&gpu.read(&buf), &want, 1e-5);
    }
}

pub fn rope_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(6);
    let (n_heads, head_dim, n, pos) = (4, 16, 3, 7);
    let table = rng.floats((pos + n) * head_dim);
    let x = rng.floats(n * n_heads * head_dim);
    let mut want = x.clone();
    Cpu.rope(&mut want, &table, pos, n_heads, head_dim);
    let table = buffer(gpu, &table);
    let mut buf = buffer(gpu, &x);
    gpu.rope(&mut buf, &table, pos, n_heads, head_dim);
    close(&gpu.read(&buf), &want, 1e-5);
}

pub fn attention_matches_cpu<D: Device>(gpu: &D) {
    let mut rng = Rng(7);
    // The last attends over more positions than one dispatch has scores
    // for, all its tokens at once.
    let cases = [(4, 8, 2, 3, 5), (16, 128, 8, 2, 300), (2, 128, 1, 1, 0), (32, 128, 4, 3, 1000), (16, 128, 4, 64, 32704)];
    for (n_heads, head_dim, n_kv_heads, n, pos) in cases {
        let kv_dim = n_kv_heads * head_dim;
        let q = rng.floats(n * n_heads * head_dim);
        let k_cache = rng.floats((pos + n) * kv_dim);
        let v_cache = rng.floats((pos + n) * kv_dim);
        let mut want = vec![0.0; q.len()];
        let (cpu_k, cpu_v) = (cache(&Cpu, &k_cache, pos * kv_dim), cache(&Cpu, &v_cache, pos * kv_dim));
        Cpu.attention(&mut want, &q, &cpu_k, &cpu_v, pos, n_heads, head_dim, n_kv_heads);
        let q = buffer(gpu, &q);
        let (k, v) = (cache(gpu, &k_cache, pos * kv_dim), cache(gpu, &v_cache, pos * kv_dim));
        let mut out = gpu.alloc(want.len());
        gpu.attention(&mut out, &q, &k, &v, pos, n_heads, head_dim, n_kv_heads);
        close(&gpu.read(&out), &want, 1e-4);
    }
}

pub fn store_rounds_like_cpu<D: Device>(gpu: &D) {
    // Attention over a single position weighs its value by exactly one, so
    // it reads back what the value cache holds.
    let head_dim = 128;
    let mut v = Rng(9).floats(head_dim);
    let edges = [
        1e6,
        -1e6,
        65504.0,
        65520.0,
        1.0 + 2.0f32.powi(-11),
        1.0 + 3.0 * 2.0f32.powi(-11),
        0.1,
        1e-9,
        -3e-7,
        2.0f32.powi(-14) * 0.99999,
        -0.0,
    ];
    v[..edges.len()].copy_from_slice(&edges);
    let q = vec![0.0; head_dim];
    let mut want = vec![0.0; head_dim];
    let cpu_v = cache(&Cpu, &v, 0);
    Cpu.attention(&mut want, &q, &cpu_v, &cpu_v, 0, 1, head_dim, 1);
    let gpu_v = cache(gpu, &v, 0);
    let mut out = gpu.alloc(head_dim);
    gpu.attention(&mut out, &buffer(gpu, &q), &gpu_v, &gpu_v, 0, 1, head_dim, 1);
    assert_eq!(gpu.read(&out), want);
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
    Cpu.add(&mut want, &b);
    let mut x = buffer(gpu, &a);
    gpu.add(&mut x, &up);
    close(&gpu.read(&x), &want, 1e-6);
}

pub fn resize_keeps_capacity<D: Device>(gpu: &D) {
    let mut buf = gpu.alloc(64);
    gpu.resize(&mut buf, 16);
    assert_eq!(gpu.read(&buf).len(), 16);
    gpu.resize(&mut buf, 64);
    assert_eq!(gpu.read(&buf).len(), 64);
}
