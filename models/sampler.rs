/// Picks the next token from a model's logits: from the `top_k` most likely
/// tokens, then the smallest set of those whose probability reaches `top_p`,
/// at `temperature`. A temperature of zero always picks the most likely token.
pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    rng: Rng,
}

impl Sampler {
    pub fn new(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            rng: Rng(seed.max(1)),
        }
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        let mut candidates: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(token, &logit)| (token as u32, logit))
            .collect();
        let k = self.top_k.clamp(1, candidates.len());
        candidates.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        candidates.truncate(k);
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        if self.temperature == 0.0 {
            return candidates[0].0;
        }

        let max = candidates[0].1;
        let mut probs: Vec<f32> = candidates
            .iter()
            .map(|&(_, logit)| ((logit - max) / self.temperature).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum;
        }

        // Keep the most likely tokens up to a total probability of top_p.
        let mut kept = 0.0;
        let mut n = 0;
        while n < probs.len() && kept < self.top_p {
            kept += probs[n];
            n += 1;
        }
        let r = self.rng.next() * kept;
        let mut acc = 0.0;
        for i in 0..n {
            acc += probs[i];
            if r < acc {
                return candidates[i].0;
            }
        }
        candidates[n - 1].0
    }
}

/// xorshift64*, uniform in [0, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545F4914F6CDD1D);
        (bits >> 40) as f32 / (1u64 << 24) as f32
    }
}
