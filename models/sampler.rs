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
        // The top k in one pass over the logits, kept in order: a vocabulary
        // is a quarter of a million entries, and k is twenty.
        let k = self.top_k.clamp(1, logits.len());
        let mut candidates: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
        for (token, &logit) in logits.iter().enumerate() {
            if candidates.len() == k && logit <= candidates[k - 1].1 {
                continue;
            }
            let at = candidates.partition_point(|&(_, l)| l >= logit);
            candidates.insert(at, (token as u32, logit));
            candidates.truncate(k);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_from_the_top_k_in_order() {
        let mut logits = vec![0.0; 1000];
        logits[7] = 5.0;
        logits[300] = 4.0;
        logits[999] = 3.0;
        let mut greedy = Sampler::new(0.0, 20, 0.95, 1);
        assert_eq!(greedy.sample(&logits), 7);
        // At a low temperature the top token is all but certain, and with
        // top_p of nearly nothing it is the only one kept.
        let mut sampler = Sampler::new(1.0, 3, 0.01, 1);
        for _ in 0..20 {
            assert_eq!(sampler.sample(&logits), 7);
        }
        // With k of two, the third never comes up.
        let mut sampler = Sampler::new(100.0, 2, 1.0, 1);
        let picks: std::collections::HashSet<u32> = (0..200).map(|_| sampler.sample(&logits)).collect();
        assert!(picks.contains(&7) && picks.contains(&300) && !picks.contains(&999));
    }
}
