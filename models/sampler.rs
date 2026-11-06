/// Picks the next token from a model's logits: from the `top_k` most likely
/// tokens, then the smallest set of those whose probability reaches `top_p`,
/// leaving out any less than `min_p` times as likely as the most likely, at
/// `temperature`. A temperature of zero always picks the most likely token.
pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    /// What the random numbers it samples with started from.
    pub seed: u64,
    rng: Rng,
}

impl Sampler {
    pub fn new(temperature: f32, top_k: usize, top_p: f32, min_p: f32, seed: u64) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            min_p,
            seed,
            rng: Rng(seed.max(1)),
        }
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        let distribution = self.distribution(logits, &[]);
        self.draw(&distribution)
    }

    /// The distribution the sampler picks from, given the logits, with
    /// `forbidden` tokens left out: the tokens that may be picked, each
    /// with its probability, in descending order, the probabilities
    /// summing to one. At a temperature of zero, the most likely token
    /// alone.
    pub fn distribution(&self, logits: &[f32], forbidden: &[u32]) -> Distribution {
        // The top k in one pass over the logits, kept in order: a vocabulary
        // is a quarter of a million entries, and k is twenty.
        let k = self.top_k.clamp(1, logits.len());
        let mut candidates: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
        for (token, &logit) in logits.iter().enumerate() {
            if (candidates.len() == k && logit <= candidates[k - 1].1)
                || forbidden.contains(&(token as u32))
            {
                continue;
            }
            let at = candidates.partition_point(|&(_, l)| l >= logit);
            candidates.insert(at, (token as u32, logit));
            candidates.truncate(k);
        }
        if self.temperature == 0.0 {
            return vec![(candidates[0].0, 1.0)];
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

        // Keep the most likely tokens up to a total probability of top_p,
        // and none less than min_p times as likely as the most likely: the
        // cut scales with how sure the model is, where top_p at a flat
        // distribution keeps a long tail of unlikely tokens. The most likely
        // is always kept.
        let floor = self.min_p * probs[0];
        let mut kept = 0.0;
        let mut n = 0;
        while n < probs.len() && kept < self.top_p && probs[n] >= floor {
            kept += probs[n];
            n += 1;
        }
        candidates[..n]
            .iter()
            .zip(&probs)
            .map(|(&(token, _), &p)| (token, p / kept))
            .collect()
    }

    /// Picks a token from a distribution.
    pub fn draw(&mut self, distribution: &Distribution) -> u32 {
        if distribution.len() == 1 {
            return distribution[0].0;
        }
        let r = self.rng.next();
        let mut acc = 0.0;
        for &(token, p) in distribution {
            acc += p;
            if r < acc {
                return token;
            }
        }
        distribution[distribution.len() - 1].0
    }

    /// A number uniform in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        self.rng.next()
    }
}

/// Tokens that may be picked, each with its probability, in descending
/// order; the probabilities sum to one.
pub type Distribution = Vec<(u32, f32)>;

/// How the next token is picked: with a sampler, leaving out tokens that
/// are forbidden at the point the conversation has reached.
pub struct Policy<'a> {
    pub sampler: &'a mut Sampler,
    pub forbidden: &'a [u32],
    /// The logits the last token was picked from, kept so that it can be
    /// picked again under another policy.
    pub logits: Option<Vec<f32>>,
}

impl Policy<'_> {
    /// Picks a token from the logits.
    pub fn pick(&mut self, logits: &[f32]) -> u32 {
        let distribution = self.sampler.distribution(logits, self.forbidden);
        self.logits = Some(logits.to_vec());
        self.sampler.draw(&distribution)
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
        let mut greedy = Sampler::new(0.0, 20, 0.95, 0.0, 1);
        assert_eq!(greedy.sample(&logits), 7);
        // At a low temperature the top token is all but certain, and with
        // top_p of nearly nothing it is the only one kept.
        let mut sampler = Sampler::new(1.0, 3, 0.01, 0.0, 1);
        for _ in 0..20 {
            assert_eq!(sampler.sample(&logits), 7);
        }
        // With k of two, the third never comes up.
        let mut sampler = Sampler::new(100.0, 2, 1.0, 0.0, 1);
        let picks: std::collections::HashSet<u32> =
            (0..200).map(|_| sampler.sample(&logits)).collect();
        assert!(picks.contains(&7) && picks.contains(&300) && !picks.contains(&999));
    }

    #[test]
    fn drops_tokens_far_less_likely_than_the_top() {
        let mut logits = vec![0.0; 1000];
        logits[7] = 5.0;
        logits[300] = 4.0;
        logits[999] = 0.0;
        // The second is e^-1 as likely as the first, the rest e^-5: a min_p
        // between keeps the first two and drops the rest, with top_p and
        // top_k wide open.
        let mut sampler = Sampler::new(1.0, 1000, 1.0, 0.1, 1);
        let picks: std::collections::HashSet<u32> =
            (0..200).map(|_| sampler.sample(&logits)).collect();
        assert!(picks.contains(&7) && picks.contains(&300) && picks.len() == 2);
        // A min_p of one keeps the most likely alone.
        let mut sampler = Sampler::new(1.0, 1000, 1.0, 1.0, 1);
        for _ in 0..20 {
            assert_eq!(sampler.sample(&logits), 7);
        }
    }
}
