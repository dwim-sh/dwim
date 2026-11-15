use serde::{Deserialize, Serialize};

/// How a model's tokens are sampled: from the `top_k` most likely tokens,
/// then the smallest set of those whose probability reaches `top_p`,
/// leaving out any less than `min_p` times as likely as the most likely,
/// at `temperature`, with random numbers that started from `seed`.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u64,
}
