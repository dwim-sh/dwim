//! The Qwen3.8 models: what is known of them that is theirs alone.

use crate::model::{File, Weights};

/// The file holding Qwen3.8-27B.
const FILE: &str = "Qwen3.8-27B-Q4_0.gguf";

/// Where Qwen3.8-27B comes from: Qwen ships no GGUF of its own, so
/// bartowski's conversion of `Qwen/Qwen3.8-27B`, quantized with an
/// importance matrix. Its Q4_0 file holds most matrices in 4-bit blocks of
/// 32 with an f16 scale, the first layers' down projections as Q4_1, the
/// linear attention's output projections as Q8_0, and the output and a few
/// query projections as Q6_K; the norms and the small linear attention
/// tensors are f32.
pub const QWEN_27B: Weights = Weights {
    repo: "bartowski/Qwen3.8-27B-GGUF",
    revision: "87410e7b9a5c21e6ca70c0c9cc5237de909fe621",
    file: FILE,
    files: &[File {
        name: FILE,
        digest: "5eb1de39e9b42aa7f99f1a4fe149d1952cc8103c698cf380b15a43d47e72b7c7",
    }],
};
