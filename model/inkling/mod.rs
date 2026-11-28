//! Inkling 2 Mini, Thinking Machines' looped mixture-of-experts model:
//! what is known of it that is its own. Its chat format writes a reply as
//! blocks, of thought, of text and of tool calls, and calls tools in JSON.

pub mod format;
mod tokenizer;

use std::sync::Arc;

use crate::model::{File, Format, Result, Sampling, Weights, gguf::Metadata};

pub use format::Blocks;
pub use tokenizer::tokenizer;

/// The file holding Inkling 2 Mini.
const FILE: &str = "Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file";

/// The file holding its drafter.
const DRAFTER_FILE: &str = "Inkling-2-Mini-DFlash2-Q8_0.gguf";

/// Where Inkling 2 Mini's drafter comes from: Inco AI's DFlash 2 for the
/// model, `incoai/Inkling-2-Mini-DFlash2`, a block drafter of five layers,
/// as llama.cpp's converter writes it, `dflash` to llama.cpp, with its
/// matrices as Q8_0. The file carries the norm of the model's embeddings
/// and the model's vocabulary, and nothing else of the model's.
const DRAFTER: Weights = Weights {
    repo: "na0x2c6/Inkling-2-Mini-DFlash2-GGUF",
    revision: "e12c9c89cf07a2e063a7a95cc753165c662976d1",
    file: DRAFTER_FILE,
    drafter: None,
    files: &[File {
        name: DRAFTER_FILE,
        digest: "c2179daa877bbdca5d8ca87c8562296578e72840aa3b5eba60ac2fb54aef2d42",
    }],
};

/// Where Inkling 2 Mini comes from: Unsloth's conversion of
/// `thinkingmachines/Inkling-2-Mini`, in the layout mainline llama.cpp
/// reads, `inkling2_loop`, which names the looped layers once and says how
/// often they run; the files at the repository's root lay the same
/// tensors out as Unsloth's own llama.cpp reads them. Its
/// `UD-Q4_K_XL` file holds the experts of the first twenty layers in
/// NVFP4, the format the model was trained in, with a scale for each
/// expert, and those of the last four as Q8_0; the attention matrices,
/// the embeddings and the output projection are Q8_0, and the rest f32 or
/// bf16.
pub const INKLING_MINI: Weights = Weights {
    repo: "unsloth/Inkling-2-Mini-GGUF",
    revision: "ea0c79b2ecd73669e774c8519b79bfabfddbc027",
    file: FILE,
    drafter: Some(&DRAFTER),
    files: &[File {
        name: FILE,
        digest: "e64276656f072e9b2197a61723667cbdb71da66a716ffd966e572c21088f3cd3",
    }],
};

/// How Inkling's tokens are sampled where no file of it says, at a host:
/// at a temperature of one with no token cut, as its file recommends.
pub const SAMPLING: Sampling = Sampling {
    temperature: 1.0,
    top_k: 0,
    top_p: 1.0,
    min_p: 0.0,
    seed: 0,
};

/// The chat format of Inkling 2 Mini, over the vocabulary its GGUF file
/// carries.
pub fn format(gguf: &Metadata) -> Result<Arc<dyn Format>> {
    Ok(Arc::new(Blocks::new(tokenizer(gguf)?)?))
}
