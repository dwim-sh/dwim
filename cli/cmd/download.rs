//! Fetching the model's weights into the cache: `dwim --download`.

use std::error::Error;

use crate::{
    session::{self, Settings},
    stderr::Stderr,
};

/// Downloads the model `settings` name if it is not in the cache yet, and
/// prints the path of its weights.
pub fn run(settings: &Settings) -> Result<(), Box<dyn Error>> {
    let mut stderr = Stderr::new();
    let dir = session::fetch_model(settings, |loading| stderr.report(&loading))?;
    if let Some(weights) = settings.model.weights() {
        println!("{}", weights.path(&dir).display());
    }
    Ok(())
}
