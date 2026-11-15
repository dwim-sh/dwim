//! A dwim server, which has the model on this machine: reaching it, and
//! sessions on it, held over its socket.

mod launch;
mod remote;

pub use launch::{Options, connect, log_path, running, same_build, stop};
pub use remote::{Connection, Remote};
