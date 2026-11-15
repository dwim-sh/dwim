//! A dwim server, which has the model on this machine: sessions on it,
//! held over its socket.

mod remote;

pub use remote::{Connection, Remote};
