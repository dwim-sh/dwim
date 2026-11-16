//! Sockets between two processes on one host, named by a path in the file
//! system.

pub use std::os::unix::net::{UnixListener as Listener, UnixStream as Stream};
