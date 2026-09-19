mod fetch;
mod models;
mod once;
mod opts;
mod run;
mod tui;

use opts::Opts;

fn main() {
    let opts: Opts = argh::from_env();
    if opts.version {
        println!("dwim {}", env!("CARGO_PKG_VERSION"));
        return;
    }

    // A prompt on the command line is a one-off: answer it and exit, with
    // nothing on standard output but the reply. Without one, open the shell.
    let prompt = opts.prompt.join(" ");
    let result = check_memory().and_then(|()| match prompt.trim() {
        "" => run::run(&opts.model, opts.device, opts.context, opts.thinking),
        prompt => once::once(&opts.model, opts.device, opts.context, prompt, opts.stats),
    });
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// Memory a Mac needs to run the model: its CPU and GPU share it, and the
/// weights alone take 6 GB of it.
#[cfg(target_os = "macos")]
const MIN_MEMORY: u64 = 16 << 30;

/// Refuses to go on, before any weights are downloaded, on a Mac with too
/// little memory to run the model.
#[cfg(target_os = "macos")]
fn check_memory() -> Result<(), Box<dyn std::error::Error>> {
    let mut memory: u64 = 0;
    let mut len = size_of::<u64>();
    let ret = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&raw mut memory).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret != 0 {
        return Err(format!(
            "cannot determine the memory size: {}",
            std::io::Error::last_os_error()
        )
        .into());
    }
    if memory < MIN_MEMORY {
        return Err(format!(
            "dwim requires 16 GiB or more DRAM on a Mac, and this one has {} GiB",
            memory >> 30
        )
        .into());
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn check_memory() -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}
