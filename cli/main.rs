mod fetch;
mod models;
mod once;
mod opts;
mod run;
mod tui;

use models::Model;
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
    let result = fetch::locate(&opts.model)
        .and_then(|(model, _)| check_memory(model))
        .and_then(|()| match prompt.trim() {
            "" => run::run(&opts.model, opts.device, opts.context, opts.thinking),
            prompt => once::once(&opts.model, opts.device, opts.context, prompt, opts.stats),
        });
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// Refuses to go on, before any weights are downloaded, on a Mac with too
/// little memory to run `model`: its CPU and GPU share the memory, so the
/// model needs all of what it takes from it.
#[cfg(target_os = "macos")]
fn check_memory(model: &Model) -> Result<(), Box<dyn std::error::Error>> {
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
    if memory < model.memory {
        return Err(format!(
            "{} requires {} GiB or more DRAM on a Mac, and this one has {} GiB",
            model.name,
            model.memory >> 30,
            memory >> 30
        )
        .into());
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn check_memory(_model: &Model) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}
