mod fetch;
mod models;
mod once;
mod opts;
mod run;
mod tui;

use std::env;

use argh::{EarlyExit, FromArgs};

use opts::{Device, Opts};

fn main() {
    let args: Vec<String> = env::args().collect();
    let command = args[0].rsplit('/').next().unwrap_or(&args[0]);
    let rest: Vec<&str> = args[1..].iter().map(String::as_str).collect();
    let opts = match Opts::from_args(&[command], &rest) {
        Ok(opts) => opts,
        Err(EarlyExit { output, status }) => {
            match status {
                Ok(()) => print!("{}", help(output, &rest)),
                Err(()) => eprint!("{output}"),
            }
            std::process::exit(status.map_or(1, |()| 0));
        }
    };
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

/// The help text, with the default context worked out for the device the
/// arguments name, if the model is downloaded and the device opens.
fn help(output: String, args: &[&str]) -> String {
    let device = match args.iter().position(|&arg| arg == "--device") {
        Some(i) if args.get(i + 1) == Some(&"cpu") => Device::Cpu,
        _ => Device::Gpu,
    };
    let Some(size) = models::default_size(device) else {
        return output;
    };
    let default = format!(
        "(default: {} on this machine, taking {:.2} GB of memory)",
        size.context,
        size.bytes as f64 / 1e9
    );
    // Put it in place of the description argh wrapped, wrapped the same way.
    let mut lines = Vec::new();
    let mut rest = output.lines().peekable();
    while let Some(line) = rest.next() {
        if !line.starts_with("  --context ") {
            lines.push(line.to_string());
            continue;
        }
        while rest.peek().is_some_and(|line| line.starts_with("    ")) {
            rest.next();
        }
        let indent = line.find("tokens").unwrap_or(20);
        let text = format!("tokens of conversation the model has room for {default}");
        let mut current = format!("{:indent$}", "  --context");
        for word in text.split(' ') {
            if current.len() > indent && current.len() + 1 + word.len() > 80 {
                lines.push(current);
                current = " ".repeat(indent);
            } else if current.len() > indent {
                current.push(' ');
            }
            current.push_str(word);
        }
        lines.push(current);
    }
    lines.join("\n") + "\n"
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
