//! Compiles the Vulkan compute kernels, written in WGSL, to SPIR-V.

use std::{env, fs, path::PathBuf};

use naga::{
    back::spv,
    front::wgsl,
    valid::{Capabilities, ValidationFlags, Validator},
};

fn main() {
    let kernels = PathBuf::from("vulkan/kernels");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-changed={}", kernels.display());

    let mut entries: Vec<_> = fs::read_dir(&kernels)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    entries.sort();
    for path in entries {
        if path.extension().is_none_or(|ext| ext != "wgsl") {
            continue;
        }
        println!("cargo:rerun-if-changed={}", path.display());
        let source = fs::read_to_string(&path).unwrap();
        let module = wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}: {}", path.display(), e.emit_to_string(&source)));
        // Every check naga has, uniformity included. Its uniformity analysis
        // rejects a derivative, a texture sample, or a call to a function
        // that contains a barrier under a condition that differs between
        // invocations; it does not look at a bare barrier, so it accepts a
        // barrier after a loop whose trip count differs per invocation, as
        // several kernels have, and would also accept one inside a divergent
        // `if`, which none has: the kernels are read for that by hand.
        let info = Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{}: {}", path.display(), e.emit_to_string(&source)));
        let options = spv::Options {
            lang_version: (1, 3),
            flags: spv::WriterFlags::empty(),
            // Every kernel writes its workgroup memory before it reads it,
            // and function-scope variables are zeroed by naga regardless.
            zero_initialize_workgroup_memory: spv::ZeroInitializeWorkgroupMemoryMode::None,
            // Every loop's bound is a constant, a push constant the host
            // checks, or a builtin that is at least one, so none needs the
            // counter this would add to each iteration.
            force_loop_bounding: false,
            ..Default::default()
        };
        let words = spv::write_vec(&module, &info, &options, None)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let name = path.file_stem().unwrap().to_str().unwrap();
        fs::write(out.join(format!("{name}.spv")), bytes).unwrap();
    }
}
