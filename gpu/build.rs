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
        // Barriers sit after loops whose trip counts differ per thread, which
        // the uniformity analysis is stricter about than the hardware.
        let flags = ValidationFlags::all() - ValidationFlags::CONTROL_FLOW_UNIFORMITY;
        let info = Validator::new(flags, Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{}: {}", path.display(), e.emit_to_string(&source)));
        let options = spv::Options {
            lang_version: (1, 3),
            flags: spv::WriterFlags::empty(),
            zero_initialize_workgroup_memory: spv::ZeroInitializeWorkgroupMemoryMode::None,
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
