#!/usr/bin/env python3
"""Runs dwim's GPU checks on an NVIDIA GPU in Modal, for a machine without
one: the CUDA kernels against the CPU reference, their timings, the real
model, and the difftest.

    modal run scripts/modal-gpu.py                     # the kernel tests
    modal run scripts/modal-gpu.py --what speed        # the kernels' timings
    modal run scripts/modal-gpu.py --what model        # the model answers a prompt, and its speed
    modal run scripts/modal-gpu.py --what difftest
    modal run scripts/modal-gpu.py --command "cargo test gpu::cuda::tests::rope"
    modal run scripts/modal-gpu.py --gpu H100          # rather than the first free of a few

The checkout goes up with the run, less what git ignores, and is built there; the
build, cargo's registry, and the model's weights are kept in the volume
`dwim-gpu` between runs, so the first run is slow and the rest are not. The
output is kept there too, as `logs/<what>.txt`, which `modal volume get
dwim-gpu logs/tests.txt -` prints, for a run left to itself. Modal keeps an
ephemeral app only a few minutes past its client, so the client stays up
for the run.
"""

import re
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
# The pinned toolchain and the files git ignores, read here; the script is
# imported again in the container, where the checkout is elsewhere and the
# image already built.
TOOLCHAIN = (
    re.search(r'channel\s*=\s*"([^"]+)"', (ROOT / "rust-toolchain.toml").read_text()).group(1)
    if modal.is_local()
    else "stable"
)
IGNORED = (
    set(
        subprocess.run(
            ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "-z"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        )
        .stdout.decode()
        .split("\0")
    )
    if modal.is_local()
    else set()
)


def ignored(path: Path) -> bool:
    """Whether a file of the checkout, named relative to it, stays out of
    the run: git's own, and what `.gitignore` leaves out, such as the
    build. Git names a nested repository as one entry, so the file's
    directories are looked up too."""
    return path.parts[0] == ".git" or any(str(p) in IGNORED for p in [path, *path.parents])

app = modal.App("dwim-gpu")
volume = modal.Volume.from_name("dwim-gpu", create_if_missing=True)

# The toolkit's image has NVRTC, which compiles the kernels; the driver is
# the host's.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("curl", "build-essential", "git")
    .run_commands(
        "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"
        f" --default-toolchain {TOOLCHAIN} -c rustfmt -c clippy"
    )
    .env(
        {
            "PATH": "/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "CARGO_HOME": "/cache/cargo",
            "CARGO_TARGET_DIR": "/cache/target",
            # Where dwim keeps the model, and the states it saves.
            "HF_HUB_CACHE": "/cache/hf",
            "XDG_CACHE_HOME": "/cache/xdg",
            "RUST_BACKTRACE": "1",
        }
    )
    .add_local_dir(ROOT, "/src", ignore=ignored)
)

WHAT = {
    "tests": [
        "cargo test gpu::cuda -- --nocapture",
        "cargo test models::qwen35 -- --nocapture",
    ],
    "speed": [
        "cargo test --release gpu::cuda::tests -- --ignored --nocapture --test-threads 1",
    ],
    # The prompt is answered from an empty directory: a one-off prompt runs
    # the agent in auto mode, which sends it on from a reply that is not
    # the last word, and it would go on through the checkout.
    "model": [
        "cargo build --release --bin dwim && mkdir -p /tmp/empty && cd /tmp/empty"
        " && /cache/target/release/dwim --device cuda --context 16384 --stats"
        " 'What is the capital of France? Answer in one word.'",
        "cargo test --release real_model -- --ignored --nocapture",
    ],
    "difftest": [
        "cargo run --release --bin dwim -- --download",
        "cargo run --release --bin difftest",
    ],
}


# Whichever of these GPUs is free first, unless `--gpu` names one; the CPUs
# are for the build, which a run starts with, and the memory for the
# difftest's CPU side, which holds the 16 GB model in the host's.
GPUS = ["L4", "A10G", "L40S", "A100"]


@app.function(
    image=image, gpu=GPUS, cpu=8, memory=49152, volumes={"/cache": volume}, timeout=3 * 3600
)
def run(what: str, commands: list[str]) -> int:
    """Runs the commands in the checkout, one after another until one
    fails, keeping their output in the volume as `logs/<what>.txt`, and
    returns the last one's exit status."""
    Path("/cache/logs").mkdir(exist_ok=True)
    log = f"/cache/logs/{what}.txt"
    Path(log).write_text("")
    commands = [
        "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv",
        *commands,
    ]
    status = 0
    for command in commands:
        print(f"$ {command}", flush=True)
        status = subprocess.run(
            ["bash", "-o", "pipefail", "-c", f"({command}) 2>&1 | tee -a {log}"], cwd="/src"
        ).returncode
        if status != 0:
            print(f"exit status {status}", flush=True)
            break
    with open(log, "a") as f:
        f.write(f"exit status {status}\n")
    volume.commit()
    return status


@app.local_entrypoint()
def main(what: str = "tests", command: str = "", gpu: str = ""):
    commands = [command] if command else WHAT[what]
    function = run.with_options(gpu=gpu) if gpu else run
    status = function.remote(what if not command else "command", commands)
    if status != 0:
        sys.exit(status)
