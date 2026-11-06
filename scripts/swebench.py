#!/usr/bin/env python3
"""Runs `dwim` over SWE-bench Lite and scores what it changed.

For each instance: take the repository as the instance's evaluation image
has it, at the commit before the fix with the project and what it needs
installed; give `dwim` the issue, with its shell commands running in a
container of that image; take whatever it changed as the patch; and score
the patch with the official SWE-bench harness, which runs the project's
tests in the same image. The patches go to predictions.jsonl as they are
made, and the harness's report is written at the end. The model is what
takes the time, so the next instance's image is pulled while the agent
works on this one, and the last one's patch is scored behind it.

    scripts/swebench.py --instance pallets__flask-4045
    scripts/swebench.py --limit 10
    scripts/swebench.py

The agent's commands run in the container, but the model runs here, and
its read and edit tools work on the checkout as you, on an untrusted issue.
Run this in a virtual machine rather than on a machine you care about.
"""
import argparse, glob, json, os, pathlib, shutil, stat, subprocess, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

DATASET = "SWE-bench/SWE-bench_Lite"
ROWS = "https://datasets-server.huggingface.co/rows?dataset=SWE-bench%2FSWE-bench_Lite&config=default&split=test"
ROOT = pathlib.Path(__file__).resolve().parent.parent

# The evaluation images only exist for x86_64; an arm64 Docker runs them
# under Rosetta or QEMU, if asked for that platform.
PLATFORM = "linux/amd64"

# What `dwim --shell` runs each command through: `docker exec` in the
# instance's container, in the same directory, which is mounted there at
# the same path, with the project's conda environment active.
SHELL = """#!/bin/sh
exec docker exec -i -w "$(pwd -P)" {container} bash -c \
'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && exec bash -c "$1"' bash "$2"
"""

# Runs the harness with two changes. An arm64 Docker refuses to pull an
# x86_64 image unless asked for that platform, so ask. And each instance's
# image is about 2 GB of its own, 600 GB over the 300, so remove it once
# the instance is scored.
HARNESS = """
import runpy, sys
import docker.models.images as images
import swebench.harness.docker_utils as docker_utils

pull = images.ImageCollection.pull
def pull_amd64(self, repository, tag=None, all_tags=False, **kwargs):
    kwargs.setdefault("platform", "linux/amd64")
    return pull(self, repository, tag, all_tags, **kwargs)
images.ImageCollection.pull = pull_amd64

cleanup_container = docker_utils.cleanup_container
def cleanup_container_and_image(client, container, logger):
    image = container.image.id if container else None
    cleanup_container(client, container, logger)
    if image:
        try:
            client.images.remove(image, force=True)
        except Exception as e:
            print(f"could not remove image {image}: {e}", file=sys.stderr)
docker_utils.cleanup_container = cleanup_container_and_image

sys.argv[0] = "run_evaluation"
runpy.run_module("swebench.harness.run_evaluation", run_name="__main__")
"""


def instances():
    """Every instance of the dataset, from the Hugging Face datasets server."""
    rows = []
    for offset in range(0, 300, 100):
        with urllib.request.urlopen(f"{ROWS}&offset={offset}&length=100") as response:
            rows += [row["row"] for row in json.load(response)["rows"]]
    return rows


def binary(path):
    """The `dwim` binary, built if it isn't already."""
    if path:
        return str(pathlib.Path(path).resolve())
    built = ROOT / "target" / "release" / "dwim"
    print("building dwim", file=sys.stderr)
    subprocess.run(["cargo", "build", "--release"], cwd=ROOT, check=True)
    return str(built)


def docker(*args, check=True):
    """Runs a docker command, quietly, and returns what it printed."""
    return subprocess.run(["docker", *args], check=check, capture_output=True, text=True).stdout


def prepare(row, work):
    """What the agent needs before it starts on an instance: its image,
    pulled, and its checkout, copied out of the image."""
    image = row["image"]
    if not docker("image", "inspect", "--format", "{{.Id}}", image, check=False):
        subprocess.run(["docker", "pull", "--quiet", "--platform", PLATFORM, image], check=True, stdout=subprocess.DEVNULL)
    return checkout(row, work)


def checkout(row, work):
    """The repository as the instance's image has it at /testbed, copied
    out to a directory of the instance's own under work/."""
    image = row["image"]
    clone = work / "checkouts" / row["instance_id"]
    shutil.rmtree(clone, ignore_errors=True)
    clone.parent.mkdir(parents=True, exist_ok=True)
    name = f"dwim.{row['instance_id']}.copy"
    docker("rm", "-f", name)
    docker("create", "--platform", PLATFORM, "--name", name, image, "true")
    docker("cp", f"{name}:/testbed", str(clone))
    docker("rm", name)
    return clone


def container(row, clone):
    """A container of the instance's image, with `clone` mounted at /testbed,
    where the project is installed from, and at its own path, where the
    model's commands name it; returns its name."""
    name = f"dwim.{row['instance_id']}"
    docker("rm", "-f", name)
    docker(
        "run", "--detach", "--platform", PLATFORM, "--name", name,
        "--volume", f"{clone}:/testbed", "--volume", f"{clone}:{clone}", "--workdir", str(clone),
        row["image"], "sleep", "infinity",
    )
    return name


def solve(row, args, dwim, work, clone):
    """Runs `dwim` over one instance, in its prepared checkout `clone`, and
    returns the patch it leaves behind."""
    name = container(row, clone)
    shell = work / f"{row['instance_id']}.sh"
    shell.write_text(SHELL.format(container=name))
    shell.chmod(shell.stat().st_mode | stat.S_IXUSR)
    log = work / f"{row['instance_id']}.log"
    transcript = work / f"{row['instance_id']}.jsonl"
    try:
        with open(log, "w") as output:
            subprocess.run(
                [
                    dwim, "--model", args.model, "--shell", str(shell), "--transcript", str(transcript),
                    row["problem_statement"],
                ],
                cwd=clone,
                stdout=output,
                stderr=output,
                timeout=args.timeout,
            )
    except subprocess.TimeoutExpired:
        print(f"  timed out after {args.timeout}s", file=sys.stderr)
    finally:
        docker("rm", "-f", name, check=False)
    patch = subprocess.run(["git", "diff"], cwd=clone, capture_output=True, text=True).stdout
    shutil.rmtree(clone, ignore_errors=True)
    return patch


def run(args, predictions):
    """Runs `dwim` over the instances, writing a prediction for each, and
    scoring it unless told not to. The model is the bottleneck, so one
    thread prepares the next instance while the agent works on this one,
    and another scores the patches made, one at a time, behind it."""
    rows = instances()
    if args.instance:
        rows = [row for row in rows if row["instance_id"] in args.instance]
        missing = set(args.instance) - {row["instance_id"] for row in rows}
        if missing:
            sys.exit(f"no such instance: {', '.join(sorted(missing))}")
    if args.limit is not None:
        rows = rows[: args.limit]

    dwim = binary(args.dwim)
    if not args.no_score:
        scorable()
    work = pathlib.Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    preparer = ThreadPoolExecutor(max_workers=1)
    scorer = ThreadPoolExecutor(max_workers=1)
    prepared = preparer.submit(prepare, rows[0], work) if rows else None
    with open(predictions, "w") as out:
        for i, row in enumerate(rows, 1):
            print(f"[{i}/{len(rows)}] {row['instance_id']}", file=sys.stderr)
            try:
                clone = prepared.result()
            except subprocess.CalledProcessError as e:
                print(f"  could not prepare {row['instance_id']}: {e}", file=sys.stderr)
                clone = None
            if i < len(rows):
                prepared = preparer.submit(prepare, rows[i], work)
            patch = solve(row, args, dwim, work, clone) if clone else ""
            print(f"  {len(patch)} bytes of patch, log and transcript in {work}", file=sys.stderr)
            prediction = {
                "instance_id": row["instance_id"],
                "model_name_or_path": f"dwim-{args.model}",
                "model_patch": patch,
            }
            out.write(json.dumps(prediction) + "\n")
            out.flush()
            if patch and not args.no_score:
                scorer.submit(score_one, args, predictions, work, row["instance_id"])
    preparer.shutdown()
    scorer.shutdown()
    if not args.no_score:
        # The harness reports on the instances of each run: write the report
        # over them all, from the test outputs kept.
        harness(args, predictions, "--rewrite_reports", "True", log=work / "report.log")
        report(args)
    return len(rows)


def score_one(args, predictions, work, instance_id):
    """Scores one instance's patch with the official harness, keeping what
    the harness printed in work/, and prints the verdict."""
    harness(args, predictions, "--instance_ids", instance_id, "--max_workers", "1", log=work / f"{instance_id}.score.log")
    print(f"  {instance_id}: {verdict(args, instance_id)}", file=sys.stderr)


def scorable():
    """Exits with what to install if the patches can't be scored here."""
    if shutil.which("docker") is None:
        sys.exit("docker is needed to run the agent's commands and score its patches: install it")
    if subprocess.run([sys.executable, "-I", "-c", "import swebench.harness"], capture_output=True).returncode:
        sys.exit(f"the harness is needed to score the patches: {sys.executable} -m pip install swebench")


def harness(args, predictions, *more, log=None):
    """Runs the official harness over the predictions, with `more` arguments,
    writing what it prints to `log` if given."""
    with open(log, "w") if log else open(os.devnull, "w") as output:
        subprocess.run(
            [
                sys.executable, "-I", "-c", HARNESS,
                "--dataset_name", DATASET,
                "--predictions_path", str(predictions),
                "--max_workers", str(args.workers),
                "--run_id", args.run_id,
                *more,
            ],
            check=False,
            stdout=output if log else None,
            stderr=output if log else None,
        )


def verdict(args, instance_id):
    """What the harness made of an instance's patch."""
    paths = glob.glob(f"logs/run_evaluation/{args.run_id}/*/{instance_id}/report.json")
    if not paths:
        return "not scored"
    with open(paths[0]) as file:
        return "resolved" if json.load(file)[instance_id]["resolved"] else "not resolved"


def score(args, predictions):
    """Scores the predictions of an earlier run with the official harness."""
    scorable()
    print("scoring the patches, which runs each project's tests in Docker", file=sys.stderr)
    harness(args, predictions)
    report(args)


def report(args):
    """Prints the counts from the harness's report, and where it is."""
    reports = glob.glob(f"*.{args.run_id}.json")
    if not reports:
        return
    newest = max(reports, key=os.path.getmtime)
    with open(newest) as file:
        counts = json.load(file)
    for name in ("total_instances", "submitted_instances", "completed_instances", "resolved_instances",
                 "unresolved_instances", "empty_patch_instances", "error_instances"):
        if name in counts:
            print(f"{name.replace('_', ' '):24} {counts[name]}")
    print(f"full report in {newest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instance", action="append", help="instance to run, repeatable (default: all 300)")
    parser.add_argument("--limit", type=int, help="run only the first so many instances")
    parser.add_argument("--model", default="qwen3.8-27b", help="model for `dwim` to run (default: %(default)s)")
    parser.add_argument("--timeout", type=int, default=600, help="seconds per instance (default: %(default)s)")
    parser.add_argument("--work", default="swebench-work", help="where checkouts and logs go (default: %(default)s)")
    parser.add_argument("--out", default="predictions.jsonl", help="where the patches go (default: %(default)s)")
    parser.add_argument("--dwim", help="dwim binary to run (default: build target/release/dwim)")
    parser.add_argument("--no-score", action="store_true", help="write the patches without scoring them")
    parser.add_argument("--score-only", action="store_true", help="score patches written by an earlier run")
    parser.add_argument("--run-id", default="dwim", help="name for this run in the harness (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=4, help="instances scored at once with --score-only (default: %(default)s)")
    args = parser.parse_args()

    predictions = pathlib.Path(args.out).resolve()
    if args.score_only:
        score(args, predictions)
    else:
        run(args, predictions)


if __name__ == "__main__":
    main()
