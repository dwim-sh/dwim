#!/usr/bin/env python3
"""Runs `dwim` over SWE-bench Lite and scores what it changed.

For each instance: check the repository out at the commit before the fix,
give `dwim` the issue, and take whatever it changed as the patch. The patches
go to predictions.jsonl, and are then scored by the official SWE-bench
harness, which runs each project's tests in Docker.

    scripts/swebench.py --instance pallets__flask-4045
    scripts/swebench.py --limit 10
    scripts/swebench.py

The agent runs shell commands from an untrusted issue, in a checkout of an
untrusted repository, as you. Run this in a container or a virtual machine.
"""
import argparse, glob, json, os, pathlib, shutil, subprocess, sys, urllib.request

DATASET = "SWE-bench/SWE-bench_Lite"
ROWS = "https://datasets-server.huggingface.co/rows?dataset=SWE-bench%2FSWE-bench_Lite&config=default&split=test"
ROOT = pathlib.Path(__file__).resolve().parent.parent


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


def checkout(repo, commit, repos):
    """A clone of repo at commit, under repos/."""
    clone = repos / repo.replace("/", "__")
    if not clone.exists():
        url = f"https://github.com/{repo}.git"
        subprocess.run(["git", "clone", "--quiet", "--filter=blob:none", url, str(clone)], check=True)
    for command in (["git", "checkout", "--quiet", "--force", commit], ["git", "clean", "-qfdx"]):
        subprocess.run(command, cwd=clone, check=True)
    return clone


def solve(row, args, dwim, repos, work):
    """Runs `dwim` over one instance, and returns the patch it leaves behind."""
    clone = checkout(row["repo"], row["base_commit"], repos)
    log = work / f"{row['instance_id']}.log"
    try:
        with open(log, "w") as output:
            subprocess.run(
                [dwim, "--model", args.model, row["problem_statement"]],
                cwd=clone,
                stdout=output,
                stderr=output,
                timeout=args.timeout,
            )
    except subprocess.TimeoutExpired:
        print(f"  timed out after {args.timeout}s", file=sys.stderr)
    return subprocess.run(["git", "diff"], cwd=clone, capture_output=True, text=True).stdout


def run(args, predictions):
    """Runs `dwim` over the instances, writing a prediction for each."""
    rows = instances()
    if args.instance:
        rows = [row for row in rows if row["instance_id"] in args.instance]
        missing = set(args.instance) - {row["instance_id"] for row in rows}
        if missing:
            sys.exit(f"no such instance: {', '.join(sorted(missing))}")
    if args.limit is not None:
        rows = rows[: args.limit]

    dwim = binary(args.dwim)
    work = pathlib.Path(args.work).resolve()
    repos = work / "repos"
    repos.mkdir(parents=True, exist_ok=True)
    with open(predictions, "w") as out:
        for i, row in enumerate(rows, 1):
            print(f"[{i}/{len(rows)}] {row['instance_id']}", file=sys.stderr)
            patch = solve(row, args, dwim, repos, work)
            print(f"  {len(patch)} bytes of patch, log in {work}", file=sys.stderr)
            prediction = {
                "instance_id": row["instance_id"],
                "model_name_or_path": f"dwim-{args.model}",
                "model_patch": patch,
            }
            out.write(json.dumps(prediction) + "\n")
    return len(rows)


def score(args, predictions):
    """Scores the predictions with the official harness, in Docker."""
    if shutil.which("docker") is None:
        sys.exit("docker is needed to score the patches: install it, or pass --no-score")
    if subprocess.run([sys.executable, "-c", "import swebench"], capture_output=True).returncode:
        sys.exit(f"the harness is needed to score the patches: {sys.executable} -m pip install swebench")
    print("scoring the patches, which runs each project's tests in Docker", file=sys.stderr)
    subprocess.run(
        [
            sys.executable, "-m", "swebench.harness.run_evaluation",
            "--dataset_name", DATASET,
            "--predictions_path", str(predictions),
            "--max_workers", str(args.workers),
            "--run_id", args.run_id,
        ],
        check=True,
    )
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
    parser.add_argument("--model", default="bonsai-2-27b", help="model for `dwim` to run (default: %(default)s)")
    parser.add_argument("--timeout", type=int, default=600, help="seconds per instance (default: %(default)s)")
    parser.add_argument("--work", default="swebench-work", help="where clones and logs go (default: %(default)s)")
    parser.add_argument("--out", default="predictions.jsonl", help="where the patches go (default: %(default)s)")
    parser.add_argument("--dwim", help="dwim binary to run (default: build target/release/dwim)")
    parser.add_argument("--no-score", action="store_true", help="write the patches without scoring them")
    parser.add_argument("--score-only", action="store_true", help="score patches written by an earlier run")
    parser.add_argument("--run-id", default="dwim", help="name for this run in the harness (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=4, help="instances scored at once (default: %(default)s)")
    args = parser.parse_args()

    predictions = pathlib.Path(args.out).resolve()
    if not args.score_only:
        run(args, predictions)
    if not args.no_score:
        score(args, predictions)


if __name__ == "__main__":
    main()
