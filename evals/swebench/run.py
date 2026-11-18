#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["swebench==5.0.2"]
# ///
"""Runs `dwim` over a SWE-bench dataset and scores what it changed.

For each instance: take the repository as the instance's evaluation image
has it, at the commit before the fix with the project and what it needs
installed; give `dwim` the issue, with its shell commands running in a
container of that image that has no network; take whatever it changed as
the patch; and score the patch with the official SWE-bench harness, which
runs the project's tests in a fresh container of the same image. The model
is what takes the time, so the next instance's image is pulled while the
agent works on this one, and the last one's patch is scored behind it.

    evals/swebench/run.py --instance django__django-11790
    evals/swebench/run.py --limit 10
    evals/swebench/run.py
    evals/swebench/run.py --dataset lite -- --temperature 0
    evals/swebench/run.py --gold
    evals/swebench/run.py --pull

What follows `--` is given to `dwim`. `--gold` scores the dataset's own
fixes instead of the agent's, which all resolve where the images and the
harness work. `--pull` only pulls the instances' images, one each of a
few gigabytes, which is where a first run's time goes; they are kept, so
it is paid once.

A run is a directory under evals/runs/, named for the dataset and the
commit of `dwim`: manifest.json says what was run, and each instance has a
directory with its result.json, patch.diff, the transcript and the logs.
The same command again takes the run up where it stopped: it leaves the
instances that are done, tries again those that failed for a reason that
was not the agent's, and scores the patches that are not scored.

The agent's commands run in the container, but the model runs here, and
its read and edit tools work on the checkout as you, on an untrusted issue.
Run this in a virtual machine rather than on a machine you care about.
"""
import argparse, datetime, glob, json, os, pathlib, platform, shutil, stat, subprocess, sys, time, urllib.parse, urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

ROOT = pathlib.Path(__file__).resolve().parents[2]

# The datasets, which share their harness.
DATASETS = {
    "verified": "SWE-bench/SWE-bench_Verified",
    "lite": "SWE-bench/SWE-bench_Lite",
}

# The parts of a dataset that go by a name of their own, each a dataset
# that says which instances they are. Verified Mini is 50 of Verified's
# 500, chosen to be as hard on the whole. Its own rows are without the
# image and the tests the harness reads, so the instances run are
# Verified's.
SUBSETS = {
    "verified-mini": ("verified", "MariusHobbhahn/swe-bench-verified-mini"),
}

# The evaluation images only exist for x86_64; an arm64 Docker runs them
# under Rosetta or QEMU, if asked for that platform.
PLATFORM = "linux/amd64"

# How the issue is put to the model. SWE-bench leaves that to the scaffold,
# and given the raw issue of a project it knows, the model takes it for the
# commit that closed it and spends its turns recalling that commit's tests
# and trying to fetch the project's history to reproduce them, rather than
# making the fix: on pallets__flask-4045 it had the change worked out in
# four minutes and had not made it after thirteen. So say what is wanted,
# as the leaderboard's scaffolds do.
TASK = """Fix the issue below in this checkout. Make the smallest change that fixes it, update the tests the change breaks, and run them. Don't try to recall or reproduce how the project fixed it upstream; make the fix from this code.

<issue>
{issue}
</issue>
"""

# What `dwim --shell` runs each command through: `docker exec` in the
# instance's container, in the same directory, which is mounted there at
# the same path, with the project's conda environment active.
SHELL = """#!/bin/sh
exec docker exec -i -w "$(pwd -P)" {container} bash -c \
'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && exec bash -c "$1"' bash "$2"
"""

# Runs the harness with one change: an arm64 Docker refuses to pull an
# x86_64 image unless asked for that platform, so ask.
HARNESS = """
import runpy, sys
import docker.models.images as images

pull = images.ImageCollection.pull
def pull_amd64(self, repository, tag=None, all_tags=False, **kwargs):
    kwargs.setdefault("platform", "linux/amd64")
    return pull(self, repository, tag, all_tags, **kwargs)
images.ImageCollection.pull = pull_amd64

sys.argv[0] = "run_evaluation"
runpy.run_module("swebench.harness.run_evaluation", run_name="__main__")
"""

# The name the harness files a run's logs and reports under. A run has a
# directory of its own, so they all take the same.
HARNESS_RUN = "score"

# Goes back to the start of the terminal's line and clears it.
CLEAR = "\r\033[K"

# How many characters wide the bar that shows how far along something is.
BAR = 30


def main():
    args = arguments()
    rows = chosen(args, instances(args.dataset))
    required()
    if args.pull:
        pull_all(rows, args.workers)
        return
    dwim = None if args.gold else binary(args.dwim)
    run = directory(args, dwim)
    print(f"run in {run}", file=sys.stderr)
    scored = set()
    if args.gold:
        gold(run, rows)
    else:
        scored = solve_all(args, run, dwim, rows)
    if not args.no_score:
        unscored = [
            row["instance_id"] for row in rows
            if row["instance_id"] not in scored and wants_scoring(result(run, row["instance_id"]))
        ]
        if unscored:
            pull_all([row for row in rows if row["instance_id"] in unscored], args.workers)
            print(f"scoring {len(unscored)} patches, {args.workers} at a time, which runs each project's tests in Docker", file=sys.stderr)
            score(args, run, unscored, args.workers)
    summary(run, rows)


def arguments():
    """The command line, with what follows `--` set apart for `dwim`."""
    argv, dwim_args = sys.argv[1:], []
    if "--" in argv:
        at = argv.index("--")
        argv, dwim_args = argv[:at], argv[at + 1:]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=[*SUBSETS, *DATASETS], default="verified-mini", help="dataset to run (default: %(default)s)")
    parser.add_argument("--instance", action="append", help="instance to run, repeatable (default: all of the dataset)")
    parser.add_argument("--limit", type=int, help="run only the first so many instances")
    parser.add_argument("--model", default="medium", help="model for `dwim` to run (default: %(default)s)")
    parser.add_argument("--timeout", type=int, default=600, help="seconds the agent has per instance (default: %(default)s)")
    parser.add_argument("--run-id", help="name of the run, to start or to take up (default: the dataset and dwim's commit)")
    parser.add_argument("--runs", default=str(ROOT / "evals" / "runs"), help="where runs are kept (default: %(default)s)")
    parser.add_argument("--dwim", help="dwim binary to run (default: build target/release/dwim)")
    parser.add_argument("--pull", action="store_true", help="pull the instances' images, which a first run spends most of its time on, and stop")
    parser.add_argument("--gold", action="store_true", help="score the dataset's own fixes, to check the images and the harness")
    parser.add_argument("--no-score", action="store_true", help="make the patches without scoring them")
    parser.add_argument("--workers", type=int, default=4, help="images pulled at once, and instances scored at once when none are being solved (default: %(default)s)")
    args = parser.parse_args(argv)
    args.dwim_args = dwim_args
    return args


def instances(name):
    """Every instance of the dataset or the part of one that goes by `name`."""
    if name in SUBSETS:
        whole, part = SUBSETS[name]
        ids = {row["instance_id"] for row in table(part)}
        return [row for row in table(DATASETS[whole]) if row["instance_id"] in ids]
    return table(DATASETS[name])


def table(dataset):
    """Every row of a dataset, from the Hugging Face datasets server."""
    url = f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(dataset, safe='')}&config=default&split=test"
    rows, total = [], None
    while total is None or len(rows) < total:
        with urllib.request.urlopen(f"{url}&offset={len(rows)}&length=100") as response:
            page = json.load(response)
        if not page["rows"]:
            break
        total = page["num_rows_total"]
        rows += [row["row"] for row in page["rows"]]
    return rows


def chosen(args, rows):
    """The instances of `rows` the command line asks for."""
    if args.instance:
        missing = set(args.instance) - {row["instance_id"] for row in rows}
        if missing:
            sys.exit(f"no such instance in {args.dataset}: {', '.join(sorted(missing))}")
        rows = [row for row in rows if row["instance_id"] in args.instance]
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def required():
    """Exits with what to install if the run can't be made here. The
    harness is not among it: uv, which runs this script, installs what the
    script's first lines ask for."""
    if shutil.which("docker") is None:
        sys.exit("docker is needed to run the agent's commands and score its patches: install it")


def binary(path):
    """The `dwim` binary, built if it isn't already."""
    if path:
        return str(pathlib.Path(path).resolve())
    print("building dwim", file=sys.stderr)
    subprocess.run(["cargo", "build", "--release"], cwd=ROOT, check=True)
    return str(ROOT / "target" / "release" / "dwim")


def directory(args, dwim):
    """The run's directory, with its manifest: made for a new run, and
    checked for one taken up, which has to be of the same dataset, by the
    same `dwim`, asked the same way, for its results to be of one thing."""
    now = manifest(args, dwim)
    agent = "gold"
    if dwim:
        agent = now["agent"]["commit"][:12] + ("-dirty" if now["agent"]["dirty"] else "")
    run = pathlib.Path(args.runs).resolve() / (args.run_id or f"{args.dataset}-{agent}")
    run.mkdir(parents=True, exist_ok=True)
    path = run / "manifest.json"
    if path.exists():
        was = json.loads(path.read_text())
        differs = [key for key in ("dataset", "revision", "agent") if was.get(key) != now[key]]
        if differs:
            sys.exit(f"{run} is a run with another {' and '.join(differs)}, as its manifest.json has it: give this one another --run-id")
    else:
        write(path, json.dumps(now, indent=2) + "\n")
    return run


def manifest(args, dwim):
    """What is run: the dataset as it is now, and the agent, which is
    `dwim` as built from this checkout and asked as the command line says,
    or the dataset's own fixes."""
    revision = {}
    for dataset in sources(args.dataset):
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{dataset}") as response:
            revision[dataset] = json.load(response)["sha"]
    agent = "gold"
    if dwim:
        agent = {
            "version": subprocess.run([dwim, "--version"], capture_output=True, text=True).stdout.strip(),
            "commit": git("rev-parse", "HEAD", cwd=ROOT).strip(),
            "dirty": bool(git("status", "--porcelain", cwd=ROOT).strip()),
            "model": args.model,
            "arguments": args.dwim_args,
            "timeout": args.timeout,
            "task": TASK,
        }
    return {
        "dataset": args.dataset,
        "revision": revision,
        "agent": agent,
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "host": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
    }


def sources(name):
    """The datasets on Hugging Face that `name` is read from, the one the
    harness scores against first."""
    if name in SUBSETS:
        whole, part = SUBSETS[name]
        return [DATASETS[whole], part]
    return [DATASETS[name]]


def gold(run, rows):
    """Records the dataset's own fix as each instance's patch."""
    for row in rows:
        if result(run, row["instance_id"]) is None:
            record(run, row["instance_id"], {"agent": "gold"}, row["patch"])


def solve_all(args, run, dwim, rows):
    """Runs `dwim` over the instances of `rows` that are not done, recording
    a result for each, and scoring it unless told not to. The model is the
    bottleneck, so one thread prepares the next instance while the agent
    works on this one, and another scores the patches made, one at a time,
    behind it. Returns the instances it had scored."""
    rows = [row for row in rows if not done(result(run, row["instance_id"]))]
    preparer = ThreadPoolExecutor(max_workers=1)
    scorer = ThreadPoolExecutor(max_workers=1)
    scored = set()
    prepared = preparer.submit(prepare, row=rows[0], run=run) if rows else None
    for i, row in enumerate(rows, 1):
        instance_id = row["instance_id"]
        print(f"[{i}/{len(rows)}] {instance_id}", file=sys.stderr)
        upcoming = preparer.submit(prepare, row=rows[i], run=run) if i < len(rows) else None
        try:
            outcome, patch = solve(row, args, dwim, run, prepared.result())
        except (subprocess.CalledProcessError, OSError) as e:
            # Not the agent's doing: a pull that failed, a full disk, a
            # Docker that went away. Say so, and leave it for the next go.
            outcome, patch = {"agent": "error", "error": failure(e)}, ""
        prepared = upcoming
        record(run, instance_id, outcome, patch)
        print(f"  {describe(outcome)}, {len(patch)} bytes of patch", file=sys.stderr)
        if patch and not args.no_score:
            scored.add(instance_id)
            scorer.submit(score_one, args, run, instance_id)
    preparer.shutdown()
    scorer.shutdown()
    return scored


def prepare(row, run):
    """What the agent needs before it starts on an instance: its image,
    pulled, and its checkout, copied out of the image."""
    if not here(row["image"]):
        pull(row)
    return checkout(row, run)


def pull_all(rows, workers):
    """Pulls the images of `rows` that are not here, so many at once, and
    says so as each arrives. Scoring many patches is nearly all this: an
    image is a few minutes of download and unpacking, and its tests are
    seconds. One that can't be pulled is left for the harness to report."""
    missing = [row for row in rows if not here(row["image"])]
    if not missing:
        print(f"all {len(rows)} images are here", file=sys.stderr)
        return
    print(f"pulling {len(missing)} of {len(rows)} images, {workers} at a time", file=sys.stderr)
    live = sys.stderr.isatty()
    started = time.monotonic()
    pulled = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pulls = {pool.submit(pull, row) for row in missing}
        while pulls:
            arrived, pulls = wait(pulls, timeout=1, return_when=FIRST_COMPLETED)
            for pulling in arrived:
                pulled += 1
                try:
                    instance_id, seconds = pulling.result()
                    said = f"pulled {instance_id} in {seconds}s"
                except subprocess.CalledProcessError as e:
                    said = failure(e)
                print(f"{CLEAR if live else ''}  [{pulled}/{len(missing)}] {said}", file=sys.stderr)
            if live and pulls:
                bar(pulled, len(missing), started, "pulled")
    if live:
        print(CLEAR, end="", file=sys.stderr, flush=True)


def pull(row):
    """Pulls an instance's image, and returns the instance and the seconds
    it took."""
    started = time.monotonic()
    docker("pull", "--quiet", "--platform", PLATFORM, row["image"])
    return row["instance_id"], round(time.monotonic() - started)


def here(image):
    """Whether Docker has an image."""
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


def checkout(row, run):
    """The repository as the instance's image has it at /testbed, copied
    out to the instance's directory in the run."""
    clone = run / row["instance_id"] / "checkout"
    shutil.rmtree(clone, ignore_errors=True)
    if clone.exists():
        raise OSError(f"could not remove {clone}")
    clone.parent.mkdir(parents=True, exist_ok=True)
    name = f"dwim.{row['instance_id']}.copy"
    docker("rm", "-f", name, check=False)
    docker("create", "--platform", PLATFORM, "--name", name, row["image"], "true")
    try:
        docker("cp", f"{name}:/testbed", str(clone))
    finally:
        docker("rm", name, check=False)
    return clone


def solve(row, args, dwim, run, clone):
    """Runs `dwim` over one instance, in its prepared checkout `clone`, and
    returns how that went and the patch it leaves behind. The patch is
    everything that differs from the checkout as the image had it: files
    added as well as changed, whether or not the agent committed them. An
    agent out of time is stopped, and what it had changed is its patch."""
    out = clone.parent
    before = snapshot(clone, out)
    name = container(row, clone)
    shell = out / "shell.sh"
    shell.write_text(SHELL.format(container=name))
    shell.chmod(shell.stat().st_mode | stat.S_IXUSR)
    started = time.monotonic()
    outcome = {}
    try:
        with open(out / "dwim.log", "w") as log:
            code = subprocess.run(
                [
                    dwim, "--model", args.model, "--shell", str(shell), "--transcript", str(out / "transcript.jsonl"),
                    *args.dwim_args, TASK.format(issue=row["problem_statement"]),
                ],
                cwd=clone,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                timeout=args.timeout,
            ).returncode
        outcome = {"agent": "failed", "exit_code": code} if code else {"agent": "finished"}
    except subprocess.TimeoutExpired:
        outcome = {"agent": "timed out"}
    finally:
        docker("rm", "-f", name, check=False)
    outcome["seconds"] = round(time.monotonic() - started)
    patch = git(
        "diff", "--no-ext-diff", "--no-color", "--binary", "--src-prefix=a/", "--dst-prefix=b/",
        before, snapshot(clone, out), cwd=clone,
    )
    shutil.rmtree(clone, ignore_errors=True)
    return outcome, patch


def snapshot(clone, out):
    """Every file of `clone` that git does not ignore, as a tree in its
    repository. The tree is made through an index kept in `out`, so that
    the agent finds the checkout's own index as the image left it."""
    index = out / "index"
    index.unlink(missing_ok=True)
    env = {**os.environ, "GIT_INDEX_FILE": str(index)}
    git("add", "--all", cwd=clone, env=env)
    tree = git("write-tree", cwd=clone, env=env).strip()
    index.unlink()
    return tree


def container(row, clone):
    """A container of the instance's image, with `clone` mounted at /testbed,
    where the project is installed from, and at its own path, where the
    model's commands name it; returns its name. It has no network: the
    project's history and its fix are a download away. Its commands run as
    you, not as root, so that what they write in the checkout is yours, for
    `dwim` to edit and for the patch to be read from; they have a home to
    write in, and a git that takes the checkout for its own whoever the
    mount says owns it."""
    name = f"dwim.{row['instance_id']}"
    docker("rm", "-f", name, check=False)
    docker(
        "run", "--detach", "--platform", PLATFORM, "--name", name, "--network", "none",
        "--user", f"{os.getuid()}:{os.getgid()}", "--env", "HOME=/tmp",
        "--env", "GIT_CONFIG_COUNT=1", "--env", "GIT_CONFIG_KEY_0=safe.directory", "--env", "GIT_CONFIG_VALUE_0=*",
        "--volume", f"{clone}:/testbed", "--volume", f"{clone}:{clone}", "--workdir", str(clone),
        row["image"], "sleep", "infinity",
    )
    return name


def failure(error):
    """What went wrong in a command, in a line."""
    if isinstance(error, subprocess.CalledProcessError):
        said = (error.stderr or "").strip().splitlines()
        return f"{' '.join(map(str, error.cmd[:3]))}: {said[-1] if said else f'exit {error.returncode}'}"
    return str(error)


def describe(outcome):
    """How the agent's go at an instance went, in a few words."""
    if outcome["agent"] == "error":
        return f"could not run: {outcome['error']}"
    if outcome["agent"] == "failed":
        return f"dwim exited with {outcome['exit_code']} after {outcome['seconds']}s"
    return f"{outcome['agent']} after {outcome['seconds']}s"


def score_one(args, run, instance_id):
    """Scores one instance's patch, behind the agent."""
    try:
        score(args, run, [instance_id], 1)
    except Exception as e:
        print(f"  {instance_id}: could not be scored: {e}", file=sys.stderr)


def score(args, run, instance_ids, workers):
    """Scores the patches of `instance_ids` with the official harness, so
    many at once, and records and prints the verdict on each as the harness
    reaches it: resolved, unresolved, or error where the harness ends
    without a report, as for a patch that does not apply. Scoring many, it
    draws how far it is as a bar that it keeps up to date, since the first
    verdict is minutes of pulling images away. The harness works in the
    run's directory, where its logs stay."""
    predictions = run / "predictions.jsonl"
    patches = sorted(path for path in run.glob("*/patch.diff") if path.stat().st_size)
    lines = [
        json.dumps({"instance_id": path.parent.name, "model_name_or_path": "dwim", "model_patch": path.read_text()})
        for path in patches
    ]
    write(predictions, "".join(line + "\n" for line in lines))
    waiting = list(instance_ids)
    live = len(instance_ids) > 1 and sys.stderr.isatty()
    started = time.monotonic()

    def settle(last):
        for instance_id in list(waiting):
            verdict = reported(run, instance_id) or ("error" if last else None)
            if verdict:
                waiting.remove(instance_id)
                amend(run, instance_id, {"verdict": verdict})
                at = f"[{len(instance_ids) - len(waiting)}/{len(instance_ids)}] " if len(instance_ids) > 1 else ""
                print(f"{CLEAR if live else ''}  {at}{instance_id}: {verdict}", file=sys.stderr)
        if live and not last:
            bar(len(instance_ids) - len(waiting), len(instance_ids), started, "scored")

    with open(run / "harness.log", "a") as log:
        harness = subprocess.Popen(
            [
                sys.executable, "-I", "-c", HARNESS,
                "--dataset_name", sources(args.dataset)[0],
                "--predictions_path", str(predictions),
                "--max_workers", str(workers),
                "--run_id", HARNESS_RUN,
                "--instance_ids", *instance_ids,
            ],
            cwd=run,
            stdout=log,
            stderr=log,
        )
        try:
            while harness.poll() is None:
                settle(last=False)
                time.sleep(1)
        finally:
            if live:
                print(CLEAR, end="", file=sys.stderr, flush=True)
            # Stopped half way, as by ctrl+c: the harness takes the
            # interrupt for its own and scores on, unseen, unless ended.
            if harness.poll() is None:
                harness.terminate()
            harness.wait()
    settle(last=True)


def bar(count, total, started, what):
    """Draws on the terminal's line how far along something is: a bar, the
    count, and the time it has taken so far."""
    minutes, seconds = divmod(round(time.monotonic() - started), 60)
    filled = BAR * count // total
    drawn = "\u2588" * filled + "\u2591" * (BAR - filled)
    print(f"{CLEAR}  {drawn} {count}/{total} {what}, {minutes}m{seconds:02}s", end="", file=sys.stderr, flush=True)


def reported(run, instance_id):
    """The harness's verdict on an instance, once it has written one."""
    reports = glob.glob(str(run / "logs" / "run_evaluation" / HARNESS_RUN / "*" / instance_id / "report.json"))
    try:
        with open(reports[0]) as file:
            return "resolved" if json.load(file)[instance_id]["resolved"] else "unresolved"
    except (IndexError, ValueError):
        # None yet, or one the harness is in the middle of writing.
        return None


def summary(run, rows):
    """Prints what became of the instances: the verdicts first, of which
    only the resolved count for anything, and then what stands beside
    them."""
    results = [result(run, row["instance_id"]) or {} for row in rows]
    def count(key, value):
        return sum(1 for r in results if r.get(key) == value)
    empty = sum(1 for r in results if done(r) and not r["patch_bytes"])
    unscored = sum(1 for r in results if wants_scoring(r) and not r.get("verdict"))
    lines = [
        ("resolved", f"{count('verdict', 'resolved')} of {len(rows)}"),
        ("unresolved", count("verdict", "unresolved")),
        ("empty patch", empty),
        ("not scored", unscored),
        ("harness error", count("verdict", "error")),
        ("could not run", len(results) - sum(1 for r in results if done(r))),
        ("timed out", count("agent", "timed out")),
        ("dwim failed", count("agent", "failed")),
    ]
    for name, value in lines:
        if value or name == "resolved":
            print(f"{name:14} {value}")
    print(f"results in {run}")


def done(result):
    """Whether an instance has had its go: the agent ran on it, however
    that ended."""
    return result is not None and result["agent"] != "error"


def wants_scoring(result):
    """Whether an instance has a patch without a verdict on it, or with one
    that says only that the harness failed."""
    return done(result) and result["patch_bytes"] > 0 and result.get("verdict") not in ("resolved", "unresolved")


def result(run, instance_id):
    """What is recorded of an instance, if anything is."""
    try:
        return json.loads((run / instance_id / "result.json").read_text())
    except FileNotFoundError:
        return None


def record(run, instance_id, outcome, patch):
    """Records how the agent's go at an instance went, and its patch."""
    out = run / instance_id
    out.mkdir(parents=True, exist_ok=True)
    write(out / "patch.diff", patch)
    write(out / "result.json", json.dumps({"instance_id": instance_id, **outcome, "patch_bytes": len(patch)}, indent=2) + "\n")


def amend(run, instance_id, fields):
    """Adds `fields` to what is recorded of an instance."""
    write(run / instance_id / "result.json", json.dumps({**result(run, instance_id), **fields}, indent=2) + "\n")


def write(path, text):
    """Writes a file whole: a run that is killed leaves the file as it
    was or as it is to be, and not half of it."""
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text)
    partial.replace(path)


def git(*args, cwd, env=None):
    """Runs a git command in `cwd`, and returns what it printed."""
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True, errors="replace").stdout


def docker(*args, check=True):
    """Runs a docker command, quietly, and returns what it printed."""
    return subprocess.run(["docker", *args], check=check, capture_output=True, text=True).stdout


if __name__ == "__main__":
    main()
