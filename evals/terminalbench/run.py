#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["harbor==0.27.0"]
# ///
"""Runs `dwim` over Terminal-Bench with Harbor, its harness, and reports
how many tasks it solved.

A task of Terminal-Bench is an instruction and a container to carry it out
in, and the task's tests then check what the container has become. Harbor
builds or pulls each task's image, starts its container, has the agent
work in it, and runs the tests there. `dwim` runs in the container, where
its tools see what the tests will, as agent.py has it; the model runs on
this machine, in a `dwim` server this script starts and keeps running for
the run, which the container reaches through its socket.

    evals/terminalbench/run.py --task hello-world
    evals/terminalbench/run.py --limit 10
    evals/terminalbench/run.py
    evals/terminalbench/run.py --dataset terminal-bench@2.0
    evals/terminalbench/run.py --provider openrouter -- --model large

What follows `--` is given to `dwim`. The program run in the containers is
built for glibc 2.28 with cargo-zigbuild, so that the one file runs in the
tasks' images and as the server here; a session checks that the server is
of its own build. `--workers` tasks run at once, and the server has room
for that many sessions.

A run is a directory under evals/runs/, named for the dataset and the
commit of `dwim`: manifest.json says what was run, report.md what came of
it, server.log what the server said, and job/ is Harbor's, with a
directory for each task with its result.json, and in agent/ the
transcript and what `dwim` printed. A run over a whole dataset writes its
report to evals/results/ as well, to be kept in git. The same command
again takes the run up where it stopped.

The tasks run as root in their containers, with the network, and some
are built from Dockerfiles that fetch what they need: run this in a
virtual machine rather than on a machine you care about.
"""
import argparse, datetime, json, os, pathlib, platform, shutil, subprocess, sys, threading, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent

# The dataset run where none is asked for: Terminal-Bench as tbench.ai has
# it now.
DATASET = "terminal-bench/terminal-bench@4.1.0"

# The oldest glibc the program is built for: Debian 10's and Ubuntu 20.04's
# are newer, and the tasks' images are of those or later.
GLIBC = "2.28"

# Where the program and the server's socket's directory are mounted in a
# task's container, as agent.py has them.
PROGRAM = "/opt/dwim/dwim"
SOCKETS = "/run/dwim"

# How long the server has to start listening, which it does before it
# loads the model.
LISTENING = 30

# The providers `dwim` can be asked to use, as its `--provider` names them,
# and the variable each host's key is in, as agent.py gives it to `dwim`.
PROVIDERS = ("local", "openrouter", "vercel", "cerebras")
KEYS = {"openrouter": "OPENROUTER_API_KEY", "vercel": "AI_GATEWAY_API_KEY", "cerebras": "CEREBRAS_API_KEY"}


def main():
    args = arguments()
    required(args)
    if args.provider in KEYS:
        key(args.provider)
    binary = program(args)
    run = directory(args, binary)
    print(f"run in {run}", file=sys.stderr)
    server = Server(binary, run / "server.log", args.workers) if args.provider == "local" else None
    try:
        if server:
            server.start()
        harbor(args, binary, run, server)
    finally:
        if server:
            server.stop()
    report(run, keep=not (args.task or args.limit))


def arguments():
    argv = sys.argv[1:]
    dwim_args = []
    if "--" in argv:
        at = argv.index("--")
        argv, dwim_args = argv[:at], argv[at + 1:]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DATASET, help="dataset to run, as Harbor names it (default: %(default)s)")
    parser.add_argument("--task", action="append", help="task to run, or a glob of tasks, repeatable (default: all of the dataset)")
    parser.add_argument("--limit", type=int, help="run only so many tasks")
    parser.add_argument("--provider", choices=PROVIDERS, default="local", help="where the model runs: on this machine, or at a host with the key in the environment (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=2, help="tasks run at once, and sessions the server has room for (default: %(default)s)")
    parser.add_argument("--run-id", help="name of the run, to start or to take up (default: the dataset and dwim's commit)")
    parser.add_argument("--runs", default=str(ROOT / "evals" / "runs"), help="where runs are kept (default: %(default)s)")
    parser.add_argument("--binary", help="the program to run, built for Linux and glibc 2.28 (default: built with cargo-zigbuild)")
    args = parser.parse_args(argv)
    args.dwim_args = dwim_args
    if args.provider == "local" and platform.system() != "Linux":
        parser.error("the model runs here for the containers only on Linux: run with --provider and a host")
    return args


def required(args):
    """Stops with what to install if a program the run needs is missing."""
    needed = ["docker"] + ([] if args.binary else ["cargo-zigbuild", "zig"])
    missing = [name for name in needed if not shutil.which(name)]
    if missing:
        sys.exit(f"{', '.join(missing)} not found: the run needs Docker, and to build dwim, cargo-zigbuild and zig")


def key(provider):
    """Puts the key to `provider` in the environment, which agent.py gives
    the containers, from where `dwim` keeps it if the environment has none:
    `dwim` in a container does not have the file it is kept in."""
    if os.environ.get(KEYS[provider], "").strip():
        return
    config = pathlib.Path.home() / ("Library/Application Support" if platform.system() == "Darwin" else ".config")
    try:
        os.environ[KEYS[provider]] = (pathlib.Path(os.environ.get("XDG_CONFIG_HOME") or config) / "dwim" / f"{provider}.key").read_text().strip()
    except FileNotFoundError:
        sys.exit(f"no key to {provider}: set {KEYS[provider]}, or sign in with /login in dwim")


def program(args):
    """The program to run: the one named, or built for the machine's
    architecture on Linux with glibc 2.28."""
    if args.binary:
        return pathlib.Path(args.binary).resolve()
    arch = {"arm64": "aarch64", "amd64": "x86_64"}.get(platform.machine().lower(), platform.machine())
    target = f"{arch}-unknown-linux-gnu"
    subprocess.run(["cargo", "zigbuild", "--release", "--bin", "dwim", "--target", f"{target}.{GLIBC}"], cwd=ROOT, check=True)
    return ROOT / "target" / target / "release" / "dwim"


def directory(args, binary):
    """The run's directory, with its manifest: made for a new run, and
    checked for one taken up, which has to be of the same dataset by the
    same `dwim` asked the same way."""
    now = manifest(args, binary)
    agent = now["agent"]
    name = args.run_id or f"terminalbench-{args.dataset.split('/')[-1].replace('@', '-')}-dwim-{agent['commit'][:12]}{'-dirty' if agent['dirty'] else ''}"
    run = pathlib.Path(args.runs).resolve() / name
    run.mkdir(parents=True, exist_ok=True)
    path = run / "manifest.json"
    if path.exists():
        was = json.loads(path.read_text())
        differs = [key for key in ("dataset", "tasks", "agent") if was.get(key) != now[key]]
        if differs:
            sys.exit(f"{run} is a run with another {' and '.join(differs)}, as its manifest.json has it: give this one another --run-id")
    else:
        write(path, json.dumps(now, indent=2) + "\n")
    return run


def manifest(args, binary):
    """What is run: the dataset and the tasks of it, and `dwim`, told by
    the commit it is built from, the last to change anything but evals/,
    and by whether anything but evals/ has changed since."""
    source = (".", ":(exclude)evals")
    version = subprocess.run([binary, "--version"], capture_output=True, text=True).stdout.strip()
    return {
        "dataset": args.dataset,
        "tasks": {"names": args.task, "limit": args.limit},
        "agent": {
            "name": "dwim",
            "version": version.removeprefix("dwim "),
            "commit": git("log", "-1", "--format=%H", "--", *source).strip(),
            "dirty": bool(git("status", "--porcelain", "--", *source).strip()),
            "provider": args.provider,
            "arguments": args.dwim_args,
        },
        "workers": args.workers,
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "host": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
    }


class Server:
    """The `dwim` server the containers' sessions are on, run from the
    program they run, with its socket in a directory of its own: a short
    path, since a socket's has to fit in 108 bytes, and one a container
    can have mounted without the rest of the user's runtime directory. A
    server exits after a while without a session, as it can between tasks
    while an image builds, so it is started again whenever it exits until
    the run is over."""

    def __init__(self, binary, log, sessions):
        self.binary, self.log, self.sessions = binary, log, sessions
        self.dir = pathlib.Path(f"/tmp/dwim-terminalbench-{os.getuid()}")
        self.socket = f"dwim-{os.getuid()}.sock"
        self.process = None
        self.stopping = threading.Event()

    def start(self):
        self.dir.mkdir(mode=0o700, exist_ok=True)
        self.spawn()
        deadline = time.monotonic() + LISTENING
        while not (self.dir / self.socket).is_socket():
            if self.process.poll() is not None or time.monotonic() > deadline:
                sys.exit(f"the dwim server did not start: see {self.log}")
            time.sleep(0.1)
        threading.Thread(target=self.watch, daemon=True).start()

    def spawn(self):
        with open(self.log, "a") as log:
            self.process = subprocess.Popen(
                [self.binary, "--serve", "--sessions", str(self.sessions)],
                env=os.environ | {"XDG_RUNTIME_DIR": str(self.dir)},
                stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            )

    def watch(self):
        while not self.stopping.is_set():
            if self.process.poll() is not None and not self.stopping.is_set():
                self.spawn()
            self.stopping.wait(1)

    def stop(self):
        self.stopping.set()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()

    def mounts(self):
        return [{"type": "bind", "source": str(self.dir), "target": SOCKETS}]


def harbor(args, binary, run, server):
    """Has Harbor run the tasks with agent.py, or take up the job it left
    in the run's directory. Harbor is run as this script's dependency, with
    agent.py where it can import it."""
    job = run / "job"
    env = os.environ | {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": os.pathsep.join(filter(None, [str(HERE), os.environ.get("PYTHONPATH")]))}
    executable = pathlib.Path(sys.executable).parent / "harbor"
    if (job / "config.json").exists():
        command = [executable, "jobs", "resume", "--job-path", job]
    else:
        mounts = [{"type": "bind", "source": str(binary), "target": PROGRAM, "read_only": True}]
        kwargs = {"provider": args.provider, "args": args.dwim_args, "version": json.loads((run / "manifest.json").read_text())["agent"]["version"]}
        if server:
            mounts += server.mounts()
            kwargs["socket"] = server.socket
        command = [
            executable, "run", "--dataset", args.dataset, "--agent", "agent:Dwim",
            "--jobs-dir", run, "--job-name", "job", "--n-concurrent", str(args.workers),
            "--mounts", json.dumps(mounts), "--yes",
            *[option for key, value in kwargs.items() for option in ("--ak", f"{key}={json.dumps(value)}")],
        ]
        for task in args.task or []:
            command += ["--include-task-name", task]
        if args.limit:
            command += ["--n-tasks", str(args.limit)]
    subprocess.run([str(part) for part in command], env=env, check=False)


def report(run, keep):
    """Writes the run up as Markdown in its directory: what was run, what
    came of it, and each task. A run of a whole dataset is one to `keep`,
    and its report goes to evals/results/ as well, named for the agent and
    the dataset, so that a later run's takes the place of an earlier one's."""
    said = json.loads((run / "manifest.json").read_text())
    agent = said["agent"]
    trials = sorted(trial(path) for path in (run / "job").glob("*/result.json"))
    if not trials:
        print(f"no tasks ran: see {run / 'job'}", file=sys.stderr)
        return
    known = [facts(t["transcript"]) for t in trials]
    served = sorted({fact["served"] for fact in known if fact.get("served")})
    solved = sum(1 for t in trials if t["outcome"] == "solved")
    dataset = said["dataset"].split("/")[-1].split("@")[0]
    name = f"dwim-{dataset}"
    lines = [
        f"# {name}", "", "| | |", "|---|---|",
        f"| Dataset | {said['dataset']}, {len(trials)} task{'' if len(trials) == 1 else 's'} |",
        f"| Agent | dwim {agent['version']}, commit {agent['commit'][:12]}{', with changes not committed' if agent['dirty'] else ''} |",
        f"| Model | {'; '.join(served) or agent['provider']} |",
        *settings(known),
        f"| Arguments | {' '.join(agent['arguments']) or 'none'} |",
        f"| Tasks at once | {said['workers']} |",
        f"| Started | {said['started']} |",
        "", f"**{solved} of {len(trials)} solved.**", "", "| Outcome | Tasks |", "|---|---|",
    ]
    tally = {}
    for t in trials:
        tally[t["outcome"]] = tally.get(t["outcome"], 0) + 1
    lines += [f"| {outcome} | {count} |" for outcome, count in sorted(tally.items(), key=lambda item: -item[1])]
    lines += [
        "", "A task is solved when its tests pass in the container `dwim` left, as",
        "Harbor runs them; a task whose agent ran out of the task's time is",
        "tested all the same, as it was left.",
        "", "## Tasks", "", "| Task | Seconds | Tool calls | Outcome |", "|---|---|---|---|",
    ]
    lines += [f"| {t['task']} | {t['seconds']} | {fact.get('calls', '')} | {t['outcome']} |" for t, fact in zip(trials, known)]
    text = "\n".join(lines) + "\n"
    write(run / "report.md", text)
    print(f"{solved} of {len(trials)} solved: report in {run / 'report.md'}")
    if keep:
        kept = ROOT / "evals" / "results" / f"{name}.md"
        write(kept, text)
        print(f"report in {kept}")


def trial(path):
    """What became of a task, from Harbor's result.json of its trial."""
    result = json.loads(path.read_text())
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    error = (result.get("exception_info") or {}).get("exception_type")
    if rewards:
        outcome = "solved" if all(value == 1 for value in rewards.values()) else "not solved"
        if error == "AgentTimeoutError":
            outcome += ", out of time"
    else:
        outcome = f"not tested: {error}" if error else "not tested"
    timing = result.get("agent_execution") or {}
    seconds = ""
    if timing.get("started_at") and timing.get("finished_at"):
        took = datetime.datetime.fromisoformat(timing["finished_at"]) - datetime.datetime.fromisoformat(timing["started_at"])
        seconds = round(took.total_seconds())
    return {"task": result["task_name"], "outcome": outcome, "seconds": seconds, "transcript": path.parent / "agent" / "transcript.jsonl"}


def facts(transcript):
    """What a task's transcript tells: how many tool calls `dwim` made,
    where the model was served and with how much context, and the
    reasoning effort, thinking budget and sampling it was started with."""
    try:
        rows = [json.loads(line) for line in transcript.read_text().splitlines() if line.startswith("{")]
    except (FileNotFoundError, ValueError):
        return {}
    known = {"calls": sum(1 for row in rows if row.get("type") == "call")}
    if rows and rows[0].get("type") == "start":
        start = rows[0]
        known["served"] = f"{start.get('model')}, served on {start.get('device')} with a context of {start.get('context')} tokens"
        known["effort"] = start.get("reasoning_effort")
        if "thinking_budget" in start:
            known["budget"] = start["thinking_budget"] or 0
        if "temperature" in start:
            known["sampling"] = f"temperature {start['temperature']}, top-p {start.get('top_p')}"
    return known


def settings(known):
    """The rows of the report's table for what `dwim` was started with, as
    the transcripts have it: every value any of them had."""
    def values(key, say=str):
        return ", ".join(say(value) for value in sorted({fact[key] for fact in known if fact.get(key) is not None}, key=str))
    rows = [
        ("Reasoning effort", values("effort")),
        ("Thinking budget", values("budget", lambda tokens: f"{tokens} tokens a thought" if tokens else "none")),
        ("Sampling", values("sampling")),
    ]
    return [f"| {name} | {value} |" for name, value in rows if value]


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


if __name__ == "__main__":
    main()
