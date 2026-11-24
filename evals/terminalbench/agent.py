"""`dwim` as a Harbor agent, which run.py has Harbor run Terminal-Bench with.

Terminal-Bench scores a task by the state the agent leaves its container
in, so `dwim` runs in the container, where its tools see the files the
tests do. It is the program run.py built, mounted at /opt/dwim/dwim, which
runs on any image with glibc 2.28 or later. The model runs on this
machine, in the `dwim` server run.py started from the same program, whose
socket's directory is mounted at /run/dwim: a session checks that its
server is of its own build, and the program mounted is the same file.
With a host for its provider, `dwim` asks the host from the container,
with the key from Harbor's environment.
"""
import asyncio, os, shlex

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

# Where run.py mounts the program and the server's socket's directory.
PROGRAM = "/opt/dwim/dwim"
SOCKETS = "/run/dwim"

# Where `dwim` looks for the server's socket in the container: a directory
# of its own, with a link named for the user the task runs as to the
# socket mounted, which is named for the user the server runs as.
RUNTIME = "/tmp/dwim-runtime"

# The keys to the hosts, given to `dwim` in the container if Harbor has
# them.
KEYS = ("OPENROUTER_API_KEY", "AI_GATEWAY_API_KEY", "CEREBRAS_API_KEY")


class Dwim(BaseAgent):
    """`dwim` answering the task's instruction as a one-off prompt."""

    def __init__(self, logs_dir, model_name=None, *, provider="local", socket=None, args=None, version=None, **kwargs):
        super().__init__(logs_dir, None, **kwargs)
        self.provider = provider
        self.socket = socket
        self.args = list(args or [])
        self._version = version

    @staticmethod
    def name():
        return "dwim"

    def version(self):
        return self._version

    async def setup(self, environment: BaseEnvironment) -> None:
        """Checks that the program runs in the task's image and, for the
        model on this machine, that the server's socket is there: without
        it, `dwim` would start a server of its own in the container, and
        download the model to it."""
        ran = await environment.exec(f"{PROGRAM} --version", timeout_sec=60)
        if ran.return_code != 0:
            raise RuntimeError(f"dwim does not run in this image: {(ran.stderr or ran.stdout or '').strip()}")
        if self.provider == "local":
            found = await environment.exec(f"test -S {SOCKETS}/{shlex.quote(self.socket)}", timeout_sec=60)
            if found.return_code != 0:
                raise RuntimeError(f"the dwim server's socket is not at {SOCKETS}/{self.socket} in the container")

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        """Runs `dwim` on the instruction in the task's directory, with its
        transcript and what it prints in the logs Harbor keeps. Stopped by
        Harbor's time limit, it is killed in the container too, so that it
        does not go on changing files while the tests run."""
        logs = self.environment_logs_dir
        command = [
            PROGRAM, "--provider", self.provider, "--transcript", f"{logs}/transcript.jsonl",
            *self.args, "--", instruction,
        ]
        script = f"mkdir -p {logs} {RUNTIME} && chmod 700 {RUNTIME} && echo $$ > {RUNTIME}/pid && "
        if self.provider == "local":
            script += f"ln -sf {SOCKETS}/{shlex.quote(self.socket)} {RUNTIME}/dwim-$(id -u).sock && "
        script += f"exec {shlex.join(command)} > {logs}/dwim.txt 2>&1"
        env = {"XDG_RUNTIME_DIR": RUNTIME} | {key: os.environ[key] for key in KEYS if os.environ.get(key)}
        try:
            ran = await environment.exec(script, env=env)
        except asyncio.CancelledError:
            await asyncio.shield(environment.exec(f"kill $(cat {RUNTIME}/pid) 2>/dev/null; true", timeout_sec=30))
            raise
        context.metadata = {"exit_code": ran.return_code}
