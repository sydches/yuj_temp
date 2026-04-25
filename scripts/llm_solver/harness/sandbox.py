"""Sandbox — bubblewrap mount namespace enforcement for the bash tool.

Every bash tool call runs inside a bwrap mount namespace where:
  - The host filesystem is bound read-only at /.
  - /tmp is a fresh tmpfs (isolated per call, no state leaks across
    tool invocations).
  - The task's cwd is the ONLY writable path, bound at its real host
    path (matched source/target) so `docker run -v $PWD:/testbed`
    inside the sandbox still resolves correctly.
  - /proc and /dev are provided for a working process view.
  - The host's docker socket is bound in when present (resolved to
    the canonical /run path — /var/run is typically a symlink).
  - --die-with-parent tears the sandbox down if the harness exits.
  - --chdir ensures $PWD resolves to the task dir.

This module is load-bearing for the experiment: the model cannot
escape the cwd through any bash invocation. See ``docs/sandbox.md``
for the full security rationale and escape-attempt verification.
"""
from __future__ import annotations

from pathlib import Path

# Default path; the effective path comes from config.toml [tools] bwrap_bin.
_DEFAULT_BWRAP_BIN = "/usr/bin/bwrap"


_DOCKER_SOCK_CACHE: tuple[bool, str | None] = (False, None)


def _resolve_docker_sock() -> str | None:
    """Return the canonical path to the host's docker socket, or None.

    Most modern Linux distros ship /var/run as a symlink to /run, so the
    "real" socket lives at /run/docker.sock. Binding through the symlink
    (/var/run/docker.sock) fails under bwrap because the symlink target
    resolution happens after the bind is attempted, so the mount source
    reads as missing. Resolving here avoids that whole class of failure.

    Memoized for the process lifetime: the socket path is stable once
    the daemon is up, and _build_bwrap_argv runs per-bash-call, so the
    uncached version costs one stat syscall per tool invocation for no
    benefit.
    """
    global _DOCKER_SOCK_CACHE
    cached, value = _DOCKER_SOCK_CACHE
    if cached:
        return value
    for candidate in ("/run/docker.sock", "/var/run/docker.sock"):
        p = Path(candidate)
        if p.exists():
            _DOCKER_SOCK_CACHE = (True, str(p.resolve()))
            return _DOCKER_SOCK_CACHE[1]
    _DOCKER_SOCK_CACHE = (True, None)
    return None


def _build_bwrap_argv(cmd: str, cwd: str, bwrap_bin: str = _DEFAULT_BWRAP_BIN) -> list[str]:
    """Build a bwrap argv that runs `cmd` inside a sandbox.

    Sandbox shape:
      - Entire host filesystem bound read-only at /
      - Fresh /tmp as tmpfs (isolated per call, no state leaks across
        tool invocations). Mounted BEFORE the cwd bind so a cwd that
        happens to live under /tmp (e.g. in tests) isn't wiped by the
        tmpfs mount.
      - cwd bound writable at its real path (matched source/target) so
        that any `docker run -v $PWD:/testbed` inside the sandbox still
        resolves correctly — the docker daemon lives on the host and
        reads HOST paths for bind mounts, so the sandbox view's $PWD
        must equal the host path. Avoid remapping cwd to /work or
        anything else; docker would then fail to find the source dir.
      - /proc and /dev for a working process view.
      - Docker socket bound in if present (resolved to the canonical
        /run path — /var/run is typically a symlink). Needed for
        pretest.sh which execs `docker run`.
      - --die-with-parent so the sandbox tears down instantly if the
        harness exits.
      - --chdir to the cwd so $PWD resolves correctly to the task dir.

    The result is passed to subprocess.run as an argv list (no shell).
    The final `bash -c "$cmd"` runs the model's shell command inside the
    namespace where only `cwd` is writable.
    """
    argv = [
        bwrap_bin,
        "--ro-bind", "/", "/",
        "--tmpfs", "/tmp",
        "--bind", cwd, cwd,
        "--proc", "/proc",
        "--dev", "/dev",
        "--die-with-parent",
        "--chdir", cwd,
        # Determinism: pin MPLCONFIGDIR so matplotlib doesn't emit the
        # random-suffixed "Matplotlib created a temporary cache directory
        # at /tmp/matplotlib-XXXXXXXX" banner that varies per run.
        "--setenv", "MPLCONFIGDIR", "/tmp/mpl",
    ]
    sock = _resolve_docker_sock()
    if sock is not None:
        argv += ["--bind", sock, sock]
    argv += ["bash", "-c", cmd]
    return argv
