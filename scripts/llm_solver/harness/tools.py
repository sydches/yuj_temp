"""Tool implementations — bash, read, write, edit, glob, grep + dispatch."""
from dataclasses import dataclass
import logging
import re
import shutil
import subprocess
from typing import Callable
from pathlib import Path

from ..config import Config
from .sandbox import _DEFAULT_BWRAP_BIN, _build_bwrap_argv

log = logging.getLogger(__name__)

# ── ANSI pattern ────────────────────────────────────────────────────────
# Terminal control protocol — universal across any subprocess output,
# not specific to any task format. Stripping it is content-blind noise
# removal. Task-environment-specific path conventions (e.g. container mount
# points) are NOT rewritten in the harness — that knowledge belongs in the
# environment/setup layer, not in the generic tool surface.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# ls -l style listing: permissions + metadata + date + filename.
# Under temp=0, varying wall-clock mtimes on harness-owned files (e.g.
# .trace.jsonl which appends every turn) flip the model's sampled path
# turn-over-turn.  Replace the date column with a fixed placeholder
# while preserving column alignment; filename and content are untouched.
_LS_LONG_RE = re.compile(
    r'^([-dlcbps][rwxstST-]{9}[+.@]?\s+\d+\s+\S+\s+\S+\s+\d+\s+)'
    r'[A-Z][a-z]{2}\s+\d{1,2}\s+(?:\d{1,2}:\d{2}|\d{4})'
    r'(\s+\S)',
    re.MULTILINE,
)


def _strip_ls_timestamps(output: str) -> str:
    """Replace ls -l style date columns with a fixed placeholder.

    Only touches lines matching the exact ls long-format shape; other
    dates elsewhere in output are left alone.
    """
    return _LS_LONG_RE.sub(r'\1Jan  1  2020\2', output)


def _resolve(cwd: str, path: str) -> Path:
    """Resolve a tool path relative to cwd, even if absolute.

    Absolute paths are resolved relative to cwd (sandbox-style)
    so the model can't escape the working directory.
    """
    if path.startswith("/"):
        path = path.lstrip("/")
    return Path(cwd) / path


def _path_hint(cwd: str, path: str) -> str:
    """Suggest a corrected path when a file-not-found error occurs.

    Catches the `.seaborn/X` → `seaborn/X` pattern where the model
    confuses `.solver/` (hidden dir) with the package directory.
    """
    stripped = path.lstrip("./")
    if stripped != path:
        candidate = Path(cwd) / stripped
        if candidate.exists():
            return f" (did you mean '{stripped}'?)"
    return ""


def _record_truncation_savings(input_chars: int, output_chars: int,
                               head_ratio: float) -> None:
    """Log head+tail truncation savings to the ledger when a cut happened."""
    if output_chars >= input_chars:
        return
    from .savings import get_ledger
    get_ledger().record(
        bucket="truncate_output",
        layer="harness",
        mechanism="head_tail_truncation",
        input_chars=input_chars,
        output_chars=output_chars,
        measure_type="exact",
        ctx={"head_ratio": head_ratio},
    )


def truncate_output(text: str, cfg: Config) -> str:
    """Head+tail truncation when output exceeds max_output_chars.

    Budget-targeted slice: any tool result at or under max_output_chars
    passes through untouched. Over-budget results are head+tail sliced
    such that the resulting output is roughly max_output_chars, split
    60/40 head/tail and rounded to full line boundaries when possible.

    The earlier design sliced by fixed truncate_head_lines /
    truncate_tail_lines counts, which was a problem once max_output_chars
    was raised: an 87 KB Python source file with ~100-char lines would
    be cut to ~30 KB (300 lines) even though the budget allowed 80 KB,
    silently throwing away ~70% of the code on first read. The budget
    SHOULD be what governs; line counts are a derived thing.

    Line-count fields (truncate_head_lines / truncate_tail_lines) are
    still used as a floor — for command output with a few very long
    lines, they guarantee at least N logical lines of each end survive
    even if the char budget math would leave nothing. This preserves
    the readability property for bash log tails.
    """
    budget = cfg.max_output_chars
    if len(text) <= budget:
        return text

    # Reserve a small overhead for the "[... omitted ...]" marker so the
    # final result fits under budget.
    marker_reserve = 80
    slice_budget = max(1, budget - marker_reserve)
    head_budget = int(slice_budget * cfg.truncate_head_ratio)
    tail_budget = slice_budget - head_budget

    # Char-based head/tail respecting full lines where possible.
    head = text[:head_budget]
    # Back up to the last newline so we don't cut a line mid-way.
    last_nl = head.rfind("\n")
    if last_nl > head_budget // 2:
        head = head[: last_nl + 1]

    tail = text[-tail_budget:]
    first_nl = tail.find("\n")
    if 0 <= first_nl < tail_budget // 2:
        tail = tail[first_nl + 1 :]

    omitted = len(text) - len(head) - len(tail)
    truncated = f"{head}\n[... {omitted} chars omitted ...]\n{tail}"
    _record_truncation_savings(len(text), len(truncated), cfg.truncate_head_ratio)
    return truncated


def _collapse_duplicate_lines(output: str) -> str:
    """Collapse runs of byte-identical consecutive lines into '<line> [×N]'.

    Content-blind redundancy compression: compresses on byte equality,
    knows nothing about what the lines represent. Works on retry-loop
    spam, progress-bar repeats, log rotation, test runners that print
    identical status lines — anything that repeats verbatim. No task-
    format vocabulary is named.

    A run of length 1 passes through unchanged (no overhead annotation
    for unique lines). Runs of 2+ identical lines collapse to the line
    followed by a compact count suffix.
    """
    lines = output.split("\n")
    out: list[str] = []
    prev: str | None = None
    count = 0
    for line in lines:
        if prev is not None and line == prev:
            count += 1
            continue
        if prev is not None:
            if count > 1:
                out.append(f"{prev} [×{count}]")
            else:
                out.append(prev)
        prev = line
        count = 1
    if prev is not None:
        if count > 1:
            out.append(f"{prev} [×{count}]")
        else:
            out.append(prev)
    return "\n".join(out)


# ── Structural skeleton patterns ────────────────────────────────────────
# Used by _collapse_similar_lines to detect structurally identical lines
# that differ only in variable alphanumeric content (names, numbers, etc.).
_ALNUM_RE = re.compile(r"[A-Za-z0-9]+")
_WS_RE = re.compile(r"\s+")


def _line_skeleton(line: str) -> str:
    """Return the structural skeleton of a line.

    Replaces every run of alphanumeric characters with a NUL placeholder
    and collapses whitespace runs.  Lines with identical skeletons share
    the same punctuation/delimiter template and differ only in their
    variable alphanumeric content (names, numbers, percentages).

    Content-blind: operates only on character-class properties, not on
    any knowledge of what the alphanumeric values represent.
    """
    s = _ALNUM_RE.sub("\x00", line)
    s = _WS_RE.sub(" ", s)
    return s.strip()


def _collapse_similar_lines(output: str) -> str:
    """Collapse high-frequency structural templates, keep rare lines verbatim.

    Two-pass, frequency-based:
      1. Skeleton every line, count skeleton frequencies.
      2. Skeletons that account for > 50% of non-blank lines are "bulk" —
         consecutive runs of bulk lines collapse to first + count + last.
         All other lines pass through unchanged.

    The effect: in a 14K-line pytest run, the 12K PASSED lines share one
    dominant skeleton and collapse.  The 119 FAILED lines, the header,
    the summary, and every structurally unique line survive intact.

    For small outputs or outputs with no dominant template, nothing
    collapses — every skeleton is rare.

    Content-blind: decides by skeleton frequency, not by what the lines
    say.  Works on any tool output where one structural pattern dominates.
    """
    lines = output.split("\n")
    n_nonblank = sum(1 for l in lines if l.strip())
    if n_nonblank < 10:
        return output

    # Pass 1: skeleton frequencies.
    skeletons = []
    freq: dict[str, int] = {}
    for line in lines:
        if not line.strip():
            skeletons.append("")
            continue
        skel = _line_skeleton(line)
        skeletons.append(skel)
        freq[skel] = freq.get(skel, 0) + 1

    # Bulk threshold: skeletons covering > 50% of non-blank lines.
    threshold = n_nonblank * 0.5
    bulk = {s for s, c in freq.items() if s and c > threshold}

    if not bulk:
        return output

    # Pass 2: emit. Consecutive bulk lines collapse; rare lines pass through.
    out: list[str] = []
    i = 0
    while i < len(lines):
        skel = skeletons[i]
        if skel not in bulk:
            out.append(lines[i])
            i += 1
            continue

        # Start of a bulk run — scan forward.
        j = i + 1
        while j < len(lines) and skeletons[j] == skel:
            j += 1
        run_len = j - i
        if run_len < 3:
            out.extend(lines[i:j])
        else:
            out.append(lines[i])
            out.append(f"  ... [×{run_len} similar lines]")
            out.append(lines[j - 1])
        i = j

    return "\n".join(out)



def _filter_bash_output(output: str, cmd: str, cfg: Config) -> str:
    """Content-agnostic filtering of bash output before truncation.

    Transforms (each gated by a config toggle):
      1. Strip ANSI escape sequences  (cfg.strip_ansi)
      2. Collapse runs of blank lines  (cfg.collapse_blank_lines)
      3. Collapse runs of byte-identical lines  (cfg.collapse_duplicate_lines)

    Every transformation is content-blind: no task-format vocabulary,
    no test-runner detection, no error-message pattern matching. Each
    operates on universal properties of text — terminal control
    protocol, whitespace, byte equality. Task-format parsing belongs
    in the analysis layer, never in the solve loop.
    """
    # 1. ANSI escapes — terminal control protocol, universal noise.
    if cfg.strip_ansi:
        output = _ANSI_RE.sub("", output)

    # 2. Collapse blank-line runs (3+ consecutive newlines → 2).
    if cfg.collapse_blank_lines:
        output = re.sub(r"\n{3,}", "\n\n", output)

    # 3. Collapse runs of byte-identical consecutive lines.
    if cfg.collapse_duplicate_lines:
        output = _collapse_duplicate_lines(output)

    # 4. Collapse runs of structurally similar lines (same skeleton).
    #    Only activates when the output is large enough that truncation
    #    is a real threat (>50% of the char budget).  Small outputs pass
    #    through untouched — collapsing an `ls` or short grep loses
    #    unique information (filenames, paths) for negligible savings.
    if cfg.collapse_similar_lines and len(output) > cfg.max_output_chars * 0.5:
        output = _collapse_similar_lines(output)

    return output


_STATUS_WORD_RE = re.compile(
    r'\b(?:passed|failed|error|warnings?|deselected|no tests ran|no tests collected)\b'
)
_TIMING_RE = re.compile(r'\s*in\s+\d+\.\d+s')


def _strip_runner_timing(output: str) -> str:
    """Strip wall-clock timing from pytest/unittest lines in bash output.

    Applies to any line containing a pytest status word (passed/failed/
    error/warning/deselected) or the "no tests ran"/"no tests collected"
    phrases. Removes ` in X.YZs` — varies sub-second per invocation and
    flips temp=0 paths.
    """
    out_lines = []
    for line in output.split('\n'):
        if _STATUS_WORD_RE.search(line):
            line = _TIMING_RE.sub('', line)
        out_lines.append(line)
    return '\n'.join(out_lines)


def _strip_cwd_absolute(output: str, cwd: str) -> str:
    """Rewrite cwd absolute path to ``.`` in bash output.

    ``pwd`` and Python ``__file__`` resolve to the task's absolute path
    which embeds the run_dir timestamp. Under temp=0 the timestamp
    bytes flip the model's sampled next token on subsequent turns.
    Collapsing to ``.`` makes output byte-identical across runs.
    """
    return output.replace(cwd, ".")


# Python object repr at hex memory address: "at 0x7ff2a756e110".
# Memory allocation is non-deterministic (ASLR, allocator state),
# so the same object has different addresses across runs.
_HEX_ADDR_RE = re.compile(r'\bat 0x[0-9a-fA-F]+\b')


def _strip_hex_addresses(output: str) -> str:
    """Replace ``at 0xDEADBEEF`` with ``at 0xXXXX`` for determinism."""
    return _HEX_ADDR_RE.sub('at 0xXXXX', output)


def bash(cmd: str, *, cwd: str, timeout: int, sandbox: bool = True,
         bwrap_bin: str = _DEFAULT_BWRAP_BIN) -> str:
    """Run a shell command, return stdout+stderr.

    When `sandbox=True` and /usr/bin/bwrap exists, the command runs inside
    a bwrap sandbox that treats the entire host as read-only and only
    `cwd` as writable. See `_build_bwrap_argv` for the shape.

    If bwrap is unavailable, falls back to plain `subprocess.run(shell=True,
    cwd=cwd)`. The fallback is logged once per process so the degradation
    is visible, not silent.
    """
    try:
        if sandbox and Path(bwrap_bin).is_file():
            argv = _build_bwrap_argv(cmd, cwd, bwrap_bin)
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        else:
            if sandbox:
                log.warning(
                    "sandbox_bash=true but %s not found — running without sandbox",
                    bwrap_bin,
                )
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        out = result.stdout + result.stderr
        out = _strip_ls_timestamps(out)
        out = _strip_runner_timing(out)
        out = _strip_cwd_absolute(out, cwd)
        out = _strip_hex_addresses(out)
        if result.returncode != 0:
            out += f"\n[exit code: {result.returncode}]"
        return out
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {timeout}s"
    except Exception as e:
        return f"ERROR: {e}"


def _record_read_reminder(kind: str, path: str, reminder_chars: int) -> None:
    """Record a read-tool reminder injection on the savings ledger.

    Bucket ``tool_result_reminder``; ``kind`` ∈ {"truncated", "empty"}.
    Input chars = 0 (nothing pre-existed); output chars = reminder length,
    so ``delta_chars`` is the positive cost paid to inject the block.
    """
    from .savings import get_ledger
    get_ledger().record(
        bucket="tool_result_reminder",
        layer="harness",
        mechanism=f"read_{kind}",
        input_chars=0,
        output_chars=int(reminder_chars),
        measure_type="exact",
        ctx={"kind": kind, "path": path},
    )


def read(path: str, *, cwd: str, offset: int = 0, limit: int = 0,
         cfg: Config | None = None) -> str:
    """Read a file, return contents with line numbers.

    When ``cfg`` is provided, appends a ``<system-reminder>`` block to
    the result in two cases:
      - the caller's ``limit`` capped the output before EOF
        (``read_truncated_reminder``);
      - the file exists but is 0-byte (``read_empty_reminder``).

    Reminders are off when ``cfg`` is None — preserves the signature
    for non-dispatch callers (tests, direct imports).
    """
    try:
        target = _resolve(cwd, path)
        all_lines = target.read_text().splitlines()
        total = len(all_lines)
        if offset > 0:
            lines = all_lines[offset:]
        else:
            lines = all_lines
        truncated = False
        returned = len(lines)
        if limit > 0 and returned > limit:
            lines = lines[:limit]
            returned = limit
            truncated = True
        start = (offset or 0) + 1
        numbered = [f"{start + i}: {line}" for i, line in enumerate(lines)]
        body = "\n".join(numbered)
        if cfg is None:
            return body
        if total == 0:
            tail = cfg.read_empty_reminder.format(path=path)
            _record_read_reminder("empty", path, len(tail))
            return body + ("\n" if body else "") + tail
        if truncated:
            tail = cfg.read_truncated_reminder.format(
                returned_lines=returned, path=path)
            _record_read_reminder("truncated", path, len(tail))
            return body + "\n" + tail
        return body
    except FileNotFoundError:
        return f"ERROR: file not found: {path}" + _path_hint(cwd, path)
    except Exception as e:
        return f"ERROR: {e}"


def write(path: str, content: str, *, cwd: str,
          cfg: Config | None = None) -> str:
    """Create or overwrite a file.

    When post-edit validation is enabled and matching checks fire,
    their outcome is applied:
      - on_fail="append" / "warn": tail appended to the OK result
      - on_fail="block": file reverted to prior state; ERROR returned
    """
    target = _resolve(cwd, path)
    existed_before = target.exists()
    previous_content: str | None = None
    if existed_before:
        try:
            previous_content = target.read_text()
        except OSError:
            previous_content = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        head = f"OK: wrote {len(content)} bytes to {path}"
        from .post_edit import run_post_edit_checks
        res = run_post_edit_checks(path, cwd=cwd, cfg=cfg, trigger="write")
        if res.action == "block":
            if previous_content is not None:
                target.write_text(previous_content)
            elif not existed_before:
                try:
                    target.unlink()
                except OSError:
                    pass
            return (
                f"ERROR: write blocked by post-edit check "
                f"'{res.check_name}' for {path}{res.output}"
            )
        return head + res.output
    except Exception as e:
        return f"ERROR: {e}"


def _whitespace_normalized_match(text: str, old_str: str) -> tuple[int, int] | None:
    """Back-compat shim — delegates to the new cascade module.

    Retained for external callers and for tests that import the name
    directly. New code should import from ``edit_replacers`` instead.
    """
    from .edit_replacers import whitespace_normalized
    return whitespace_normalized(text, old_str)


def _record_edit_recovery(mechanism: str, path: str, old_str_len: int) -> None:
    """Record that a fuzzy-edit strategy rescued a non-exact match.

    Bucket ``fuzzy_edit_recovery`` on the savings ledger. Char delta
    is zero — the event records that the cascade fired, not a token
    saving. Use ``ctx.strategy`` when aggregating.
    """
    from .savings import get_ledger
    get_ledger().record(
        bucket="fuzzy_edit_recovery",
        layer="harness",
        mechanism=mechanism,
        input_chars=old_str_len,
        output_chars=old_str_len,
        measure_type="estimate",
        ctx={"strategy": mechanism, "path": path},
    )


def _format_candidates_block(text: str, candidates, path: str) -> str:
    """Render a ranked-candidate block for strict-mode miss reporting.

    Uses XML-shape envelope so the agent can parse by tag shape. Each
    inner <candidate> quotes the exact substring of ``text`` between
    the candidate's (start, end) offsets — the agent can copy it
    verbatim into a retry.
    """
    if not candidates:
        return ""
    # cause hint: the strategy of the top-ranked candidate
    top = candidates[0]
    cause_map = {
        "whitespace_normalized": "whitespace_drift",
        "line_trimmed": "trailing_whitespace",
        "indentation_flexible": "indent_drift",
        "escape_normalized": "escape_artifact",
        "trimmed_boundary": "boundary_whitespace",
        "block_anchor": "interior_drift",
    }
    cause = cause_map.get(top.strategy, top.strategy)
    lines = [
        f'<candidates total="{len(candidates)}" cause_hint="{cause}" path="{_xml_attr(path)}">'
    ]
    for rank, c in enumerate(candidates, start=1):
        snippet = text[c.start:c.end]
        lines.append(
            f'  <candidate rank="{rank}" strategy="{c.strategy}" '
            f'similarity="{c.similarity:.2f}" line="{c.line_number}">'
        )
        lines.append(snippet)
        lines.append('  </candidate>')
    lines.append('</candidates>')
    return "\n".join(lines)


def edit(path: str, old_str: str, new_str: str, *, cwd: str,
         cfg: Config | None = None) -> str:
    """Replace first occurrence of old_str with new_str in a file.

    Match policy is controlled by two cfg flags:

      edit_strict_match (default true) + edit_fuzzy_cascade_enabled
      (default false):
          exact match only; on miss, return a ranked <candidates/>
          block so the agent can choose and retry.

      edit_fuzzy_cascade_enabled = true:
          fall through to the cascade after an exact miss and
          auto-apply the first passing strategy (the aa81a62
          behavior, preserved as a DOE arm).

    When cfg is None (test-only convenience), strict mode is used.
    """
    from .edit_replacers import find_span, rank_candidates
    from .post_edit import run_post_edit_checks
    try:
        target = _resolve(cwd, path)
        text = target.read_text()
        new_text: str | None = None
        head = ""
        # Pass 1: exact.
        if old_str in text:
            new_text = text.replace(old_str, new_str, 1)
            head = "OK"
        elif cfg is not None and cfg.edit_fuzzy_cascade_enabled:
            # DOE arm: auto-apply first matching strategy.
            hit = find_span(text, old_str)
            if hit is not None:
                mechanism, start, end = hit
                new_text = text[:start] + new_str + text[end:]
                head = f"OK ({mechanism.replace('_', '-')})"
                _record_edit_recovery(mechanism, path, len(old_str))
        if new_text is None:
            # Strict-mode miss (or cascade-miss): surface ranked
            # candidates so the agent can retry with correct bytes.
            k = cfg.edit_candidate_count if cfg is not None else 3
            candidates = rank_candidates(text, old_str, k=k)
            head = f"ERROR: old_str not found in {path}"
            block = _format_candidates_block(text, candidates, path)
            return f"{head}\n{block}" if block else head
        target.write_text(new_text)
        res = run_post_edit_checks(path, cwd=cwd, cfg=cfg, trigger="edit")
        if res.action == "block":
            target.write_text(text)
            return (
                f"ERROR: edit blocked by post-edit check "
                f"'{res.check_name}' for {path}{res.output}"
            )
        return head + res.output
    except FileNotFoundError:
        return f"ERROR: file not found: {path}" + _path_hint(cwd, path)
    except Exception as e:
        return f"ERROR: {e}"


def _xml_attr(s: str) -> str:
    """Escape a string for inclusion as an XML attribute value.

    Only the five XML-reserved characters are escaped; the payload
    inside tags stays verbatim.
    """
    return (
        s.replace("&", "&amp;")
         .replace('"', "&quot;")
         .replace("'", "&apos;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )


def _paginated_envelope(
    *, tool: str, pattern: str, scope: str, lines: list[str],
    page: int, per_page: int,
) -> str:
    """Wrap ``lines`` in a ``<search_result/>`` envelope for grep/glob.

    Emits opening tag with total/shown/page/next_page/pattern/scope
    attributes, then the page's slice of raw lines verbatim between
    the tags. Lines themselves are not escaped — downstream parsers
    that split on ``\\n`` and read ``path:line:content`` continue to
    work inside the envelope.
    """
    total = len(lines)
    if per_page <= 0:
        per_page = total or 1
    page = max(1, page)
    start = (page - 1) * per_page
    end = start + per_page
    shown_slice = lines[start:end]
    next_page = page + 1 if end < total else 0
    opening = (
        f'<search_result tool="{tool}" total="{total}" '
        f'shown="{len(shown_slice)}" page="{page}" '
        f'next_page="{next_page}" pattern="{_xml_attr(pattern)}" '
        f'scope="{_xml_attr(scope)}">'
    )
    body = "\n".join(shown_slice) if shown_slice else ""
    return f"{opening}\n{body}\n</search_result>"


def glob_files(pattern: str, path: str = ".", *, cwd: str,
               page: int = 1, cfg: Config | None = None) -> str:
    """Find files matching a glob pattern.

    When ``cfg.search_pagination_enabled`` is true, wraps the result
    in a ``<search_result/>`` envelope with total/shown/page/next_page
    attributes. When false or ``cfg`` is None, returns the raw line
    list (backwards compatible with pre-pagination callers).
    """
    try:
        base = _resolve(cwd, path)
        matches = sorted(base.glob(pattern))
        rel = [str(m.relative_to(cwd)) for m in matches if m.is_file()]
        if cfg is None or not cfg.search_pagination_enabled:
            if not rel:
                return "No files found."
            return "\n".join(rel)
        return _paginated_envelope(
            tool="glob", pattern=pattern, scope=path,
            lines=rel, page=page,
            per_page=cfg.glob_max_matches_per_page,
        )
    except Exception as e:
        return f"ERROR: {e}"


def grep_files(
    pattern: str, path: str = ".", glob_filter: str = "",
    *, cwd: str, timeout: int = 30,
    page: int = 1, cfg: Config | None = None,
) -> str:
    """Search file contents with regex using ripgrep or grep fallback.

    When ``cfg.search_pagination_enabled`` is true, wraps the result
    in a ``<search_result/>`` envelope with total/shown/page/next_page
    attributes. When false or ``cfg`` is None, returns the raw
    line-per-match text (backwards compatible).
    """
    resolved_path = str(_resolve(cwd, path))
    rg = shutil.which("rg")
    if rg:
        cmd = [rg, "-n", "--no-heading"]
        if glob_filter:
            cmd.extend(["--glob", glob_filter])
        cmd.extend([pattern, resolved_path])
    else:
        cmd = ["grep", "-rn"]
        if glob_filter:
            cmd.extend(["--include", glob_filter])
        cmd.extend([pattern, resolved_path])
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        raw = result.stdout
        if cfg is None or not cfg.search_pagination_enabled:
            return raw or "No matches found."
        lines = raw.splitlines() if raw else []
        scope = f"{path}" + (f" glob={glob_filter}" if glob_filter else "")
        return _paginated_envelope(
            tool="grep", pattern=pattern, scope=scope,
            lines=lines, page=page,
            per_page=cfg.grep_max_matches_per_page,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: grep timed out after {timeout}s"
    except Exception as e:
        return f"ERROR: {e}"


_DISPATCH = {
    "bash": lambda args, cwd, cfg: bash(
        args["cmd"], cwd=cwd, timeout=cfg.bash_timeout, sandbox=cfg.sandbox_bash,
        bwrap_bin=cfg.bwrap_bin,
    ),
    "read": lambda args, cwd, cfg: read(
        args["path"], cwd=cwd, offset=args.get("offset", 0),
        limit=args.get("limit", 0), cfg=cfg,
    ),
    "write": lambda args, cwd, cfg: write(
        args["path"], args["content"], cwd=cwd, cfg=cfg,
    ),
    "edit": lambda args, cwd, cfg: edit(
        args["path"], args["old_str"], args["new_str"], cwd=cwd, cfg=cfg,
    ),
    "glob": lambda args, cwd, cfg: glob_files(
        args["pattern"], args.get("path", "."), cwd=cwd,
        page=int(args.get("page", 1)), cfg=cfg,
    ),
    "grep": lambda args, cwd, cfg: grep_files(
        args["pattern"], args.get("path", "."), args.get("glob", ""),
        cwd=cwd, timeout=cfg.grep_timeout,
        page=int(args.get("page", 1)), cfg=cfg,
    ),
    "done": lambda args, cwd, cfg: "done",
}


@dataclass(frozen=True)
class ToolRegistry:
    """Composable tool-dispatch registry."""

    handlers: dict[str, Callable[[dict, str, Config], str]]


def build_tool_registry(
    *,
    overrides: dict[str, Callable[[dict, str, Config], str]] | None = None,
) -> ToolRegistry:
    """Build the effective tool registry with optional handler overrides."""
    handlers = dict(_DISPATCH)
    if overrides:
        handlers.update(overrides)
    return ToolRegistry(handlers=handlers)


def validate_tool_handlers(schema_names: list[str], *,
                           allow_extra_handlers: bool = True,
                           registry: ToolRegistry | None = None) -> None:
    """Fail fast when tool schemas and dispatch handlers drift."""
    declared = set(schema_names)
    reg = registry or build_tool_registry()
    handlers = set(reg.handlers.keys())
    missing_handlers = declared - handlers
    undeclared_handlers = handlers - declared
    if missing_handlers or (undeclared_handlers and not allow_extra_handlers):
        raise ValueError(
            "Tool surface mismatch: "
            f"missing handlers={sorted(missing_handlers)}, "
            f"undeclared handlers={sorted(undeclared_handlers)}"
        )


def dispatch(name: str, arguments: dict, *, cwd: str, cfg: Config,
             output_control=None, universal_rewrites=None,
             tool_registry: ToolRegistry | None = None) -> str:
    """Route a tool call to its implementation, truncate output.

    output_control: optional OutputControl from bash_quirks, loaded
    from the active language_quirks/<runner>.toml [output_control].
    When present, test commands get rewritten (failure-only flags) and
    output gets condensed (passing lines stripped). When None, no
    runner-specific transforms apply.
    universal_rewrites: optional list of RewriteRule from bash_quirks,
    loaded from bash_quirks/rewrites.toml. When present, noisy commands
    (pip, npm, make) get quieted. When None, no universal rewrites apply.
    """
    reg = tool_registry or build_tool_registry()
    handler = reg.handlers.get(name)
    if handler is None:
        return f"ERROR: unknown tool '{name}'"

    # Pre-execution: rewrite bash commands (quiet flags, test flags).
    if name == "bash" and (output_control is not None or universal_rewrites):
        from ..bash_quirks import rewrite_command
        rewritten_cmd = rewrite_command(
            arguments.get("cmd", ""), output_control, universal_rewrites)
        if rewritten_cmd != arguments.get("cmd", ""):
            arguments = {**arguments, "cmd": rewritten_cmd}

    try:
        result = handler(arguments, cwd, cfg)
    except (KeyError, TypeError) as e:
        return f"ERROR: bad arguments for {name}: {e}"

    if name == "bash":
        cmd = arguments.get("cmd", "")
        result = _filter_bash_output(result, cmd, cfg)
        # Post-execution: condense passing-test lines.
        if output_control is not None:
            from ..bash_quirks import condense_output
            result = condense_output(result, cmd, output_control)

    return truncate_output(result, cfg)
