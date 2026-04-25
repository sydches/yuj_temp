"""CLI entry point — python -m scripts.llm_solver <run_dir> [options]."""
import argparse
import logging
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .config import PROJECT_ROOT, load_config, require_runtime_mode
from .harness import collect_pending, solve_task
from .harness.context_strategies import (
    list_context_modes,
    resolve_context_class,
)
from .models import resolve_model
from .server import LlamaClient, load_profile

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Solve tasks via local LLM on llama-server"
    )
    parser.add_argument("run_dir", type=Path, help="Run directory")
    parser.add_argument("--model", "-m", help="Model name or short alias")
    parser.add_argument("--port", "-p", type=int, help="Override llama-server port")
    parser.add_argument("--config", "-c", type=Path, action="append", default=[],
                        help="User config TOML overlay (pass multiple --config "
                             "flags to layer atomic toggles; later flags win)")
    parser.add_argument("--max-sessions", type=int, help="Override max sessions per task")
    parser.add_argument("--task", type=Path, help="Single task repo dir (debug mode)")
    parser.add_argument("--prompt-file", type=Path, default=None,
                        help="Single task mode: use this file as the task prompt instead of <task>/prompt.txt")
    parser.add_argument("--prompt-text", default=None,
                        help="Single task mode: use this literal text as the task prompt instead of <task>/prompt.txt")
    parser.add_argument("--system-prompt", type=Path, default=None,
                        help="File to prepend to system prompt (e.g. a protocol definition)")
    _context_modes = list_context_modes()
    parser.add_argument(
        "--context",
        choices=_context_modes,
        default="full",
        help=f"Context strategy: {' | '.join(_context_modes)}",
    )
    parser.add_argument("--prompt-addendum", default=None,
                        help="Text appended to task prompt (experiment variant)")
    parser.add_argument("--variant-name", default=None,
                        help="Name for this experiment variant (tags results)")
    parser.add_argument("--tool-desc", default=None, choices=["minimal", "opencode"],
                        help="Tool description mode (overrides [experiment] tool_desc in config.toml)")
    parser.add_argument("--rumination-threshold", type=int, default=None,
                        help="Override rumination_nudge_threshold (%% of max_turns, e.g. 20 = 20%%)")
    parser.add_argument("--duplicate-abort", type=int, default=None,
                        help="Override duplicate_abort (per-strategy tuning)")
    parser.add_argument("--require-intent", action="store_true", default=None,
                        help="Reject silent tool calls (empty assistant content)")
    parser.add_argument("--dry-run", action="store_true", help="Print config and pending tasks, exit")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    if args.prompt_file is not None and args.prompt_text is not None:
        parser.error("--prompt-file and --prompt-text are mutually exclusive")
    if (args.prompt_file is not None or args.prompt_text is not None) and args.task is None:
        parser.error("--prompt-file/--prompt-text require --task")

    # Logging — stderr + file in run_dir
    level = logging.DEBUG if args.verbose else logging.INFO
    log_fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    log_datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=log_fmt, datefmt=log_datefmt)

    run_dir = args.run_dir.resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.model or "default"
    log_path = run_dir / f"harness_{model_tag}_{ts}.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(log_fmt, datefmt=log_datefmt))
    logging.getLogger().addHandler(fh)
    log.info("Log file: %s", log_path)

    # Build overrides from CLI flags
    overrides: dict = {}
    if args.model:
        overrides["model"] = resolve_model(args.model)
    if args.port:
        # Reuse scheme+host from [server] base_url; only the port changes.
        from urllib.parse import urlparse, urlunparse
        from .config import get_server_base_url
        parsed = urlparse(get_server_base_url())
        netloc = f"{parsed.hostname or 'localhost'}:{args.port}"
        overrides["base_url"] = urlunparse(parsed._replace(netloc=netloc))
    if args.max_sessions:
        overrides["max_sessions"] = args.max_sessions
    if args.rumination_threshold is not None:
        overrides["rumination_nudge_threshold"] = args.rumination_threshold
    if args.duplicate_abort is not None:
        overrides["duplicate_abort"] = args.duplicate_abort
    if args.require_intent is not None:
        overrides["require_intent"] = args.require_intent
    if args.prompt_addendum is not None:
        overrides["prompt_addendum"] = args.prompt_addendum
    if args.variant_name is not None:
        overrides["variant_name"] = args.variant_name
    if args.tool_desc is not None:
        overrides["tool_desc"] = args.tool_desc

    cfg = load_config(user_config=args.config, overrides=overrides)
    require_runtime_mode(cfg, expected="measurement", caller="scripts.llm_solver")

    # Echo resolved config for every run (not just --dry-run): reproducibility.
    log.info(
        "Config: model=%s ctx=%d max_turns=%d max_sessions=%d tool_desc=%s "
        "variant=%s",
        cfg.model, cfg.context_size, cfg.max_turns, cfg.max_sessions,
        cfg.tool_desc, cfg.variant_name or "(none)",
    )

    # Load model profile
    profiles_dir = PROJECT_ROOT / "profiles"
    profile = None
    if profiles_dir.is_dir():
        try:
            profile = load_profile(cfg.model, profiles_dir)
            log.info("Loaded profile: %s (inherits=%s)", profile.name, profile.inherits)
        except FileNotFoundError:
            log.info("No profile found for '%s', using legacy mode", cfg.model)

    if args.dry_run:
        print(f"Config: {cfg}")
        print(f"Profile: {profile.name if profile else 'none (legacy)'}")
        print(f"System prompt: {args.system_prompt or '(default)'}")
        print(f"Context: {args.context}")
        if args.task:
            print(f"Task: {args.task}")
        else:
            pending = collect_pending(args.run_dir)
            print(f"Pending: {len(pending)} tasks")
            for p in pending:
                print(f"  {p.name}")
        return 0

    # Wire server layer
    client = LlamaClient(cfg, profile=profile)

    # Query server for effective context size
    server_ctx = client.query_server_context()
    if server_ctx:
        effective_ctx = min(cfg.context_size, server_ctx) if cfg.context_size > 0 else server_ctx
        if effective_ctx != cfg.context_size:
            log.info(
                "Context: config=%d, server=%d → effective=%d",
                cfg.context_size, server_ctx, effective_ctx,
            )
            cfg = replace(cfg, context_size=effective_ctx)
        else:
            log.info("Context: %d (config matches server)", effective_ctx)
    else:
        log.warning("Could not query server context — using config value %d", cfg.context_size)

    # Derive char budgets from effective context size.
    # Rolling window + single tool result must fit within ~45% of the
    # token budget (rest goes to system prompt, state.json, task prompt,
    # generation headroom). At ~4 chars/token this gives the char caps.
    _ROLLING_WINDOW_RATIO = 0.45   # fraction of token budget for rolling window
    _MAX_OUTPUT_RATIO = 0.40       # fraction of token budget for single tool result
    _CHARS_PER_TOKEN = 4
    token_budget = int(cfg.context_size * cfg.context_fill_ratio)
    derived_recent = int(token_budget * _ROLLING_WINDOW_RATIO * _CHARS_PER_TOKEN)
    derived_output = int(token_budget * _MAX_OUTPUT_RATIO * _CHARS_PER_TOKEN)
    if derived_recent != cfg.recent_tool_results_chars or derived_output != cfg.max_output_chars:
        log.info(
            "Char budgets from ctx=%d (%.0f%% fill): "
            "recent_tool_results %d→%d, max_output %d→%d",
            cfg.context_size, cfg.context_fill_ratio * 100,
            cfg.recent_tool_results_chars, derived_recent,
            cfg.max_output_chars, derived_output,
        )
        cfg = replace(cfg, recent_tool_results_chars=derived_recent, max_output_chars=derived_output)

    # Context strategy (single-source registry in context_strategies).
    context_class = resolve_context_class(args.context)

    # Single task mode
    if args.task:
        initial_prompt = None
        if args.prompt_file is not None:
            initial_prompt = args.prompt_file.read_text()
        elif args.prompt_text is not None:
            initial_prompt = args.prompt_text
        ok = solve_task(
            args.task, cfg, client,
            system_prompt_file=args.system_prompt,
            context_class=context_class,
            initial_prompt=initial_prompt,
        )
        return 0 if ok else 1

    # Multi-task mode
    pending = collect_pending(run_dir)
    if not pending:
        print("No pending tasks.")
        return 0

    print(f"Solving {len(pending)} tasks (model={cfg.model})")
    results = {}
    for repo_dir in pending:
        ok = solve_task(
            repo_dir, cfg, client,
            system_prompt_file=args.system_prompt,
            context_class=context_class,
        )
        results[repo_dir.name] = ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {repo_dir.name}")

    passed = sum(1 for v in results.values() if v)
    print(f"\n{passed}/{len(results)} tasks completed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
