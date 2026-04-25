"""Scenario runner — run standard scenarios against a model endpoint, capture raw responses.

Usage:
    python -m llm_solver.profiles.run_scenarios --endpoint <url> --model <name> --output <dir>

Runs each scenario JSON in scenarios/ against the model, saves raw request/response
pairs to the output directory for use by the profiling process.
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import openai

from ..config import (
    get_model_default_max_tokens,
    get_server_config,
    load_config,
)

log = logging.getLogger(__name__)

SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"


def load_scenarios(scenarios_dir: Path | None = None) -> list[dict]:
    """Load all scenario JSON files from the scenarios directory."""
    d = scenarios_dir or SCENARIOS_DIR
    if not d.is_dir():
        raise FileNotFoundError(f"Scenarios directory not found: {d}")

    scenarios = []
    for path in sorted(d.glob("*.json")):
        with open(path) as f:
            scenario = json.load(f)
        scenario["_file"] = path.name
        scenarios.append(scenario)
    return scenarios


def run_scenario(
    client: openai.OpenAI,
    model: str,
    scenario: dict,
    max_tokens: int = 4096,
) -> dict:
    """Run a single scenario against the endpoint. Returns raw request + response."""
    sid = scenario.get("id", scenario.get("_file", "unknown"))
    messages = scenario["messages"]
    tools = scenario.get("tools", [])

    request_payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if tools:
        request_payload["tools"] = tools
        request_payload["tool_choice"] = "auto"

    log.debug("[%s] messages=%d tools=%d model=%s max_tokens=%d",
              sid, len(messages), len(tools), model, max_tokens)

    t0 = time.monotonic()
    try:
        resp = client.chat.completions.create(**request_payload)
        elapsed = time.monotonic() - t0

        # Extract raw response fields
        choice = resp.choices[0]
        msg = choice.message

        raw_tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                raw_tool_calls.append({
                    "id": getattr(tc, "id", None),
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

        raw_response = {
            "content": getattr(msg, "content", None),
            "tool_calls": raw_tool_calls,
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            },
        }

        log.debug("[%s] finish_reason=%s tool_calls=%d content_len=%d (%.3fs)",
                  sid, raw_response["finish_reason"], len(raw_tool_calls),
                  len(raw_response["content"] or ""), elapsed)
        log.debug("[%s] response: %s", sid, json.dumps(raw_response, indent=2))

        return {
            "scenario_id": scenario.get("id", scenario.get("_file", "unknown")),
            "description": scenario.get("description", ""),
            "request": request_payload,
            "response": raw_response,
            "elapsed_s": round(elapsed, 3),
            "error": None,
        }

    except Exception as e:
        elapsed = time.monotonic() - t0
        return {
            "scenario_id": scenario.get("id", scenario.get("_file", "unknown")),
            "description": scenario.get("description", ""),
            "request": request_payload,
            "response": None,
            "elapsed_s": round(elapsed, 3),
            "error": f"{type(e).__name__}: {e}",
        }


def evaluate_scenario(scenario: dict, result: dict) -> dict:
    """Check expect assertions against a scenario result.

    Returns dict with 'passed' (bool), 'checks' (dict of name->bool),
    and 'skipped' (bool). Skipped when no expect block or response is None.
    """
    expect = scenario.get("expect")
    if not expect:
        return {"passed": True, "checks": {}, "skipped": True}

    resp = result.get("response")
    if resp is None:
        return {"passed": False, "checks": {"response_present": False}, "skipped": False}

    checks: dict[str, bool] = {}
    tc = resp.get("tool_calls", [])

    # ── Existing fields ──────────────────────────────────────────────
    if "has_tool_calls" in expect:
        checks["has_tool_calls"] = bool(tc) == expect["has_tool_calls"]
    if "min_tool_calls" in expect:
        checks["min_tool_calls"] = len(tc) >= expect["min_tool_calls"]
    if "max_tool_calls" in expect:
        checks["max_tool_calls"] = len(tc) <= expect["max_tool_calls"]
    if "finish_reason_in" in expect:
        checks["finish_reason_in"] = resp["finish_reason"] in expect["finish_reason_in"]
    if "finish_reason" in expect:
        checks["finish_reason"] = resp["finish_reason"] == expect["finish_reason"]
    if "has_content" in expect:
        checks["has_content"] = bool(resp.get("content")) == expect["has_content"]

    # ── New fields for gauntlets ─────────────────────────────────────
    if "tool_names" in expect:
        actual = [t["function"]["name"] for t in tc]
        checks["tool_names"] = actual == expect["tool_names"]
    if "tool_names_include" in expect:
        actual = {t["function"]["name"] for t in tc}
        checks["tool_names_include"] = all(n in actual for n in expect["tool_names_include"])
    if "args_contain" in expect:
        for i, rule in enumerate(expect["args_contain"]):
            matching = [t for t in tc if t["function"]["name"] == rule["tool_name"]]
            found = False
            for t in matching:
                try:
                    args = json.loads(t["function"]["arguments"])
                except (json.JSONDecodeError, TypeError):
                    continue
                if rule["substring"] in str(args.get(rule["key"], "")):
                    found = True
                    break
            checks[f"args_contain[{i}]"] = found
    if "args_exclude" in expect:
        for i, rule in enumerate(expect["args_exclude"]):
            matching = [t for t in tc if t["function"]["name"] == rule["tool_name"]]
            violated = False
            for t in matching:
                try:
                    args = json.loads(t["function"]["arguments"])
                except (json.JSONDecodeError, TypeError):
                    continue
                if rule["substring"] in str(args.get(rule["key"], "")):
                    violated = True
                    break
            checks[f"args_exclude[{i}]"] = not violated

    return {
        "passed": all(checks.values()) if checks else True,
        "checks": checks,
        "skipped": False,
    }


def run_all(
    endpoint: str,
    model: str,
    output_dir: Path,
    max_tokens: int = 4096,
    scenarios_dir: Path | None = None,
    evaluate: bool = False,
) -> list[dict]:
    """Run all scenarios and save results. Returns list of result dicts."""
    sc = get_server_config()
    client = openai.OpenAI(
        base_url=endpoint,
        api_key=sc["api_key"],
        timeout=openai.Timeout(
            connect=sc["timeout_connect"],
            read=sc["timeout_read"],
            write=sc["timeout_read"],
            pool=sc["timeout_connect"],
        ),
    )

    scenarios = load_scenarios(scenarios_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # File handler — always DEBUG, captures everything regardless of console level
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"run_scenarios_{ts}.log"
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
    ))
    logging.getLogger().addHandler(fh)
    log.info("Log file: %s", log_path)

    try:
        results = []
        for scenario in scenarios:
            sid = scenario.get("id", scenario["_file"])
            log.info("Running scenario: %s", sid)

            result = run_scenario(client, model, scenario, max_tokens)

            if evaluate:
                ev = evaluate_scenario(scenario, result)
                result["evaluation"] = ev
                if not ev["skipped"]:
                    for check_name, passed in ev["checks"].items():
                        log.debug("  eval %s: %s: %s", sid, check_name,
                                  "PASS" if passed else "FAIL")

            results.append(result)

            # Save individual result
            out_path = output_dir / f"{sid}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

            if result["error"]:
                log.error("  ERROR: %s", result["error"])
            else:
                resp = result["response"]
                tc_count = len(resp.get("tool_calls", []))
                ev_tag = ""
                if evaluate and "evaluation" in result:
                    ev = result["evaluation"]
                    if ev["skipped"]:
                        ev_tag = " eval=SKIP"
                    else:
                        n = sum(ev["checks"].values())
                        t = len(ev["checks"])
                        ev_tag = f" eval={'PASS' if ev['passed'] else 'FAIL'} ({n}/{t})"
                log.info(
                    "  finish_reason=%s content=%s tool_calls=%d (%.1fs)%s",
                    resp["finish_reason"],
                    "yes" if resp.get("content") else "no",
                    tc_count,
                    result["elapsed_s"],
                    ev_tag,
                )

        # Save combined results
        combined_path = output_dir / "_all_results.json"
        with open(combined_path, "w") as f:
            json.dump(results, f, indent=2)

        # Print summary
        errors = sum(1 for r in results if r["error"])
        print(f"\nScenario results: {len(results)} total, {errors} errors")
        for r in results:
            status = "ERROR" if r["error"] else "OK"
            ev_tag = ""
            if evaluate and "evaluation" in r:
                ev = r["evaluation"]
                if not ev["skipped"]:
                    n = sum(ev["checks"].values())
                    t = len(ev["checks"])
                    ev_tag = f" eval={'PASS' if ev['passed'] else 'FAIL'}({n}/{t})"
            print(f"  [{status}] {r['scenario_id']}: {r['description']}{ev_tag}")

        # Log summary
        if evaluate:
            evaluated = [r for r in results if r.get("evaluation") and not r["evaluation"]["skipped"]]
            passed = sum(1 for r in evaluated if r["evaluation"]["passed"])
            failed_ids = [r["scenario_id"] for r in evaluated if not r["evaluation"]["passed"]]
            log.info("Evaluation: %d/%d passed", passed, len(evaluated))
            if failed_ids:
                log.info("Failed: %s", ", ".join(failed_ids))
        log.info("Log file: %s", log_path)

        return results
    finally:
        logging.getLogger().removeHandler(fh)
        fh.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run profiling scenarios against a model endpoint"
    )
    parser.add_argument(
        "--endpoint", default=None,
        help="OpenAI-compatible API endpoint (default: from config.toml)",
    )
    parser.add_argument(
        "--model", required=True,
        help="Model name to use in API calls",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Directory to write scenario results",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Max tokens per response (default: from config.toml)",
    )
    parser.add_argument(
        "--scenarios-dir", type=Path, default=None,
        help="Custom scenarios directory (default: built-in scenarios/)",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Evaluate expect assertions and include pass/fail in output",
    )
    parser.add_argument(
        "--log-level", choices=["debug", "info", "quiet"], default="info",
        help="Console verbosity: debug (full payloads), info (summaries), quiet (errors only)",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Extra TOML config layered on top of config.toml + config.local.toml",
    )
    args = parser.parse_args(argv)

    console_level = {"debug": logging.DEBUG, "info": logging.INFO,
                     "quiet": logging.WARNING}[args.log_level]
    logging.basicConfig(
        level=console_level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # --config is accepted for parity with other entry points; it applies to the
    # layered Config only. run_all() currently uses server settings from the
    # central config, so we trigger a load_config() call to surface any errors
    # in an extra user TOML before we start hammering the server.
    if args.config is not None:
        load_config(user_config=args.config)
    sc = get_server_config()
    results = run_all(
        endpoint=args.endpoint or sc["base_url"],
        model=args.model,
        output_dir=args.output,
        max_tokens=args.max_tokens or get_model_default_max_tokens(),
        scenarios_dir=args.scenarios_dir,
        evaluate=args.evaluate,
    )

    errors = sum(1 for r in results if r["error"])
    return 1 if errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
