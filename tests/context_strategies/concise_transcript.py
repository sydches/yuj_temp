"""ConciseTranscript — wo_yuj concise baseline.

Action-oriented replacement for the original concise experiment. Uses the
shared WorkingSet baseline and generic section labels.
"""
from __future__ import annotations

from collections.abc import Callable

from ..context import chars_div_4
from ._working_set_baseline import WorkingSetBaselineContext


class ConciseTranscript(WorkingSetBaselineContext):
    """Generic-label concise mode for the wo_yuj arm."""

    def __init__(
        self,
        original_prompt: str,
        *,
        cwd: str,
        recent_results_chars: int,
        trace_reasoning_chars: int,
        min_turns: int,
        args_summary_chars: int,
        inspect_repeat_threshold: int = 0,
        token_estimator: Callable[[list[dict]], int] = chars_div_4,
    ):
        super().__init__(
            cwd=cwd,
            original_prompt=original_prompt,
            recent_results_chars=recent_results_chars,
            trace_reasoning_chars=trace_reasoning_chars,
            min_turns=min_turns,
            args_summary_chars=args_summary_chars,
            style="generic",
            inspect_repeat_threshold=inspect_repeat_threshold,
            savings_mechanism="concise_transcript",
            token_estimator=token_estimator,
        )


CONTEXT_MODE = "concise"
CONTEXT_CLASS = ConciseTranscript
