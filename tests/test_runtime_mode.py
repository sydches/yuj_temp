from types import SimpleNamespace

import pytest

from scripts.llm_solver.config import require_runtime_mode


def test_require_runtime_mode_rejects_wrong_entrypoint():
    cfg = SimpleNamespace(runtime_mode="measurement")
    with pytest.raises(ValueError):
        require_runtime_mode(cfg, expected="assistant", caller="scripts.llm_assist")
