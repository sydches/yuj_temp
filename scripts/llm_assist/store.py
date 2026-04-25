"""SQLite-backed session store for assistant-mode runs."""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..llm_solver._shared.paths import project_root


_SCHEMA = """
create table if not exists sessions (
    session_id text primary key,
    created_at text not null,
    updated_at text not null,
    cwd text not null,
    artifact_dir text not null,
    model text not null,
    status text not null,
    last_finish_reason text,
    prompt_text text not null,
    prompt_source text not null,
    context_mode text not null,
    system_prompt_path text,
    config_paths_json text not null
);

create table if not exists active_sessions (
    cwd text primary key,
    session_id text not null,
    updated_at text not null
)
"""


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    created_at: str
    updated_at: str
    cwd: str
    artifact_dir: str
    model: str
    status: str
    last_finish_reason: str | None
    prompt_text: str
    prompt_source: str
    context_mode: str
    system_prompt_path: str | None
    config_paths_json: str

    @property
    def artifact_path(self) -> Path:
        return Path(self.artifact_dir)

    @property
    def config_paths(self) -> list[str]:
        return list(json.loads(self.config_paths_json))


def assist_home() -> Path:
    """Return the assistant-state root."""
    raw = os.environ.get("HARNESS_ASSIST_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    return project_root() / ".llm_assist"


class SessionStore:
    """Persistent assistant session metadata."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else assist_home()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "sessions").mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "sessions.sqlite3"
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create_session(
        self,
        *,
        cwd: Path,
        model: str,
        prompt_text: str,
        prompt_source: str,
        context_mode: str,
        system_prompt_path: Path | None,
        config_paths: list[Path],
    ) -> SessionRecord:
        now = _utc_now()
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        artifact_dir = self.root / "sessions" / session_id
        record = SessionRecord(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            cwd=str(Path(cwd).resolve()),
            artifact_dir=str(artifact_dir),
            model=model,
            status="created",
            last_finish_reason=None,
            prompt_text=prompt_text,
            prompt_source=prompt_source,
            context_mode=context_mode,
            system_prompt_path=str(system_prompt_path.resolve()) if system_prompt_path else None,
            config_paths_json=json.dumps([str(Path(p).resolve()) for p in config_paths]),
        )
        with self._connect() as conn:
            conn.execute(
                """
                insert into sessions (
                    session_id, created_at, updated_at, cwd, artifact_dir, model, status,
                    last_finish_reason, prompt_text, prompt_source, context_mode,
                    system_prompt_path, config_paths_json
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.session_id,
                    record.created_at,
                    record.updated_at,
                    record.cwd,
                    record.artifact_dir,
                    record.model,
                    record.status,
                    record.last_finish_reason,
                    record.prompt_text,
                    record.prompt_source,
                    record.context_mode,
                    record.system_prompt_path,
                    record.config_paths_json,
                ),
            )
        return record

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "select * from sessions where session_id = ?",
                (session_id,),
            ).fetchone()
        return _row_to_record(row) if row is not None else None

    def list_sessions(self, *, limit: int = 50) -> list[SessionRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                select * from sessions
                order by updated_at desc
                limit ?
                """,
                (limit,),
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def set_active_session(self, cwd: Path | str, session_id: str) -> None:
        now = _utc_now()
        resolved_cwd = str(Path(cwd).resolve())
        with self._connect() as conn:
            conn.execute(
                """
                insert into active_sessions (cwd, session_id, updated_at)
                values (?, ?, ?)
                on conflict(cwd) do update set
                    session_id = excluded.session_id,
                    updated_at = excluded.updated_at
                """,
                (resolved_cwd, session_id, now),
            )

    def clear_active_session(self, cwd: Path | str, *, session_id: str | None = None) -> None:
        resolved_cwd = str(Path(cwd).resolve())
        with self._connect() as conn:
            if session_id is None:
                conn.execute(
                    "delete from active_sessions where cwd = ?",
                    (resolved_cwd,),
                )
                return
            conn.execute(
                "delete from active_sessions where cwd = ? and session_id = ?",
                (resolved_cwd, session_id),
            )

    def get_active_session_id(self, cwd: Path | str) -> str | None:
        resolved_cwd = str(Path(cwd).resolve())
        with self._connect() as conn:
            row = conn.execute(
                "select session_id from active_sessions where cwd = ?",
                (resolved_cwd,),
            ).fetchone()
        return str(row["session_id"]) if row is not None else None

    def get_active_session(self, cwd: Path | str) -> SessionRecord | None:
        session_id = self.get_active_session_id(cwd)
        if session_id is None:
            return None
        record = self.get_session(session_id)
        if record is not None:
            return record
        self.clear_active_session(cwd, session_id=session_id)
        return None

    def list_active_session_ids(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute("select session_id from active_sessions").fetchall()
        return {str(row["session_id"]) for row in rows}

    def update_session(
        self,
        session_id: str,
        *,
        status: str,
        last_finish_reason: str | None,
    ) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                update sessions
                set updated_at = ?, status = ?, last_finish_reason = ?
                where session_id = ?
                """,
                (now, status, last_finish_reason, session_id),
            )


def _row_to_record(row: sqlite3.Row) -> SessionRecord:
    return SessionRecord(
        session_id=row["session_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        cwd=row["cwd"],
        artifact_dir=row["artifact_dir"],
        model=row["model"],
        status=row["status"],
        last_finish_reason=row["last_finish_reason"],
        prompt_text=row["prompt_text"],
        prompt_source=row["prompt_source"],
        context_mode=row["context_mode"],
        system_prompt_path=row["system_prompt_path"],
        config_paths_json=row["config_paths_json"],
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["SessionRecord", "SessionStore", "assist_home"]
