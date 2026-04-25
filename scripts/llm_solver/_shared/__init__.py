"""Shared primitives used across harness, server, profiles, and analysis.

Modules in this package are dependency-free w.r.t. the rest of llm_solver
(no imports of harness/, server/, profiles/, analysis/). They are the
concrete implementation of cross-cutting concerns that used to be
duplicated across the codebase.
"""
