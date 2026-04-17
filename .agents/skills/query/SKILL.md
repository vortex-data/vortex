---
name: query
description: Answer questions about the Vortex codebase or pull requests. Use when asked a question via "/query" or when the user wants to understand code, architecture, behavior, or implementation details.
---

# Vortex Query Skill

Answer questions about the Vortex project, its pull requests, and its implementation.

## Key Context

- Vortex is a Rust workspace for columnar arrays, compression encodings, file IO, and scan
  integrations.
- `vortex-array` defines the core array traits, dtype system, canonical arrays, and base
  encodings.
- `vortex-buffer` owns aligned zero-copy buffers.
- `vortex-file` and `vortex-layout` implement file and layout reading.
- `encodings/*` contains specialized compressed encodings.
- Python, Java, DuckDB, and DataFusion integrations live in their own workspace areas.

## Workflow

1. Read `AGENTS.md` and any closer scoped `AGENTS.md` before relying on conventions.
2. Use `rg` and targeted file reads to identify the relevant crate, module, and tests.
3. If the question is about a PR, inspect the diff and comments before answering.
4. If the question is about behavior, trace the implementation through public entry points,
   encoding-specific implementations, and tests.
5. Answer with concrete file paths and line numbers when they help.

## Answering Guidelines

- Separate confirmed facts from inference.
- Prefer precise code references over broad descriptions.
- Mention important uncertainty and describe what would verify it.
- Do not invent architecture. If the repository does not answer the question, say what you
  checked and what is still missing.
