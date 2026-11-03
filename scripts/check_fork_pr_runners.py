# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Check that pull requests from forks never run on self-hosted runners.

The self-hosted runners run in our AWS account and share a build cache with trusted builds. A pull
request from a fork runs code nobody has reviewed, so it must only ever run on GitHub-hosted
runners, or not at all.

A job is checked when a workflow that fork pull requests can trigger declares it, directly or
through a reusable workflow, and the job can select a self-hosted runner. Such a job must prove that
the run is trusted in one of two places:

- its `if:` condition, which then skips the job for fork pull requests, or
- the self-hosted branch of a `runs-on` expression that falls back to a GitHub-hosted runner.

Both places accept the proof as a top-level `&&` operand. The usual form is:

    (github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository)

The check also rejects the `pull_request_target` event, which runs with secrets in the context of
the fork pull request, and it exempts workflows that only start when a label is added, because only
maintainers can add labels.

Run from the repository root:

    uv run --no-project --with pyyaml python scripts/check_fork_pr_runners.py
"""

import re
import sys
from collections.abc import Iterator
from pathlib import Path

import yaml

WORKFLOWS_DIR = Path(".github/workflows")

# Every self-hosted runner label starts with this prefix.
SELF_HOSTED_MARKER = "runs-on="

# Either clause proves that the run is not a fork pull request. The first form also admits pushes
# and manual runs; the second is for jobs that only run on pull request events.
TRUSTED_RUN_CLAUSES = frozenset(
    {
        "(github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository)",
        "github.event.pull_request.head.repo.full_name == github.repository",
    }
)

# Steps that need the self-hosted runner image must check the runner, not the repository.
SELF_HOSTED_STEP_ACTION = "runs-on/action"
SELF_HOSTED_STEP_CLAUSE = "runner.environment == 'self-hosted'"

# These events run fork code with the base repository's secrets and write token.
FORBIDDEN_EVENTS = frozenset({"pull_request_target"})

# A `pull_request` trigger limited to these types only fires when a maintainer acts.
LABEL_EVENT_TYPES = frozenset({"labeled"})

# A parsed YAML mapping. Keys are usually strings, but PyYAML reads the bare `on` key as `True`.
Node = dict[object, object]


def load_workflows(root: Path) -> dict[str, Node]:
    workflows: dict[str, Node] = {}
    for path in sorted((root / WORKFLOWS_DIR).iterdir()):
        if path.suffix not in {".yml", ".yaml"}:
            continue
        loaded = yaml.safe_load(path.read_text())
        if isinstance(loaded, dict):
            workflows[path.name] = loaded
    return workflows


def triggers(workflow: Node) -> dict[str, object]:
    raw = workflow.get("on", workflow.get(True))
    if isinstance(raw, str):
        return {raw: None}
    if isinstance(raw, list):
        return {str(event): None for event in raw}
    if isinstance(raw, dict):
        return {str(event): config for event, config in raw.items()}
    return {}


def jobs(workflow: Node) -> dict[str, Node]:
    raw = workflow.get("jobs")
    if not isinstance(raw, dict):
        return {}
    return {str(name): job for name, job in raw.items() if isinstance(job, dict)}


def steps(job: Node) -> Iterator[Node]:
    """Every step of a job, including steps nested under `parallel`."""

    def walk(items: list[object]) -> Iterator[Node]:
        for item in items:
            if not isinstance(item, dict):
                continue
            nested = item.get("parallel")
            if isinstance(nested, list):
                yield from walk(nested)
            else:
                yield item

    raw = job.get("steps")
    if isinstance(raw, list):
        yield from walk(raw)


def starts_on_fork_code(workflow: Node) -> bool:
    """Whether a push to a fork pull request can start this workflow."""
    config = triggers(workflow).get("pull_request", ...)
    if config is ...:
        return False
    types = config.get("types") if isinstance(config, dict) else None
    if isinstance(types, list) and types and {str(event) for event in types} <= LABEL_EVENT_TYPES:
        return False
    return True


def called_workflows(workflow: Node) -> Iterator[str]:
    for job in jobs(workflow).values():
        uses = job.get("uses")
        if isinstance(uses, str) and uses.startswith("./.github/workflows/"):
            yield uses.removeprefix("./.github/workflows/")


def fork_reachable(workflows: dict[str, Node]) -> set[str]:
    """Workflows that a fork pull request can run, directly or as a reusable workflow."""
    pending = [name for name, workflow in workflows.items() if starts_on_fork_code(workflow)]
    reachable: set[str] = set()
    while pending:
        name = pending.pop()
        if name in reachable or name not in workflows:
            continue
        reachable.add(name)
        pending.extend(called_workflows(workflows[name]))
    return reachable


def split_top_level(expr: str, op: str) -> list[str]:
    """Split on `op` outside parentheses and single-quoted strings."""
    parts: list[str] = []
    depth = 0
    quoted = False
    start = 0
    i = 0
    while i < len(expr):
        char = expr[i]
        if quoted:
            quoted = char != "'"
        elif char == "'":
            quoted = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and expr.startswith(op, i):
            parts.append(expr[start:i])
            i += len(op)
            start = i
            continue
        i += 1
    parts.append(expr[start:])
    return parts


def normalize(expr: str) -> str:
    return " ".join(expr.split())


def unwrap(expr: str) -> str:
    expr = expr.strip()
    if expr.startswith("${{") and expr.endswith("}}"):
        expr = expr[3:-2]
    return expr.strip()


def conjuncts(condition: str) -> list[str] | None:
    """The top-level `&&` operands of a condition, or None when a top-level `||` can bypass them."""
    expr = unwrap(condition)
    if len(split_top_level(expr, "||")) != 1:
        return None
    return [normalize(part) for part in split_top_level(expr, "&&")]


def proves_trusted(condition: object) -> bool:
    if not isinstance(condition, str):
        return False
    parts = conjuncts(condition)
    return parts is not None and bool(TRUSTED_RUN_CLAUSES & set(parts))


GITHUB_HOSTED_LABEL = re.compile(r"'[^']*'")


def self_hosted_guards(runs_on: str) -> list[str] | None:
    """The `&&` operands that guard the self-hosted runner of a `runs-on` with a GitHub-hosted fallback.

    Matches `${{ <guards> && <self-hosted label> || '<github-hosted label>' }}` and returns None for
    every other shape.
    """
    expr = runs_on.strip()
    if not (expr.startswith("${{") and expr.endswith("}}")):
        return None
    branches = split_top_level(unwrap(expr), "||")
    if len(branches) != 2:
        return None
    fallback = branches[1].strip()
    if not GITHUB_HOSTED_LABEL.fullmatch(fallback) or SELF_HOSTED_MARKER in fallback:
        return None
    return [normalize(part) for part in split_top_level(branches[0], "&&")]


def check_job(workflow_name: str, job_name: str, job: Node) -> Iterator[str]:
    where = f"{workflow_name}: job `{job_name}`"
    runs_on = job.get("runs-on")
    if runs_on is None:
        return
    runs_on_text = runs_on if isinstance(runs_on, str) else yaml.safe_dump(runs_on)
    strategy = job.get("strategy")
    matrix = strategy.get("matrix") if isinstance(strategy, dict) else None
    trusted = proves_trusted(job.get("if"))

    if "matrix." in runs_on_text and not isinstance(matrix, dict):
        if not trusted:
            yield f"{where} reads its runner from a matrix the check cannot see; `if:` must prove the run is trusted"
        return
    if SELF_HOSTED_MARKER not in runs_on_text and SELF_HOSTED_MARKER not in yaml.safe_dump(matrix):
        return

    guards = self_hosted_guards(runs_on) if isinstance(runs_on, str) else None
    if guards is None:
        if not trusted:
            yield f"{where} can only run on a self-hosted runner, so its `if:` must prove the run is trusted"
        return
    if not trusted and not TRUSTED_RUN_CLAUSES & set(guards):
        yield (
            f"{where} falls back to a GitHub-hosted runner, "
            "but the self-hosted branch of `runs-on` does not prove the run is trusted"
        )
    for step in steps(job):
        uses = step.get("uses")
        if not isinstance(uses, str) or not uses.startswith(SELF_HOSTED_STEP_ACTION):
            continue
        condition = step.get("if")
        parts = conjuncts(condition) if isinstance(condition, str) else None
        if parts is None or SELF_HOSTED_STEP_CLAUSE not in parts:
            yield f"{where} runs `{SELF_HOSTED_STEP_ACTION}` in a step that is not gated on `{SELF_HOSTED_STEP_CLAUSE}`"


def check_workflows(workflows: dict[str, Node]) -> list[str]:
    problems: list[str] = []
    for name, workflow in sorted(workflows.items()):
        for event in sorted(FORBIDDEN_EVENTS & set(triggers(workflow))):
            problems.append(f"{name}: uses the `{event}` event, which runs fork code with secrets")
    for name in sorted(fork_reachable(workflows)):
        for job_name, job in jobs(workflows[name]).items():
            problems.extend(check_job(name, job_name, job))
    return problems


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    problems = check_workflows(load_workflows(root))
    for problem in problems:
        print(problem)
    if problems:
        print(f"\n{len(problems)} problem(s). The rule is documented in {Path(__file__).name}.")
        return 1
    print("Fork pull requests cannot reach self-hosted runners.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
