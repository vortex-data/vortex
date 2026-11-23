"""Find the open issues and pull requests that involve an external contributor or the fuzzer."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from .team import TEAM

EXTERNAL_AUTHOR = "external_author"
EXTERNAL_COMMENT = "external_comment"
FUZZER = "fuzzer"

# A reply to an inline review comment is itself a review, so `reviews` covers those too.
_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    items: %(connection)s(first: 50, after: $cursor, states: OPEN) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title url createdAt
        author { login __typename }
        labels(first: 50) { nodes { name } }
        assignees(first: 20) { nodes { login } }
        comments(last: 50) { nodes { at: createdAt author { login __typename } } }
        %(extra)s
      }
    }
  }
}
"""
_CONNECTIONS = {
    "issue": {"connection": "issues", "extra": ""},
    "pr": {
        "connection": "pullRequests",
        "extra": "reviews(last: 50) { nodes { at: submittedAt author { login __typename } } }",
    },
}

Query = Callable[[str, dict[str, str | int]], dict[str, Any]]


@dataclass(frozen=True)
class Item:
    """An open issue or pull request. An item with no `reasons` needs no tracking."""

    number: int
    kind: str
    title: str
    url: str
    author: str
    created_at: str
    assignees: tuple[str, ...]
    reasons: tuple[str, ...]
    # The latest activity by an external contributor, and the latest comment or review by the
    # team. Timestamps are GitHub's ISO 8601 UTC strings, which sort chronologically.
    external_at: str | None = None
    external_by: str | None = None
    team_at: str | None = None


def gh_graphql(query: str, variables: dict[str, str | int]) -> dict[str, Any]:
    """Run a GraphQL query with the `gh` CLI, which reads its token from `GH_TOKEN`."""
    args = ["gh", "api", "graphql", "-f", f"query={query}"]
    for name, value in variables.items():
        # `-F` sends a number, which a GraphQL `Int` needs; `-f` keeps the value a string.
        args += ["-F" if isinstance(value, int) else "-f", f"{name}={value}"]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"GitHub query failed: {result.stderr.strip() or result.stdout.strip()}")
    return json.loads(result.stdout)["data"]


def _login(actor: dict[str, Any] | None) -> str:
    # GitHub reports the author of a deleted account as null and shows it as `ghost`.
    return actor["login"] if actor else "ghost"


def _is_bot(actor: dict[str, Any] | None) -> bool:
    return actor is not None and (actor["__typename"] == "Bot" or actor["login"].endswith("[bot]"))


def _is_team(actor: dict[str, Any] | None) -> bool:
    return actor is not None and actor["login"].lower() in TEAM


def classify(kind: str, node: dict[str, Any]) -> Item:
    """Turn one GraphQL issue or pull request node into an `Item`."""
    author = node["author"]
    reasons = []
    external_at = external_by = team_at = None
    if not _is_bot(author) and not _is_team(author):
        reasons.append(EXTERNAL_AUTHOR)
        external_at, external_by = node["createdAt"], _login(author)

    activity = node["comments"]["nodes"] + node.get("reviews", {}).get("nodes", [])
    # A review that is not submitted yet has no timestamp and is not visible to anyone else.
    people = [a for a in activity if a["at"] and not _is_bot(a["author"])]
    people.sort(key=lambda a: a["at"])
    for entry in people:
        if _is_team(entry["author"]):
            team_at = entry["at"]
        else:
            external_at, external_by = entry["at"], _login(entry["author"])
    if people and not _is_team(people[-1]["author"]):
        reasons.append(EXTERNAL_COMMENT)

    if kind == "issue" and any(label["name"] == FUZZER for label in node["labels"]["nodes"]):
        reasons.append(FUZZER)

    return Item(
        number=node["number"],
        kind=kind,
        title=node["title"],
        url=node["url"],
        author=_login(author),
        created_at=node["createdAt"],
        assignees=tuple(sorted(a["login"] for a in node["assignees"]["nodes"])),
        reasons=tuple(reasons),
        external_at=external_at,
        external_by=external_by,
        team_at=team_at,
    )


def _nodes(kind: str, repo: str, query: Query) -> Iterator[dict[str, Any]]:
    owner, name = repo.split("/")
    variables: dict[str, str | int] = {"owner": owner, "name": name}
    while True:
        page = query(_QUERY % _CONNECTIONS[kind], variables)["repository"]["items"]
        yield from page["nodes"]
        if not page["pageInfo"]["hasNextPage"]:
            return
        variables["cursor"] = page["pageInfo"]["endCursor"]


def scan(repo: str, query: Query = gh_graphql) -> list[Item]:
    """Return every open issue and pull request in `repo`, sorted by number."""
    items = [classify(kind, node) for kind in _CONNECTIONS for node in _nodes(kind, repo, query)]
    return sorted(items, key=lambda item: item.number)


def closed_state(repo: str, number: int, query: Query = gh_graphql) -> str | None:
    """Return `closed` or `merged` for an item that is no longer open, or None if it is open."""
    owner, name = repo.split("/")
    data = query(
        "query($owner: String!, $name: String!, $number: Int!) {"
        " repository(owner: $owner, name: $name) {"
        " item: issueOrPullRequest(number: $number) {"
        " ... on Issue { state } ... on PullRequest { state } } } }",
        {"owner": owner, "name": name, "number": number},
    )
    state = data["repository"]["item"]["state"].lower()
    return None if state == "open" else state
