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

_BODY_LIMIT = 1000
_COMMENT_LIMIT = 600
_RECENT_LIMIT = 8

# A reply to an inline review comment is itself a review, so `reviews` covers those too.
_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    items: %(connection)s(first: 50, after: $cursor, states: OPEN) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title url createdAt body
        author { login __typename }
        labels(first: 50) { nodes { name } }
        assignees(first: 20) { nodes { login } }
        # GitHub GraphQL caps a single page at 100; more than 100 trailing bot comments
        # would push older human comments out of this window.
        comments(last: 100) { nodes { at: createdAt body author { login __typename } } }
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
        "extra": (
            "reviews(last: 100) {"
            " nodes { at: submittedAt body state author { login __typename } } }"
        ),
    },
}

Query = Callable[[str, dict[str, str | int]], dict[str, Any]]


@dataclass(frozen=True)
class Comment:
    """One human comment or submitted review on an issue or pull request."""

    at: str
    author: str
    team: bool
    kind: str
    body: str = ""
    state: str = ""


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
    body: str = ""
    recent: tuple[Comment, ...] = ()


def gh_graphql(query: str, variables: dict[str, str | int]) -> dict[str, Any]:
    """Run a GraphQL query with the `gh` CLI, which reads its token from `GH_TOKEN`."""
    args = ["gh", "api", "graphql", "-f", f"query={query}"]
    for name, value in variables.items():
        # `-F` sends a number, which a GraphQL `Int` needs; `-f` keeps the value a string.
        args += ["-F" if isinstance(value, int) else "-f", f"{name}={value}"]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        # A deleted or transferred issue returns partial `data` with a NOT_FOUND GraphQL error.
        try:
            body = json.loads(result.stdout)
        except ValueError:
            body = None
        if (
            isinstance(body, dict)
            and "data" in body
            and body.get("errors")
            and all(err.get("type") == "NOT_FOUND" for err in body["errors"])
        ):
            return body["data"]
        raise RuntimeError(f"GitHub query failed: {result.stderr.strip() or result.stdout.strip()}")
    return json.loads(result.stdout)["data"]


def _login(actor: dict[str, Any] | None) -> str:
    # GitHub reports the author of a deleted account as null and shows it as `ghost`.
    return actor["login"] if actor else "ghost"


def _is_bot(actor: dict[str, Any] | None) -> bool:
    return actor is not None and (actor["__typename"] == "Bot" or actor["login"].endswith("[bot]"))


def _is_team(actor: dict[str, Any] | None) -> bool:
    return actor is not None and actor["login"].lower() in TEAM


def _trim(text: str | None, limit: int) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def classify(kind: str, node: dict[str, Any]) -> Item:
    """Turn one GraphQL issue or pull request node into an `Item`."""
    author = node["author"]
    reasons = []
    external_at = external_by = team_at = None
    if not _is_bot(author) and not _is_team(author):
        reasons.append(EXTERNAL_AUTHOR)
        external_at, external_by = node["createdAt"], _login(author)

    activity = [("comment", entry) for entry in node["comments"]["nodes"]] + [
        ("review", entry) for entry in node.get("reviews", {}).get("nodes", [])
    ]
    # A review that is not submitted yet has no timestamp and is not visible to anyone else.
    people = [(entry_kind, a) for entry_kind, a in activity if a["at"] and not _is_bot(a["author"])]
    people.sort(key=lambda pair: pair[1]["at"])
    recent: list[Comment] = []
    for entry_kind, entry in people:
        team_member = _is_team(entry["author"])
        if team_member:
            team_at = entry["at"]
        else:
            external_at, external_by = entry["at"], _login(entry["author"])
        recent.append(
            Comment(
                at=entry["at"],
                author=_login(entry["author"]),
                team=team_member,
                kind=entry_kind,
                body=_trim(entry.get("body"), _COMMENT_LIMIT),
                state=str(entry.get("state") or ""),
            )
        )
    if people and not _is_team(people[-1][1]["author"]):
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
        body=_trim(node.get("body"), _BODY_LIMIT),
        recent=tuple(recent[-_RECENT_LIMIT:]),
    )


def _pages(repo: str, kind: str, query: Query) -> Iterator[dict[str, Any]]:
    owner, name = repo.split("/", 1)
    text = _QUERY % _CONNECTIONS[kind]
    cursor: str | None = None
    while True:
        variables: dict[str, str | int] = {"owner": owner, "name": name}
        if cursor is not None:
            variables["cursor"] = cursor
        page = query(text, variables)["repository"]["items"]
        yield from page["nodes"]
        if not page["pageInfo"]["hasNextPage"]:
            return
        cursor = page["pageInfo"]["endCursor"]


def scan(repo: str, query: Query = gh_graphql) -> list[Item]:
    """Return every open issue and pull request in `repo`, sorted by number.

    Items with no `reasons` are included so the publisher can tell them apart from closed items.
    """
    items = [classify(kind, node) for kind in _CONNECTIONS for node in _pages(repo, kind, query)]
    return sorted(items, key=lambda item: item.number)


def closed_state(repo: str, number: int, query: Query = gh_graphql) -> str | None:
    """Return `closed`, `merged`, or `deleted` for a non-open item, or None if it is open."""
    owner, name = repo.split("/", 1)
    data = query(
        """
        query($owner: String!, $name: String!, $number: Int!) {
          repository(owner: $owner, name: $name) {
            item: issueOrPullRequest(number: $number) {
              ... on Issue { state }
              ... on PullRequest { state }
            }
          }
        }
        """,
        {"owner": owner, "name": name, "number": number},
    )
    item = data["repository"]["item"]
    if item is None:
        return "deleted"
    state = item["state"].lower()
    return None if state == "open" else state
