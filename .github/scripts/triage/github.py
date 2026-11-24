"""Read the open issues and pull requests of a repository with GitHub GraphQL."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from .team import TEAM

OPENED = "opened"
COMMENT = "comment"
REVIEW = "review"
PUSH = "push"
READY = "ready"

_BODY_LIMIT = 1000
_ATTEMPTS = 3
_TEXT_LIMIT = 600

# Each connection holds its newest 100 nodes, which is enough because the rules only look at recent
# activity. A pull request page also fetches 300 timeline nodes, so pages hold 50 items to stay well
# inside GitHub's limit of 500,000 nodes per query.
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
        comments(last: 100) { nodes { createdAt body author { login __typename } } }
        %(extra)s
      }
    }
  }
}
"""
_PR_FIELDS = """
        isDraft additions deletions
        reviews(last: 100) { nodes { submittedAt state body author { login __typename } } }
        timelineItems(
          last: 100
          itemTypes: [PULL_REQUEST_COMMIT, HEAD_REF_FORCE_PUSHED_EVENT, READY_FOR_REVIEW_EVENT]
        ) {
          nodes {
            __typename
            ... on PullRequestCommit {
              commit {
                committedDate
                author { user { login __typename } }
                committer { user { login __typename } }
              }
            }
            ... on HeadRefForcePushedEvent { createdAt actor { login __typename } }
            ... on ReadyForReviewEvent { createdAt actor { login __typename } }
          }
        }
"""
_CONNECTIONS = {"issue": ("issues", ""), "pr": ("pullRequests", _PR_FIELDS)}

Query = Callable[[str, dict[str, str | int]], dict[str, Any]]


@dataclass(frozen=True)
class Event:
    """One action by a person on an issue or pull request. Bots never produce events."""

    # GitHub's ISO 8601 UTC timestamps sort chronologically as strings.
    at: str
    actor: str
    team: bool
    kind: str
    # The review state, such as `APPROVED`, for reviews only.
    state: str = ""
    text: str = ""


@dataclass(frozen=True)
class Item:
    """An open issue or pull request with the human activity on it, oldest first."""

    number: int
    kind: str
    title: str
    url: str
    author: str
    author_bot: bool
    author_team: bool
    created_at: str
    labels: frozenset[str]
    assignees: tuple[str, ...]
    events: tuple[Event, ...]
    draft: bool = False
    changed_lines: int = 0
    body: str = ""


def gh_graphql(query: str, variables: dict[str, str | int]) -> dict[str, Any]:
    """Run a GraphQL query with the `gh` CLI, which reads its token from `GH_TOKEN`.

    A failed query is retried, because GitHub sometimes cuts a large response short, and a failed
    scan fails the whole run.
    """
    args = ["gh", "api", "graphql", "-f", f"query={query}"]
    for name, value in variables.items():
        # `-F` sends a number, which a GraphQL `Int` needs, and `-f` keeps the value a string.
        args += ["-F" if isinstance(value, int) else "-f", f"{name}={value}"]
    for attempt in range(_ATTEMPTS):
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return json.loads(result.stdout)["data"]
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
        if attempt + 1 < _ATTEMPTS:
            time.sleep(2**attempt)
    raise RuntimeError(f"GitHub query failed: {result.stderr.strip() or result.stdout.strip()}")


def _login(actor: dict[str, Any] | None) -> str:
    # GitHub reports the author of a deleted account as null and shows it as `ghost`.
    return str(actor["login"]) if actor else "ghost"


def _is_bot(actor: dict[str, Any] | None) -> bool:
    return actor is not None and (
        str(actor["__typename"]) == "Bot" or _login(actor).endswith("[bot]")
    )


def _is_team(login: str) -> bool:
    return login.lower() in TEAM


def _trim(text: str | None, limit: int) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _event(at: str | None, actor: dict[str, Any] | None, kind: str, **extra: str) -> Event | None:
    if not at or _is_bot(actor):
        return None
    login = _login(actor)
    return Event(at=at, actor=login, team=_is_team(login), kind=kind, **extra)


def _push(node: dict[str, Any], author: dict[str, Any] | None) -> Event | None:
    if node["__typename"] == "PullRequestCommit":
        commit = node["commit"]
        # A teammate who rebases a contributor's commits becomes their committer and gives them a
        # new `committedDate`, while the contributor stays their author.
        committer = (commit.get("committer") or {}).get("user")
        if committer is not None and _is_team(_login(committer)):
            return None
        # A commit email with no GitHub account has no user, so credit the pull request's author.
        user = (commit.get("author") or {}).get("user") or author
        # GitHub no longer reports when a commit was pushed, so a commit counts from its commit
        # time. A commit made before a review and pushed after it sorts before the review.
        event = _event(commit.get("committedDate"), user, PUSH)
    else:
        kind = READY if node["__typename"] == "ReadyForReviewEvent" else PUSH
        event = _event(node.get("createdAt"), node.get("actor"), kind)
    # A teammate's push is not an answer to the contributor. "Update branch" on a contributor's PR,
    # or a push to a team PR after an external comment, must not make the contributor owe a reply.
    return None if event is None or event.team else event


def parse(kind: str, node: dict[str, Any]) -> Item:
    """Turn one GraphQL issue or pull request node into an `Item`."""
    author = node["author"]
    candidates = [
        _event(node["createdAt"], author, OPENED, text=_trim(node.get("body"), _TEXT_LIMIT))
    ]
    candidates += [
        _event(c["createdAt"], c["author"], COMMENT, text=_trim(c.get("body"), _TEXT_LIMIT))
        for c in node["comments"]["nodes"]
    ]
    if kind == "pr":
        # A review that is not submitted yet has no timestamp and is not visible to anyone else.
        candidates += [
            _event(
                r["submittedAt"],
                r["author"],
                REVIEW,
                state=r["state"],
                text=_trim(r.get("body"), _TEXT_LIMIT),
            )
            for r in node["reviews"]["nodes"]
        ]
        candidates += [_push(t, author) for t in node["timelineItems"]["nodes"]]
    events = sorted((e for e in candidates if e is not None), key=lambda e: e.at)
    login = _login(author)
    return Item(
        number=node["number"],
        kind=kind,
        title=node["title"],
        url=node["url"],
        author=login,
        author_bot=_is_bot(author),
        author_team=_is_team(login),
        created_at=node["createdAt"],
        labels=frozenset(label["name"] for label in node["labels"]["nodes"]),
        assignees=tuple(sorted(a["login"] for a in node["assignees"]["nodes"])),
        events=tuple(events),
        draft=bool(node.get("isDraft")),
        changed_lines=int(node.get("additions") or 0) + int(node.get("deletions") or 0),
        body=_trim(node.get("body"), _BODY_LIMIT),
    )


def _pages(repo: str, kind: str, query: Query) -> Iterator[dict[str, Any]]:
    owner, name = repo.split("/", 1)
    connection, extra = _CONNECTIONS[kind]
    text = _QUERY % {"connection": connection, "extra": extra}
    cursor: str | None = None
    while True:
        variables: dict[str, str | int] = {"owner": owner, "name": name}
        if cursor is not None:
            variables["cursor"] = cursor
        page = query(text, variables)["repository"]["items"]
        yield from page["nodes"]
        if not page["pageInfo"]["hasNextPage"]:
            return
        cursor = str(page["pageInfo"]["endCursor"])


def scan(repo: str, query: Query = gh_graphql) -> list[Item]:
    """Return every open issue and pull request in `repo`, sorted by number."""
    items = [parse(kind, node) for kind in _CONNECTIONS for node in _pages(repo, kind, query)]
    return sorted(items, key=lambda item: item.number)


@dataclass(frozen=True)
class Ended:
    """An item that is no longer open. `state` is `closed`, `merged`, or `deleted`."""

    state: str
    title: str = ""
    url: str = ""


def closed_state(repo: str, number: int, query: Query = gh_graphql) -> Ended | None:
    """Return how an item ended, or None if it is still open."""
    owner, name = repo.split("/", 1)
    data = query(
        """
        query($owner: String!, $name: String!, $number: Int!) {
          repository(owner: $owner, name: $name) {
            item: issueOrPullRequest(number: $number) {
              ... on Issue { state title url }
              ... on PullRequest { state title url }
            }
          }
        }
        """,
        {"owner": owner, "name": name, "number": number},
    )
    item = data["repository"]["item"]
    if item is None:
        return Ended("deleted")
    state = item["state"].lower()
    return None if state == "open" else Ended(state, item["title"], item["url"])
