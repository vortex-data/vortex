#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tell the benchmarks website that new results exist, then warm its default pages.

Three workflows run it through `refresh-bench-site.yml`, each once per run:

- `develop-bench.yml`, on a develop push, after every benchmark job of that push has finished
  and ingested its results with `post-ingest.py`.
- `nightly-bench.yml`, after the nightly SQL benchmarks.
- `commit-metadata.yml`, on a develop push, after it records the commit.

One revalidate per run, rather than one per benchmark job, keeps the website's Data Cache from
being flushed and refilled dozens of times an hour while benchmarks run.

The request shape is the contract in `docs/bench/CONTRACT.md` of the vortex-data/vortex.dev
repository: `POST {BENCH_SITE_BASE_URL}/api/revalidate` with
`Authorization: Bearer $BENCH_REVALIDATE_TOKEN`.

Without `BENCH_REVALIDATE_TOKEN`, the bearer token is the GitHub Actions identity token (OIDC) of
the job, with the URL of the website as its audience. The website that stores its data in Spiral
accepts it, so that website needs no secret. The job then needs `permissions: id-token: write`.

The script does nothing when the URL is unset, or when there is no token of either kind. A
revalidate that fails with a transport error, HTTP 429 or HTTP 5xx is tried again a few times; a
revalidate that still fails exits non-zero. A failed warm request only prints a warning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout of each HTTP request in seconds (default: 30).",
    )
    return parser.parse_args()


def _http(method: str, url: str, token: str | None, timeout: float) -> bytes:
    """Issue one HTTP request and return the body. Raises on any non-2xx or transport error."""
    headers = {"accept": "application/json"}
    if token is not None:
        headers["authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def has_identity_token() -> bool:
    """True when the job can ask GitHub for an identity token: it runs in GitHub Actions and has
    the `id-token: write` permission."""
    return bool(os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL") and os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN"))


def identity_token(audience: str, timeout: float = 30.0) -> str:
    """Ask GitHub for the identity token of this job, for one audience."""
    url = os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"]
    bearer = os.environ["ACTIONS_ID_TOKEN_REQUEST_TOKEN"]
    request = urllib.request.Request(
        f"{url}&audience={urllib.parse.quote(audience, safe='')}",
        headers={"authorization": f"bearer {bearer}", "accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)["value"]


def _warm_default_windows(base: str, timeout: float) -> None:
    """Best-effort warm pass: request the landing page and every group's default last-100
    bundle, so that the site refreshes them before a person asks for them. Each request is
    independent; one failure does not stop the others."""

    def warm(url: str) -> None:
        try:
            _http("GET", url, None, timeout)
        except Exception as exc:  # noqa: BLE001 -- warm is best-effort.
            print(f"warning: warm {url} failed: {exc}", file=sys.stderr)

    warm(f"{base}/")
    try:
        groups_body = _http("GET", f"{base}/api/groups", None, timeout)
        slugs = [g["slug"] for g in json.loads(groups_body).get("groups", []) if "slug" in g]
    except Exception as exc:  # noqa: BLE001 -- group discovery is best-effort.
        print(f"warning: warm group discovery failed: {exc}", file=sys.stderr)
        return
    # A whole-bundle recompute is a few seconds cold, so warm with bounded
    # concurrency rather than one slow serial pass.
    with ThreadPoolExecutor(max_workers=4) as pool:
        pool.map(lambda s: warm(f"{base}/api/group/{s}?n=100"), slugs)


# Waits between revalidate attempts. The run sends only this one revalidate, so a short outage
# or redeploy of the website must not leave the run's results behind the old cache.
_RETRY_DELAYS_S = (5, 15, 45)


def _is_transient(exc: Exception) -> bool:
    """True for a transport error, HTTP 429 or HTTP 5xx. Other HTTP errors, such as 401 for a
    wrong token, do not change when the request is sent again."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    return isinstance(exc, OSError)


def refresh_site_cache(
    base_url: str,
    token: str | Callable[[], str],
    timeout: float,
    *,
    delays: tuple[float, ...] = _RETRY_DELAYS_S,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Revalidate the site's Data Cache tag, then warm the default windows.

    `token` is the bearer token, or a function that gives a new one for each attempt: an identity
    token is valid for a short time.

    Returns False when the revalidate fails. The warm pass is then skipped: warming a cache
    that was not marked stale only keeps the old data.
    """
    base = base_url.rstrip("/")
    for attempt in range(len(delays) + 1):
        try:
            _http("POST", f"{base}/api/revalidate", token() if callable(token) else token, timeout)
            break
        except Exception as exc:  # noqa: BLE001 -- reported through the exit code.
            if attempt == len(delays) or not _is_transient(exc):
                print(f"error: cache revalidate failed: {exc}", file=sys.stderr)
                return False
            print(f"warning: cache revalidate failed ({exc}); retrying in {delays[attempt]:g}s", file=sys.stderr)
            sleep(delays[attempt])
    _warm_default_windows(base, timeout)
    return True


def main() -> int:
    args = parse_args()
    base_url = os.environ.get("BENCH_SITE_BASE_URL")
    if not base_url:
        print("BENCH_SITE_BASE_URL is not set; skipping site refresh.")
        return 0

    shared = os.environ.get("BENCH_REVALIDATE_TOKEN")
    if shared:
        return 0 if refresh_site_cache(base_url, shared, args.timeout) else 1

    if not has_identity_token():
        print("There is no BENCH_REVALIDATE_TOKEN and no identity token; skipping site refresh.")
        return 0

    audience = base_url.rstrip("/")
    return 0 if refresh_site_cache(base_url, lambda: identity_token(audience), args.timeout) else 1


if __name__ == "__main__":
    raise SystemExit(main())
