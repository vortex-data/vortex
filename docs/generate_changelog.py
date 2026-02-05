#!/usr/bin/env python3
"""Fetch the last N GitHub releases and write per-release changelog pages for Sphinx."""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = "vortex-data/vortex"
NUM_RELEASES = 5
CHANGELOG_DIR = Path(__file__).parent / "project" / "changelog"


def fetch_releases():
    # List releases (body is not available in list output).
    result = subprocess.run(
        [
            "gh",
            "release",
            "list",
            "--repo",
            REPO,
            "--limit",
            "20",
            "--json",
            "tagName,isDraft,isPrerelease,publishedAt",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    releases = json.loads(result.stdout)
    releases = [r for r in releases if not r["isDraft"] and not r["isPrerelease"]][:NUM_RELEASES]

    # Fetch body for each release individually.
    for r in releases:
        result = subprocess.run(
            ["gh", "release", "view", r["tagName"], "--repo", REPO, "--json", "body", "--jq", ".body"],
            capture_output=True,
            text=True,
            check=True,
        )
        r["body"] = result.stdout

    return releases


def linkify(body):
    """Convert PR/issue numbers and GitHub usernames to markdown links."""
    # (#1234) -> ([#1234](https://github.com/REPO/pull/1234))
    body = re.sub(
        r"\(#(\d+)\)",
        rf"([#\1](https://github.com/{REPO}/pull/\1))",
        body,
    )
    # @username -> [@username](https://github.com/username)
    # But skip @[bot](url) which release-drafter already links.
    body = re.sub(
        r"(?<!\[)@([a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?)(?!\])",
        r"[@\1](https://github.com/\1)",
        body,
    )
    return body


def write_release_page(release):
    tag = release["tagName"]
    date = datetime.fromisoformat(release["publishedAt"].replace("Z", "+00:00")).strftime("%Y-%m-%d")
    body = linkify(release["body"].replace("\r\n", "\n").strip())

    content = f"""# {tag}

Released {date} — [GitHub Release](https://github.com/{REPO}/releases/tag/{tag})

{body}
"""

    filename = f"v{tag}.md"
    (CHANGELOG_DIR / filename).write_text(content)
    return filename


def write_index(filenames):
    toctree_entries = "\n".join(filenames)

    content = f"""# Changelog

For older releases, see the full [release history on GitHub](https://github.com/{REPO}/releases).

```{{toctree}}
---
maxdepth: 1
---

{toctree_entries}
```
"""

    (CHANGELOG_DIR / "index.md").write_text(content)


def main():
    releases = fetch_releases()
    if not releases:
        print("warning: no releases found, skipping changelog generation", file=sys.stderr)
        return

    CHANGELOG_DIR.mkdir(parents=True, exist_ok=True)

    filenames = []
    for r in releases:
        filenames.append(write_release_page(r))

    write_index(filenames)
    print(f"Wrote {len(releases)} release pages to {CHANGELOG_DIR}")


if __name__ == "__main__":
    main()
