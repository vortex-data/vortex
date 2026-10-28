# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Add Cloudflare Pages' HTML URL handling to a Build Output API config.

Pages serves `page.html` at `/page` and `dir/index.html` at `/dir/`, and
redirects every other spelling of those URLs to them. Sphinx output relies
on this: its links are relative, so a directory page must be served with
its trailing slash, and links shared since the docs moved to Pages use the
URLs without `.html`.

Usage: pages-html-routes.py OUTPUT_DIR

OUTPUT_DIR is a `.vercel/output` directory holding `static/` and a
`config.json` whose routes only set headers. The routes this script adds
follow those header routes.
"""

import json
import re
import sys
from pathlib import Path


def redirect(src: str, location: str, status: int = 308) -> dict:
    return {"src": src, "status": status, "headers": {"Location": location}}


def html_routes(static: Path) -> list[dict]:
    # Vercel serves `dir/index.html` at `/dir` as well as `/dir/`, and only a
    # route per directory can tell a directory from a page.
    directories = sorted(
        index.parent.relative_to(static).as_posix() for index in static.rglob("index.html") if index.parent != static
    )
    # Whether `/x` or `/x/` is canonical depends on whether this build has
    # `x.html` or `x/index.html`. Those redirects are temporary (307), so a
    # browser that cached one cannot loop after the docs are reorganized.
    return [
        *(redirect(f"^/{re.escape(d)}$", f"/{d}/", 307) for d in directories),
        redirect(r"^/index(?:\.html)?$", "/"),
        redirect(r"^/(.+)/index(?:\.html)?$", "/$1/"),
        redirect(r"^/(.+)\.html$", "/$1"),
        {"handle": "filesystem"},
        # Only reached when no file or directory matched.
        redirect(r"^/(.+)/$", "/$1", 307),
        {"src": r"^/(.+)$", "dest": "/$1.html", "check": True},
    ]


def main(output: Path) -> None:
    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    routes = config.get("routes", [])
    if any("handle" in route for route in routes):
        sys.exit(f"{config_path}: routes must only set headers when HTML handling is added")
    config["routes"] = routes + html_routes(output / "static")
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
