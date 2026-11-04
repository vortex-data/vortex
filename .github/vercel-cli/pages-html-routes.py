# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Add the HTML URL handling the docs had on Cloudflare Pages to a Build Output API config.

Pages served `page.html` at `/page` and `dir/index.html` at `/dir/`, and
redirected every other spelling of those URLs to them. Sphinx output relies
on this: its links are relative, so a directory page must be served with
its trailing slash, and links shared while the docs were on Pages use the
URLs without `.html`.

Usage: pages-html-routes.py OUTPUT_DIR

OUTPUT_DIR is a `.vercel/output` directory holding `static/` and a
`config.json` whose routes set headers, or redirect or rewrite a request
that the HTML routes must not see, such as the Explorer's host. The routes
this script adds follow them. A directory with its own `404.html`, such as
`explorer/`, serves it for the addresses under it that match no file.
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
    # The deepest directory first, so that its page wins over its parents'.
    pages = sorted(
        (page.parent.relative_to(static).as_posix() for page in static.rglob("404.html")),
        key=lambda d: (-d.count("/") - (d != "."), d),
    )
    error_pages = [
        {
            "src": "^/.*$" if d == "." else f"^/{re.escape(d)}/.*$",
            "dest": "/404.html" if d == "." else f"/{d}/404.html",
            "status": 404,
        }
        for d in pages
    ]
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
        # An address that matches no file gets the 404 page of the nearest
        # directory that has one, with a 404 status.
        *([{"handle": "error"}, *error_pages] if error_pages else []),
    ]


def main(output: Path) -> None:
    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    routes = config.get("routes", [])
    if any("handle" in route for route in routes):
        sys.exit(f"{config_path}: routes must not start a phase ('handle') when HTML handling is added")
    config["routes"] = routes + html_routes(output / "static")
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
