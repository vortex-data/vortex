#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Append a version to versions.json if not already present, keeping sorted order."""

import json
import sys


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <versions.json> <version>", file=sys.stderr)
        sys.exit(1)

    path, version = sys.argv[1], sys.argv[2]

    try:
        with open(path) as f:
            versions = json.load(f)
    except FileNotFoundError:
        versions = []

    if version not in versions:
        versions.append(version)
        versions.sort(key=lambda x: list(map(int, x.split("."))))

    with open(path, "w") as f:
        json.dump(versions, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
