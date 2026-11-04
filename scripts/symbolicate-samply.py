#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Resolve one ELF's Samply function names without changing samples or stack tables."""

import argparse
import gzip
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    notes = subprocess.check_output(["readelf", "-n", str(args.binary)], text=True)
    matched = re.search(r"Build ID: ([0-9a-fA-F]+)", notes)
    if matched is None:
        raise ValueError("The ELF has no GNU build ID")
    build_id = matched.group(1).lower()
    with gzip.open(args.profile, "rt", encoding="utf-8") as source:
        profile = json.load(source)
    libraries = {
        index for index, library in enumerate(profile["libs"]) if (library.get("codeId") or "").lower() == build_id
    }
    if not libraries:
        raise ValueError(f"The profile does not contain ELF build ID {build_id}")
    entries: list[tuple[dict[str, Any], int, int]] = []
    addresses: set[int] = set()
    for thread in profile["threads"]:
        functions = thread["funcTable"]
        resources = thread["resourceTable"]
        strings = thread["stringArray"]
        for index, resource in enumerate(functions["resource"]):
            if resource is None or resource < 0 or resources["lib"][resource] not in libraries:
                continue
            name = strings[functions["name"][index]]
            if re.fullmatch(r"0x[0-9a-fA-F]+", name):
                address = int(name, 16)
                entries.append((thread, index, address))
                addresses.add(address)
    symbols: dict[int, str] = {}
    if addresses:
        symbolizer = shutil.which("llvm-symbolizer")
        if symbolizer is None:
            candidates = {
                int(path.name.rsplit("-", 1)[1]): path for path in Path("/usr/bin").glob("llvm-symbolizer-[0-9]*")
            }
            if not candidates:
                raise ValueError("llvm-symbolizer is required")
            symbolizer = str(candidates[max(candidates)])
        result = subprocess.run(
            [symbolizer, f"--obj={args.binary}", "--no-inlines", "--output-style=JSON"],
            input="".join(f"0x{address:x}\n" for address in sorted(addresses)),
            text=True,
            capture_output=True,
            check=True,
        )
        for line in result.stdout.splitlines():
            value = json.loads(line)
            for item in value if isinstance(value, list) else [value]:
                names = item.get("Symbol", [])
                if names and names[0].get("FunctionName"):
                    symbols[int(item["Address"], 16)] = names[0]["FunctionName"]
        if not symbols:
            raise ValueError("No sampled addresses could be resolved")
    for thread, index, address in entries:
        if address in symbols:
            thread["funcTable"]["name"][index] = len(thread["stringArray"])
            thread["stringArray"].append(symbols[address])
    profile["meta"]["nativeSymbolication"] = {
        "buildId": build_id,
        "resolvedAddresses": len(symbols),
        "sampledAddresses": len(addresses),
    }
    with gzip.open(args.output, "wt", encoding="utf-8") as output:
        json.dump(profile, output, separators=(",", ":"))
    print(f"Resolved {len(symbols)}/{len(addresses)} native addresses for {build_id}")


if __name__ == "__main__":
    main()
