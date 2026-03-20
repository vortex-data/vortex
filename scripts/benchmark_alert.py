# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Read alert ndjson from stdin/file, push to incident.io / GitHub / terminal.

    # Local — pretty-print:
    uv run scripts/benchmark_check.py ... | uv run scripts/benchmark_alert.py

    # CI — send to incident.io:
    uv run scripts/benchmark_alert.py --webhook-url "$URL" --alerts-file alerts.ndjson
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Alert:
    benchmark_id: str
    series: str
    commit_id: str
    check: str
    current: float
    expected: float
    sigma: float
    message: str


def read_alerts(stream) -> list[Alert]:
    alerts = []
    for line in stream:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        alerts.append(Alert(**obj))
    return alerts


def group_by_suite(alerts: list[Alert]) -> dict[str, list[Alert]]:
    groups: dict[str, list[Alert]] = {}
    for a in alerts:
        groups.setdefault(a.benchmark_id, []).append(a)
    return dict(sorted(groups.items()))


# ============================================================================
# EMITTERS
# ============================================================================

def emit_terminal(alerts: list[Alert]) -> None:
    if not alerts:
        print("No regressions detected.", file=sys.stderr)
        return
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"REGRESSIONS ({len(alerts)} total)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    for suite, suite_alerts in group_by_suite(alerts).items():
        print(f"\n  {suite}:", file=sys.stderr)
        for a in suite_alerts:
            print(f"    [{a.check.upper()}] {a.series}", file=sys.stderr)
            print(f"      {a.message}", file=sys.stderr)
    print(file=sys.stderr)


def emit_github(alerts: list[Alert]) -> None:
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"alert_count={len(alerts)}\n")
            f.write(f"has_alerts={'true' if alerts else 'false'}\n")

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return

    with open(summary_file, "a") as f:
        if not alerts:
            f.write("## Benchmark Monitor\nNo regressions detected.\n")
            return
        f.write(f"## Benchmark Regressions ({len(alerts)} total)\n\n")
        for suite, suite_alerts in group_by_suite(alerts).items():
            f.write(f"### {suite}\n\n")
            f.write("| Series | Check | σ | Current | Expected |\n")
            f.write("|--------|-------|---|---------|----------|\n")
            for a in suite_alerts:
                f.write(f"| `{a.series}` | {a.check} | {a.sigma:+.2f} "
                        f"| {a.current:.1f} | {a.expected:.1f} |\n")
            f.write("\n")


def emit_incident_io(alerts: list[Alert], webhook_url: str) -> None:
    for alert in alerts:
        payload = json.dumps({
            "title": f"Benchmark regression: {alert.series}",
            "description": alert.message,
            "deduplication_key": (
                f"bench-{alert.benchmark_id}-{alert.series}-{alert.commit_id[:12]}"
            ),
            "metadata": {
                "benchmark_suite": alert.benchmark_id,
                "series": alert.series,
                "commit": alert.commit_id,
                "check": alert.check,
                "current_value": alert.current,
                "expected_value": alert.expected,
                "deviation_sigma": alert.sigma,
            },
        }).encode()
        req = urllib.request.Request(
            webhook_url, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                print(f"  incident.io: {alert.series} → HTTP {resp.status}",
                      file=sys.stderr)
        except Exception as e:
            print(f"  incident.io: {alert.series} FAILED: {e}",
                  file=sys.stderr)


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Read alert ndjson, emit to incident.io / GitHub / terminal.",
    )
    p.add_argument("--webhook-url", type=str, default=None,
                   help="incident.io webhook URL (omit for local use)")
    p.add_argument("--alerts-file", type=str, default=None,
                   help="Read alerts from file instead of stdin")
    args = p.parse_args()

    if args.alerts_file:
        with open(args.alerts_file) as f:
            alerts = read_alerts(f)
    else:
        alerts = read_alerts(sys.stdin)

    emit_terminal(alerts)

    if os.environ.get("GITHUB_ACTIONS"):
        emit_github(alerts)

    if args.webhook_url and alerts:
        print(f"Sending {len(alerts)} alert(s) to incident.io...", file=sys.stderr)
        emit_incident_io(alerts, args.webhook_url)

    sys.exit(1 if alerts else 0)


if __name__ == "__main__":
    main()
