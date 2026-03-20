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
    noise_assessment: dict | None = None


def read_alerts(stream) -> list[Alert]:
    alerts = []
    for line in stream:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        alerts.append(Alert(**obj))
    return alerts


def is_noise(alert: Alert) -> bool:
    """Return True if the alert was classified as environmental noise."""
    if alert.noise_assessment is None:
        return False
    return alert.noise_assessment.get("classification") in (
        "engine_noise", "global_noise", "dep_upgrade",
    )


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
    real = [a for a in alerts if not is_noise(a)]
    noisy = [a for a in alerts if is_noise(a)]
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"REGRESSIONS ({len(real)} real, {len(noisy)} noise-classified, "
          f"{len(alerts)} total)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    for suite, suite_alerts in group_by_suite(alerts).items():
        print(f"\n  {suite}:", file=sys.stderr)
        for a in suite_alerts:
            noise_tag = ""
            if is_noise(a):
                classification = a.noise_assessment["classification"]
                noise_tag = f" [NOISE:{classification}]"
            print(f"    [{a.check.upper()}] {a.series}{noise_tag}", file=sys.stderr)
            print(f"      {a.message}", file=sys.stderr)
            if a.noise_assessment and "message" in a.noise_assessment:
                print(f"      ↳ {a.noise_assessment['message']}", file=sys.stderr)
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
        real = [a for a in alerts if not is_noise(a)]
        noisy = [a for a in alerts if is_noise(a)]
        f.write(f"## Benchmark Regressions ({len(real)} real, "
                f"{len(noisy)} noise-classified)\n\n")
        for suite, suite_alerts in group_by_suite(alerts).items():
            f.write(f"### {suite}\n\n")
            f.write("| Series | Check | σ | Current | Expected | Noise? |\n")
            f.write("|--------|-------|---|---------|----------|--------|\n")
            for a in suite_alerts:
                noise_col = ""
                if is_noise(a):
                    noise_col = f"⚠️ {a.noise_assessment['classification']}"
                elif a.noise_assessment and a.noise_assessment.get("classification") == "vortex_only":
                    noise_col = "✅ confirmed"
                f.write(f"| `{a.series}` | {a.check} | {a.sigma:+.2f} "
                        f"| {a.current:.1f} | {a.expected:.1f} | {noise_col} |\n")
            f.write("\n")


def emit_incident_io(alerts: list[Alert], webhook_url: str) -> None:
    skipped = sum(1 for a in alerts if is_noise(a))
    if skipped:
        print(f"  incident.io: skipping {skipped} noise-classified alert(s)",
              file=sys.stderr)
    for alert in alerts:
        if is_noise(alert):
            continue
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
