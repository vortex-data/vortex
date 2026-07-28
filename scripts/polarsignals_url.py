# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Build Polar Signals Cloud links for profiled CI benchmark runs."""

from __future__ import annotations

import argparse
import os
import time
from urllib.parse import urlencode

DEFAULT_CLOUD_HOSTNAME = "cloud.polarsignals.com"
DEFAULT_PROFILE_METRIC = "parca_agent:samples:count:cpu:nanoseconds:delta"
DEFAULT_RELATIVE_WINDOW = "relative:hour|6"


def parse_labels(labels: str) -> dict[str, str]:
    """Parse the semicolon-delimited label format used by the profiling action."""

    parsed = {}
    for raw_label in labels.split(";"):
        key, separator, value = raw_label.strip().partition("=")
        if key and separator:
            parsed[key.strip()] = value.strip()
    return parsed


def _quote_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def label_selector(labels: dict[str, str]) -> str:
    """Render labels as a Prometheus-style selector."""

    if not labels:
        return ""
    selectors = [f'{key}="{_quote_label_value(value)}"' for key, value in labels.items()]
    return "{" + ",".join(selectors) + "}"


def _timestamp_ms(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def build_query_url(
    project_uuid: str,
    labels: dict[str, str],
    *,
    cloud_hostname: str = DEFAULT_CLOUD_HOSTNAME,
    profile_metric: str = DEFAULT_PROFILE_METRIC,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> str:
    """Build a Polar Signals Cloud query URL for one benchmark profile."""

    expression = f"{profile_metric}{label_selector(labels)}"
    params = {
        "query_browser_mode": "simple",
        "expression_a": expression,
        "sum_by_a": "comm",
        "selection_a": expression,
    }

    if start_ms is not None:
        if end_ms is None:
            end_ms = int(time.time() * 1000)
        duration_seconds = max(0, (end_ms - start_ms) // 1000)
        params.update(
            {
                "step_count": str(min(max(duration_seconds // 10, 50), 500)),
                "from_a": str(start_ms),
                "to_a": str(end_ms),
                "time_selection_a": f"absolute:{start_ms}-{end_ms}",
                "merge_from_a": str(start_ms * 1_000_000),
                "merge_to_a": str(end_ms * 1_000_000),
            }
        )
    else:
        params.update(
            {
                "step_count": "50",
                "time_selection_a": DEFAULT_RELATIVE_WINDOW,
            }
        )

    return f"https://{cloud_hostname}/projects/{project_uuid}?{urlencode(params)}"


def build_query_url_from_env(*, now_ms: int | None = None) -> str | None:
    """Build a profiling URL from POLARSIGNALS_* environment variables."""

    project_uuid = os.environ.get("POLARSIGNALS_PROJECT_UUID")
    raw_labels = os.environ.get("POLARSIGNALS_LABELS", "")
    labels = parse_labels(raw_labels)
    if not project_uuid or not labels:
        return None

    start_ms = _timestamp_ms(os.environ.get("POLARSIGNALS_START_MS"))
    end_ms = _timestamp_ms(os.environ.get("POLARSIGNALS_END_MS"))
    if end_ms is None:
        end_ms = now_ms

    return build_query_url(
        project_uuid,
        labels,
        cloud_hostname=os.environ.get("POLARSIGNALS_CLOUD_HOSTNAME", DEFAULT_CLOUD_HOSTNAME),
        profile_metric=os.environ.get("POLARSIGNALS_PROFILE_METRIC", DEFAULT_PROFILE_METRIC),
        start_ms=start_ms,
        end_ms=end_ms,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--github-output",
        metavar="NAME",
        help="also write the generated URL to this GitHub Actions output",
    )
    args = parser.parse_args()

    url = build_query_url_from_env()
    if url is None:
        return

    print(url)
    if args.github_output:
        output_path = os.environ.get("GITHUB_OUTPUT")
        if output_path:
            with open(output_path, "a", encoding="utf-8") as output:
                print(f"{args.github_output}={url}", file=output)


if __name__ == "__main__":
    main()
