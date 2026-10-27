# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for `scripts/s3-latency-analyze.py` on synthetic runs with known latency parameters."""

import csv
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/s3-latency-analyze.py"
MIB = 1 << 20
FIRST_BYTE_S = 0.02
BYTES_PER_S = 100 * MIB
COLUMNS = (
    "run,phase,scenario,kind,scan_id,file,offset,length,segments,useful_bytes,concurrency,"
    "inflight_at_start,seq,start_us,headers_us,first_byte_us,end_us,bytes,chunks,error"
).split(",")


@pytest.fixture
def analyze() -> ModuleType:
    spec = importlib.util.spec_from_file_location("s3_latency_analyze", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_run(directory: Path, run: str, age_s: int, rng: random.Random) -> None:
    """One run with serial reads, HEADs, an 8 MiB concurrency cell and one serial scan."""
    rows, t, seq = [], 0, 0

    def request(phase: str, scenario: str, kind: str, length: int, **extra: int) -> None:
        nonlocal t, seq
        first_byte = int((FIRST_BYTE_S + rng.uniform(-0.002, 0.002)) * 1e6)
        body = 0 if kind == "head" else int(length / BYTES_PER_S * 1e6)
        row: dict[str, int | str] = {column: 0 for column in COLUMNS}
        row.update(run=run, phase=phase, scenario=scenario, kind=kind, length=length, bytes=length, error="")
        row.update(seq=seq, start_us=t, headers_us=t + first_byte, first_byte_us=t + first_byte)
        row.update(end_us=t + first_byte + body, concurrency=1, segments=1, useful_bytes=length, **extra)
        rows.append(row)
        t, seq = t + first_byte + body + 10, seq + 1

    for length in [1 << k for k in range(2, 24, 2)] * 3:
        request("size_sweep", "serial", "get", length)
    for _ in range(10):
        request("suffix", "len_0", "head", 0)
    for _ in range(4):
        request("concurrency", "len_8388608_c1", "get", 8 * MIB)

    scan_start = t
    request("scan", "full", "head", 0, scan_id=1)
    request("scan", "full", "footer", 65535, scan_id=1)
    for _ in range(3):
        request("scan", "full", "segment", 4 * MIB, scan_id=1)
    scans = [
        {
            "scenario": "full",
            "scan_id": 1,
            "files": 1,
            "file_concurrency": 1,
            "per_file_concurrency": 1,
            "coalesce": True,
            "start_us": scan_start,
            "end_us": t,
            "bytes": 65535 + 12 * MIB,
        }
    ]

    directory.mkdir(parents=True)
    with (directory / "requests.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (directory / "scans.json").write_text(json.dumps(scans))
    meta = {"run_label": run, "dataset_age_s": age_s, "meta": {"instance_type": "test", "availability_zone": "az"}}
    (directory / "meta.json").write_text(json.dumps(meta))


@pytest.fixture
def results(tmp_path: Path) -> Path:
    rng = random.Random(7)
    write_run(tmp_path / "run-1", "run-1", 30, rng)
    for i in (2, 3):
        write_run(tmp_path / f"run-{i}", f"run-{i}", 600, rng)
    return tmp_path


def test_model_recovers_parameters(analyze: ModuleType, results: Path):
    report, model = analyze.analyze(*analyze.load(results))

    assert model["schema"] == "vortex-s3-latency-model/1"
    assert model["environment"]["runs"] == ["run-2", "run-3"]
    assert model["environment"]["fresh_runs_excluded"] == ["run-1"]

    request = model["request_model"]
    first_byte = dict((q, v) for q, v in request["first_byte_s"]["inverse_cdf"])
    assert first_byte[0.5] == pytest.approx(FIRST_BYTE_S, rel=0.1)
    assert request["per_connection_bytes_per_s"] == pytest.approx(BYTES_PER_S, rel=0.02)
    # These runs never saturate the network, so the cap falls back to one connection's rate.
    assert request["nic_bytes_per_s"] == pytest.approx(BYTES_PER_S, rel=0.02)

    validation = model["validation"]["full"]
    assert validation["simulated_wall_s_p50"] == pytest.approx(validation["measured_wall_s_p50"], rel=0.1)
    assert "## Latency model" in report


def test_inverse_cdf_sampling_stays_in_range(analyze: ModuleType):
    cdf = analyze.inverse_cdf([0.01, 0.02, 0.03, 0.04])
    rng = random.Random(0)
    draws = [analyze.sample(cdf, rng) for _ in range(1000)]
    assert min(draws) >= 0.01
    assert max(draws) <= 0.04


def test_cli_writes_report_and_model(results: Path, tmp_path: Path):
    report, model = tmp_path / "report.md", tmp_path / "model.json"
    subprocess.run(
        [sys.executable, str(SCRIPT), str(results), "--report", str(report), "--model", str(model)],
        check=True,
        capture_output=True,
    )
    assert report.read_text().startswith("# S3 latency report")
    assert json.loads(model.read_text())["schema"] == "vortex-s3-latency-model/1"


def test_fresh_object_window_ends_where_latency_drops(analyze: ModuleType):
    # Slow (100 ms) until the dataset is 100 s old, then warm (25 ms) for much longer, so a fixed
    # cut-off past the drop would be dominated by warm requests.
    def get(age_s: float, ttfb_s: float) -> dict[str, object]:
        return {"run": "run-1", "kind": "get", "length": 4096, "start_us": int(age_s * 1e6), "ttfb_ms": ttfb_s * 1e3}

    fresh = [get(t / 10, 0.1) for t in range(0, 1000, 5)] + [get(t / 10, 0.025) for t in range(1000, 2000)]
    fit = analyze.fit_fresh_object(fresh, {"run-1": 0}, 0.025)
    assert fit is not None
    assert fit["warm_after_age_s"] == 100
    assert fit["first_byte_multiplier_p50"] == pytest.approx(4.0)


def test_compare_reports_errors_against_real(analyze: ModuleType, results: Path):
    _, real = analyze.analyze(*analyze.load(results))
    _, sim = analyze.analyze(*analyze.load(results))
    sim["environment"]["store"] = "simulated-model"
    sim["fits"]["serial"]["p50"]["a_ms"] *= 1.1

    section, comparison = analyze.compare(real, sim)
    assert comparison["store"] == "simulated-model"
    serial = comparison["groups"]["serial"]
    assert serial["max_abs_error"] == pytest.approx(0.1)
    assert comparison["groups"]["scan wall s"]["max_abs_error"] == 0
    assert "## Simulated vs real" in section


def test_simulate_finishes_with_tiny_steps(analyze: ModuleType):
    # Transfers so fast that each step is below the clock's floating-point resolution.
    model = {
        "first_byte_s": {"inverse_cdf": [[0.0, 1e3], [1.0, 1e3]]},
        "head_s": {"inverse_cdf": [[0.0, 1e3], [1.0, 1e3]]},
        "per_connection_bytes_per_s": 1e18,
        "nic_bytes_per_s": 1e18,
    }
    tasks = [[(4, [("segment", 3), ("segment", 7), ("segment", 11)])]]
    walls, latencies = analyze.simulate(model, [tasks], [1], [0.0], random.Random(0))
    assert len(latencies) == 3
    assert walls[0] == pytest.approx(1e3)


def test_model_without_body_time_falls_back_to_serial_slope(analyze: ModuleType, results: Path):
    # Like `ThrottledStore`, deliver every body in one go: no time between first byte and end.
    rows, scans, metas = analyze.load(results)
    for r in rows:
        r["first_byte_us"] = r["end_us"]
        r["ttfb_ms"] = r["total_ms"]
        r["body_ms"] = 0.0
    _, model = analyze.analyze(rows, scans, metas)
    assert model["request_model"]["per_connection_bytes_per_s"] == pytest.approx(BYTES_PER_S, rel=0.1)
