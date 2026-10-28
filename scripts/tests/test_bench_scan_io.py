# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "bench-scan-io.py"
spec = importlib.util.spec_from_file_location("bench_scan_io", SCRIPT)
assert spec is not None and spec.loader is not None
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class ScanIoComparisonTests(unittest.TestCase):
    def test_modes_override_inherited_experiments_and_share_cpu_budget(self):
        modes = bench.configurations(
            {
                "PATH": "/bin",
                "VORTEX_SCAN_V2": "0",
                "VORTEX_IO_BACKEND": "per-core-direct",
                "VORTEX_IO_REUSE_LIVE_BUFFERS": "1",
                "VORTEX_SCAN_CONCURRENCY": "99",
                "TOKIO_WORKER_THREADS": "96",
            }
        )
        self.assertEqual(modes["v1"]["VORTEX_SCAN_V2"], "0")
        self.assertEqual(modes["v1"]["VORTEX_IO_BACKEND"], "legacy")
        self.assertEqual(modes["v2-shared1"]["VORTEX_SCAN_V2"], "1")
        self.assertEqual(modes["v2-shared1"]["VORTEX_IO_BACKEND"], "shared-1-mapped")
        for env in modes.values():
            self.assertEqual(env["PATH"], "/bin")
            self.assertEqual(env["TOKIO_WORKER_THREADS"], "32")
            self.assertEqual(env["DATAFUSION_EXECUTION_TARGET_PARTITIONS"], "32")
            self.assertNotIn("VORTEX_IO_REUSE_LIVE_BUFFERS", env)
            self.assertNotIn("VORTEX_SCAN_CONCURRENCY", env)

    def test_report_excludes_warmup_and_requires_matching_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for mode, times in [("v1", [900, 100, 200]), ("v2-shared1", [1, 50, 100])]:
                (directory / f"{mode}.jsonl").write_text(
                    json.dumps(
                        {
                            "name": "tpch_q01/datafusion:vortex",
                            "unit": "ns",
                            "all_runtimes": times,
                        }
                    )
                    + "\n"
                )
            self.assertIn("**0.500×**", bench.report(directory))
            path = directory / "v2-shared1.jsonl"
            path.write_text(path.read_text().replace("tpch_q01", "tpch_q02"))
            with self.assertRaisesRegex(ValueError, "exactly the same queries"):
                bench.report(directory)


if __name__ == "__main__":
    unittest.main()
