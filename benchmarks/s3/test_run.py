# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run


class LauncherTests(unittest.TestCase):
    def test_ingest_keeps_schema_fields_and_separates_simulated_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.ingest.jsonl"
            rows = [
                {"kind": "query_measurement", "dataset": "tpch", "storage": "s3-simulated", "value_ns": 12},
                {"kind": "random_access_time", "dataset": "taxi-s3-simulated/correlated", "value_ns": 24},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            run.label_results([path], "abc")
            sql, random = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(sql["storage"], "s3")
            self.assertEqual(sql["dataset_variant"], "s3-simulated-abc")
            self.assertEqual(random["dataset"], "taxi-s3-simulated@s3-simulated-abc/correlated")
            self.assertNotIn("storage", random)

    def test_profile_identity_prevents_comparison_with_real_s3_or_another_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.json"
            rows = [{"name": "random-access", "value": 12}, {"name": "sql", "storage": "s3-simulated", "value": 24}]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            run.label_results([path], "abc")
            labelled = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["storage"] for row in labelled], ["s3-simulated-abc"] * 2)
            self.assertEqual([row["value"] for row in labelled], [12, 24])
            run.label_results([path], "def")
            self.assertNotEqual(json.loads(path.read_text().splitlines()[0])["storage"], labelled[0]["storage"])

    def test_child_has_dummy_credentials_and_all_clients_share_the_endpoint(self):
        with patch.dict(os.environ, {"AWS_SESSION_TOKEN": "private", "AWS_PROFILE": "private", "NO_PROXY": "internal"}):
            env = run.environment("http://127.0.0.1:1234")
        self.assertNotIn("AWS_SESSION_TOKEN", env)
        self.assertNotIn("AWS_PROFILE", env)
        self.assertEqual(env["AWS_ACCESS_KEY_ID"], "benchmark")
        for key in ("AWS_ENDPOINT", "AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3", "AWS_S3_ENDPOINT"):
            self.assertEqual(env[key], "http://127.0.0.1:1234")
        self.assertEqual(env["VORTEX_BENCH_S3_SIMULATED"], "1")
        self.assertIn("127.0.0.1", env["no_proxy"])
        self.assertIn("internal", env["NO_PROXY"])

    def test_every_family_gets_a_remote_path_and_conflicts_fail(self):
        for kind in ("sql", "random-access", "compress"):
            cmd = run.command(["benchmark", "--iterations", "3"], kind, "fixture")
            self.assertIn("s3://fixture/", " ".join(cmd))
            self.assertEqual(cmd[:3], ["benchmark", "--iterations", "3"])
        with self.assertRaisesRegex(ValueError, "conflicting"):
            run.command(["benchmark", "--opt", "remote-data-dir=s3://real/"], "sql", "fixture")
        with self.assertRaises(ValueError):
            run.command([], "sql", "fixture")


if __name__ == "__main__":
    unittest.main()
