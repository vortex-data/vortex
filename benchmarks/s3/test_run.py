# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import os
import unittest
from unittest.mock import patch

import run


class LauncherTests(unittest.TestCase):
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
