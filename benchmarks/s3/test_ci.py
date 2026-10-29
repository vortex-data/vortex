# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ci


class WorkflowTaskTests(unittest.TestCase):
    def test_commands_inherit_workflow_credentials_without_fetching_or_replacing_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "workflow-placeholder"}),
                patch.object(ci.subprocess, "run") as child,
                patch("builtins.print"),
            ):
                ci.execute(["benchmark", "--iterations", "5"], Path(temporary) / "benchmark.log")
                child.assert_called_once()
                positional, options = child.call_args
                self.assertEqual(positional[0], ["benchmark", "--iterations", "5"])
                self.assertNotIn("env", options)
                self.assertLess(options["timeout"], 7200)
                self.assertEqual(os.environ["AWS_ACCESS_KEY_ID"], "workflow-placeholder")

    def test_each_backend_resumes_as_one_task_and_preserves_complete_sql_coverage(self):
        args = argparse.Namespace(
            binaries=Path("/binaries"),
            output=Path("/results"),
            scale="10.0",
            iterations=5,
            engines="datafusion,duckdb,lance",
            remote_root="s3://bucket/run",
        )
        tasks = ci.tasks(args)
        expected = {
            f"sql-{suite}-{engine}-{backend}"
            for suite in ("tpch", "tpcds")
            for engine in ("datafusion", "duckdb", "lance")
            for backend in ("real", "real-repeat", "simulated")
        }
        self.assertEqual({name for name in tasks if name.startswith("sql-")}, expected)
        self.assertEqual(len(tasks), 32)
        for name in sorted(expected):
            with self.subTest(task=name), patch.object(ci, "execute") as execute:
                tasks[name]()
                execute.assert_called_once()
                command = execute.call_args.args[0]
                self.assertNotIn("--queries", command)
                self.assertEqual(command[command.index("--iterations") + 1], "5")
                if name.endswith("-simulated"):
                    self.assertEqual(command[0], sys.executable)
                    self.assertIn("benchmarks/s3/run.py", command)
                else:
                    self.assertTrue(command[0].endswith("-bench"))
                    self.assertIn("remote-data-dir=s3://bucket/run/", " ".join(command))


if __name__ == "__main__":
    unittest.main()
