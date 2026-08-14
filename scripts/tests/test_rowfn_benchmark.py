# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "rowfn_benchmark.py"


def load_module():
    spec = importlib.util.spec_from_file_location("rowfn_benchmark", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_divan(path: Path, rows: list[str]) -> None:
    path.write_text(
        "\n".join(
            [
                "Timer precision: 20 ns",
                "bench          fastest  │ slowest │ median   │ mean     │ samples │ iters",
                *rows,
                "",
            ]
        ),
        encoding="utf-8",
    )


class RowFnBenchmarkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_parse_divan_preserves_nested_benchmark_names_and_converts_units(self) -> None:
        output = self.directory / "result.txt"
        write_divan(
            output,
            [
                "├─ non_nullable           │         │          │          │         │",
                "│  ├─ 2       17.18 µs    │ 18 µs   │ 17.33 µs │ 17.4 µs │ 100     │ 100",
                "│  ╰─ 32      6.709 µs    │ 8 µs    │ 6.829 µs │ 7 µs    │ 100     │ 100",
                "╰─ nullable               │         │          │          │         │",
                "   ╰─ 2       799.7 ns    │ 1 µs    │ 979.7 ns │ 986 ns │ 100     │ 100",
            ],
        )

        self.assertEqual(
            self.module.parse_divan(output),
            {
                "non_nullable/2": 17_330.0,
                "non_nullable/32": 6_829.0,
                "nullable/2": 979.7,
            },
        )

    def test_summarize_writes_paired_ratios_and_slowest_first_markdown(self) -> None:
        measured = self.directory / "measured"
        measured.mkdir()
        results = {
            "numeric-baseline-1.txt": ["├─ add  10 ns │ 10 ns │ 10 ns │ 10 ns │ 100 │ 100"],
            "numeric-candidate-1.txt": ["├─ add  12 ns │ 12 ns │ 12 ns │ 12 ns │ 100 │ 100"],
            "numeric-baseline-2.txt": ["├─ add  20 ns │ 20 ns │ 20 ns │ 20 ns │ 100 │ 100"],
            "numeric-candidate-2.txt": ["├─ add  18 ns │ 18 ns │ 18 ns │ 18 ns │ 100 │ 100"],
            "numeric-baseline-3.txt": ["├─ add  10 ns │ 10 ns │ 10 ns │ 10 ns │ 100 │ 100"],
            "numeric-candidate-3.txt": ["├─ add  11 ns │ 11 ns │ 11 ns │ 11 ns │ 100 │ 100"],
            "numeric-baseline-4.txt": ["├─ mul  10 ns │ 10 ns │ 10 ns │ 10 ns │ 100 │ 100"],
            "numeric-candidate-4.txt": ["├─ mul   9 ns │  9 ns │  9 ns │  9 ns │ 100 │ 100"],
        }
        for name, rows in results.items():
            write_divan(measured / name, rows)

        summaries = self.module.summarize(self.module.read_measurements(measured))
        self.module.write_summary(self.directory, summaries)

        add = next(summary for summary in summaries if summary.benchmark == "add")
        self.assertEqual(add.pairs, 3)
        self.assertAlmostEqual(add.median_ratio, 1.1)
        self.assertAlmostEqual(add.ratio_mad, 0.1)

        csv_output = (self.directory / "ratios.csv").read_text(encoding="utf-8")
        markdown = (self.directory / "summary.md").read_text(encoding="utf-8")
        self.assertIn("suite,benchmark,pairs", csv_output)
        self.assertLess(markdown.index("`add`"), markdown.index("`mul`"))

    def test_summarize_rejects_unpaired_measurements(self) -> None:
        measured = self.directory / "measured"
        measured.mkdir()
        write_divan(
            measured / "numeric-baseline-1.txt",
            ["╰─ add  10 ns │ 10 ns │ 10 ns │ 10 ns │ 100 │ 100"],
        )

        with self.assertRaisesRegex(ValueError, "unpaired benchmark measurements"):
            self.module.summarize(self.module.read_measurements(measured))

    def test_summarize_excludes_and_reports_revision_only_benchmarks(self) -> None:
        measured = self.directory / "measured"
        measured.mkdir()
        results = {
            "numeric-baseline-1.txt": ["├─ add  10 ns │ 10 ns │ 10 ns │ 10 ns │ 100 │ 100"],
            "numeric-candidate-1.txt": [
                "├─ add       11 ns │ 11 ns │ 11 ns │ 11 ns │ 100 │ 100",
                "╰─ candidate  5 ns │  5 ns │  5 ns │  5 ns │ 100 │ 100",
            ],
        }
        for name, rows in results.items():
            write_divan(measured / name, rows)

        measurements = self.module.read_measurements(measured)
        summaries = self.module.summarize(measurements)
        differences = self.module.inventory_differences(measurements)
        self.module.write_summary(self.directory, summaries, differences)

        self.assertEqual([summary.benchmark for summary in summaries], ["add"])
        self.assertEqual(differences, [("numeric", "candidate only", "candidate")])
        markdown = (self.directory / "summary.md").read_text(encoding="utf-8")
        self.assertIn("`numeric/candidate`: candidate only.", markdown)

    def test_build_record_validates_identity_and_executable(self) -> None:
        target = self.directory / "target"
        target.mkdir()
        binary = target / "binary_ops-123"
        binary.write_bytes(b"first binary")
        metadata = target / "rowfn-benchmark-build.json"
        identity = {
            "settings": {"codegen_units": "1", "lto": "fat"},
            "toolchain": {"rustc": "rustc 1.97.1", "cargo": "cargo 1.97.1"},
            "revision": {"head": "abc123", "dirty_state_sha256": "clean"},
        }
        arguments = SimpleNamespace(
            output=str(metadata),
            worktree=str(self.directory),
            target=str(target),
            setting=["codegen_units=1", "lto=fat"],
            binary=[f"numeric={binary}"],
        )

        with mock.patch.object(self.module, "build_identity", return_value=identity):
            self.module.write_build_record(arguments)

        validation = SimpleNamespace(
            metadata=str(metadata),
            worktree=str(self.directory),
            target=str(target),
            setting=["codegen_units=1", "lto=fat"],
            suite=["numeric"],
        )
        with mock.patch.object(self.module, "build_identity", return_value=identity):
            self.assertEqual(
                self.module.validated_build_binaries(validation),
                {"numeric": str(binary.resolve())},
            )

        changed_identities = {
            "settings": {**identity, "settings": {"codegen_units": "16", "lto": "false"}},
            "toolchain": {
                **identity,
                "toolchain": {"rustc": "rustc 1.98.0", "cargo": "cargo 1.98.0"},
            },
            "revision": {
                **identity,
                "revision": {"head": "def456", "dirty_state_sha256": "changed"},
            },
        }
        for field, changed_identity in changed_identities.items():
            with (
                self.subTest(field=field),
                mock.patch.object(self.module, "build_identity", return_value=changed_identity),
                self.assertRaisesRegex(ValueError, f"{field} changed"),
            ):
                self.module.validated_build_binaries(validation)

        binary.write_bytes(b"second binary")
        with (
            mock.patch.object(self.module, "build_identity", return_value=identity),
            self.assertRaisesRegex(ValueError, "binary changed"),
        ):
            self.module.validated_build_binaries(validation)


if __name__ == "__main__":
    unittest.main()
