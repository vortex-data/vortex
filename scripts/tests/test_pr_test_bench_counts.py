# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for the PR test and benchmark counter in `scripts/pr_test_bench_counts.py`.

Run with:

    uv run --no-project --with pytest pytest scripts/tests/test_pr_test_bench_counts.py
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "pr_test_bench_counts.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pr_test_bench_counts", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


counter = _load()

DIFF = """\
diff --git a/vortex-array/src/foo.rs b/vortex-array/src/foo.rs
index 1111111..2222222 100644
--- a/vortex-array/src/foo.rs
+++ b/vortex-array/src/foo.rs
@@ -10,0 +11,9 @@ mod tests {
+    #[test]
+    fn one() {}
+    #[tokio::test(flavor = "multi_thread")]
+    async fn two() {}
+    #[rstest]
+    #[case(1)]
+    #[case::two(2)]
+    fn three(#[case] x: i32) {}
+    // #[test] in a comment does not count
@@ -40,2 +49,0 @@ mod tests {
-    #[test]
-    fn old() {}
diff --git a/encodings/alp/benches/alp.rs b/encodings/alp/benches/alp.rs
new file mode 100644
--- /dev/null
+++ b/encodings/alp/benches/alp.rs
@@ -0,0 +1,4 @@
+#[divan::bench(args = [1, 2])]
+fn compress() {}
+    c.bench_function("a", |b| b.iter(|| ())); c.bench_with_input(id, &x, f);
+// c.bench_function("commented", f);
diff --git a/vortex-python/test/test_io.py b/vortex-python/test/test_io.py
deleted file mode 100644
--- a/vortex-python/test/test_io.py
+++ /dev/null
@@ -1,3 +0,0 @@
-def test_read():
-async def test_write():
--- not a header inside a hunk
diff --git a/vortex-python/python/vortex/io.py b/vortex-python/python/vortex/io.py
--- a/vortex-python/python/vortex/io.py
+++ b/vortex-python/python/vortex/io.py
@@ -1,0 +1,1 @@
+def test_helper_in_library_code():
"""


def test_count_diff():
    counts = counter.count_diff(DIFF)
    totals = {category: counter.total(by_group) for category, by_group in counts.items()}

    assert (totals["rust_tests"].added, totals["rust_tests"].removed) == (3, 1)
    assert (totals["rust_cases"].added, totals["rust_cases"].removed) == (2, 0)
    assert (totals["rust_benches"].added, totals["rust_benches"].removed) == (3, 0)
    assert (totals["python_tests"].added, totals["python_tests"].removed) == (0, 2)
    assert set(counts["rust_tests"]) == {"vortex-array"}
    assert set(counts["rust_benches"]) == {"encodings/alp"}
    assert set(counts["python_tests"]) == {"vortex-python"}


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("#[test]", ["rust_tests"]),
        ("#[crate::test]", ["rust_tests"]),
        ("#[rstest::rstest]", ["rust_tests"]),
        ("#[cuda_test]", ["rust_tests"]),
        ("#[test_case(1)]", []),
        ("#[cfg(test)]", []),
        ("#[bench]", ["rust_benches"]),
        ("#[benchmark]", []),
    ],
)
def test_classify_rust(line: str, expected: list[str]):
    assert counter.classify("crate/src/lib.rs", line) == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("vortex-python/test/test_io.py", True),
        ("scripts/tests/conftest.py", True),
        ("pkg/io_test.py", True),
        ("pkg/io.py", False),
    ],
)
def test_is_python_test_file(path: str, expected: bool):
    assert counter.is_python_test_file(path) is expected


def test_codspeed_total():
    assert counter.codspeed_total(None) == 0
    assert counter.codspeed_total('{"a": {"x": 2, "y": 3}, "b": {"z": 4}}') == 9


def test_render_warns_only_above_thresholds():
    counts = counter.count_diff(DIFF)

    report, over = counter.render(counts, codspeed_delta=0, tests_warn=10, benches_warn=10, codspeed_warn=10)
    assert not over
    assert "[!WARNING]" not in report
    assert report.startswith(counter.COMMENT_MARKER)
    assert "| Rust test functions | 3 | 1 | +2 |" in report

    report, over = counter.render(counts, codspeed_delta=11, tests_warn=0, benches_warn=2, codspeed_warn=10)
    assert over
    assert "[!WARNING]" in report
    assert "adds 2 tests" in report
    assert "adds 3 benchmark functions" in report
    assert "adds 11 CodSpeed benchmarks" in report
