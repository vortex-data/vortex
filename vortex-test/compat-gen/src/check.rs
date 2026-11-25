// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::collections::BTreeSet;
use std::fmt::Display;
use std::fs;
use std::io;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;
use std::path::Path;

use clap::ValueEnum;
use serde::Serialize;
use vortex_array::VortexSessionExecute;
use vortex_array::assert_arrays_eq;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::adapter;
use crate::fixtures::Fixture;
use crate::fixtures::all_fixtures;

/// How to handle mismatches between directory and known fixtures.
#[derive(Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Mode {
    /// Directory must match fixtures exactly.
    Exact,
    /// Directory may have extra files (skip), but all known must be present.
    Subset,
    /// Directory may be missing files (skip), but no unknown files allowed.
    Superset,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct CheckResult {
    passed: Vec<String>,
    failed: Vec<FailedFixture>,
    skipped: Vec<String>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct FailedFixture {
    name: String,
    error: String,
}

/// Check `.vortex` files in `dir` against in-memory fixtures.
///
/// For each known fixture, generates fresh files in a temp directory via
/// `fixture.write_expected(tmp_dir)`, then reads both the stored file and fresh file,
/// validates the stored layout tree, and compares arrays without stopping at the first failure.
///
/// Prints JSON result to stdout, human-readable progress to stderr.
/// Returns error if any fixture failed or if mode constraints are violated.
pub fn check(dir: &Path, mode: Mode, exclude: &[String]) -> VortexResult<()> {
    let fixtures: Vec<_> = all_fixtures()
        .into_iter()
        .filter(|fixture| {
            let name = fixture.name();
            !exclude.iter().any(|pat| name.contains(pat.as_str()))
        })
        .collect();

    if !exclude.is_empty() {
        eprintln!("excluding: {}", exclude.join(", "));
    }

    let result = check_fixtures(dir, mode, &fixtures)?;

    // Print JSON result to stdout.
    let json = serde_json::to_string_pretty(&result)
        .map_err(|err| vortex_err!("failed to serialize result: {err}"))?;
    println!("{json}");

    // Summary to stderr.
    eprintln!(
        "\nresult: {} passed, {} failed, {} skipped",
        result.passed.len(),
        result.failed.len(),
        result.skipped.len()
    );

    if !result.failed.is_empty() {
        for failure in &result.failed {
            eprintln!("  FAIL {}: {}", failure.name, failure.error);
        }
        vortex_bail!("{} fixture(s) failed", result.failed.len());
    }

    Ok(())
}

fn check_fixtures(
    dir: &Path,
    mode: Mode,
    fixtures: &[Box<dyn Fixture>],
) -> VortexResult<CheckResult> {
    let mut dir_files = collect_vortex_files(dir)
        .map_err(|err| vortex_err!("failed to read dir {}: {err}", dir.display()))?;
    dir_files.sort();

    let session = VortexSession::empty();
    let tmp_dir =
        tempfile::tempdir().map_err(|err| vortex_err!("failed to create temp dir: {err}"))?;

    let (failed_expected, expected_failed_names) =
        generate_expected_fixtures(tmp_dir.path(), fixtures, &session);

    let mut fresh_files = collect_vortex_files(tmp_dir.path())
        .map_err(|err| vortex_err!("failed to read tmp dir: {err}"))?;
    fresh_files.retain(|name| !expected_failed_names.contains(name));
    fresh_files.sort();

    let mut result = CheckResult {
        passed: Vec::new(),
        failed: failed_expected,
        skipped: Vec::new(),
    };

    record_unknown_files(
        &dir_files,
        &fresh_files,
        &expected_failed_names,
        &mode,
        &mut result,
    );

    for fresh_name in &fresh_files {
        let stored_path = dir.join(fresh_name);
        if !stored_path.exists() {
            record_missing_file(fresh_name, &mode, &mut result);
            continue;
        }

        eprintln!("  checking {fresh_name}...");
        let fresh_path = tmp_dir.path().join(fresh_name);
        match run_catching_panic(|| check_single_fixture(&stored_path, &fresh_path, &session)) {
            Ok(()) => {
                eprintln!("  pass {fresh_name}");
                result.passed.push(fresh_name.clone());
            }
            Err(err) => {
                eprintln!("  FAIL {fresh_name}: {err}");
                result.failed.push(FailedFixture {
                    name: fresh_name.clone(),
                    error: err,
                });
            }
        }
    }

    Ok(result)
}

fn generate_expected_fixtures(
    tmp_dir: &Path,
    fixtures: &[Box<dyn Fixture>],
    session: &VortexSession,
) -> (Vec<FailedFixture>, BTreeSet<String>) {
    eprintln!("generating fresh fixtures for comparison...");
    let mut failed = Vec::new();
    let mut failed_names = BTreeSet::new();

    for fixture in fixtures {
        let fixture_name = fixture.name().to_string();
        let outcome = run_catching_panic(|| {
            let mut ctx = session.create_execution_ctx();
            fixture.write_expected(tmp_dir, &mut ctx)
        });
        if let Err(err) = outcome {
            let error = format!("failed to build expected fixture: {err}");
            eprintln!("  FAIL {fixture_name}: {error}");
            failed_names.insert(fixture_name.clone());
            failed.push(FailedFixture {
                name: fixture_name,
                error,
            });
        }
    }

    (failed, failed_names)
}

fn collect_vortex_files(dir: &Path) -> io::Result<Vec<String>> {
    let entries = fs::read_dir(dir)?;
    Ok(entries
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().to_string_lossy().to_string();
            name.ends_with(".vortex").then_some(name)
        })
        .collect())
}

fn record_unknown_files(
    dir_files: &[String],
    fresh_files: &[String],
    expected_failed_names: &BTreeSet<String>,
    mode: &Mode,
    result: &mut CheckResult,
) {
    for file_name in dir_files {
        if fresh_files.contains(file_name) || expected_failed_names.contains(file_name) {
            continue;
        }
        match mode {
            Mode::Exact | Mode::Superset => {
                result.failed.push(FailedFixture {
                    name: file_name.clone(),
                    error: "unknown fixture (not in current fixture set)".to_string(),
                });
            }
            Mode::Subset => {
                eprintln!("  skip {file_name} (unknown)");
                result.skipped.push(file_name.clone());
            }
        }
    }
}

fn record_missing_file(fresh_name: &str, mode: &Mode, result: &mut CheckResult) {
    match mode {
        Mode::Exact | Mode::Subset => {
            result.failed.push(FailedFixture {
                name: fresh_name.to_string(),
                error: "file missing from directory".to_string(),
            });
        }
        Mode::Superset => {
            eprintln!("  skip {fresh_name} (missing)");
            result.skipped.push(fresh_name.to_string());
        }
    }
}

fn check_single_fixture(
    stored_path: &Path,
    fresh_path: &Path,
    session: &VortexSession,
) -> Result<(), String> {
    let stored_bytes = ByteBuffer::from(
        fs::read(stored_path).map_err(|err| format!("failed to read stored file: {err}"))?,
    );
    let fresh_bytes = ByteBuffer::from(
        fs::read(fresh_path).map_err(|err| format!("failed to read fresh file: {err}"))?,
    );

    // Validate the full layout tree of the stored file (reads every segment
    // including zone maps, dictionaries, etc.).
    adapter::read_layout_tree(stored_bytes.clone())
        .map_err(|err| format!("stored file layout tree invalid: {err}"))?;

    // Scan data arrays from both files and compare.
    let stored_array = adapter::read_file(stored_bytes)
        .map_err(|err| format!("failed to decode stored vortex file: {err}"))?;
    let fresh_array = adapter::read_file(fresh_bytes)
        .map_err(|err| format!("failed to decode fresh vortex file: {err}"))?;

    let mut ctx = session.create_execution_ctx();
    assert_arrays_eq!(stored_array, fresh_array, &mut ctx);

    Ok(())
}

fn run_catching_panic<T, E: Display>(func: impl FnOnce() -> Result<T, E>) -> Result<T, String> {
    match catch_unwind(AssertUnwindSafe(func)) {
        Ok(Ok(val)) => Ok(val),
        Ok(Err(err)) => Err(err.to_string()),
        Err(payload) => Err(format!("panic: {}", panic_message(payload.as_ref()))),
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        (*msg).to_string()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use vortex_array::ArrayRef;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::FieldNames;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;

    use super::Mode;
    use super::check_fixtures;
    use crate::adapter;
    use crate::fixtures::Fixture;
    use crate::manifest::FixtureEntry;

    enum ExpectedBehavior {
        Array(ArrayRef),
        Error(&'static str),
        Panic(&'static str),
    }

    struct StubFixture {
        name: &'static str,
        behavior: ExpectedBehavior,
    }

    impl Fixture for StubFixture {
        fn name(&self) -> &str {
            self.name
        }

        fn description(&self) -> &str {
            "stub test fixture"
        }

        fn write(&self, dir: &Path, _ctx: &mut ExecutionCtx) -> VortexResult<Vec<FixtureEntry>> {
            let ExpectedBehavior::Array(array) = &self.behavior else {
                vortex_bail!("write() is only used with array stub fixtures");
            };
            adapter::write_file(&dir.join(self.name), array.clone())?;
            Ok(vec![FixtureEntry {
                name: self.name.to_string(),
                description: self.description().to_string(),
            }])
        }

        #[allow(clippy::panic)]
        fn write_expected(
            &self,
            dir: &Path,
            ctx: &mut ExecutionCtx,
        ) -> VortexResult<Vec<FixtureEntry>> {
            match &self.behavior {
                ExpectedBehavior::Array(_) => self.write(dir, ctx),
                ExpectedBehavior::Error(msg) => vortex_bail!("{msg}"),
                ExpectedBehavior::Panic(msg) => panic!("{msg}"),
            }
        }
    }

    fn make_two_col_struct(vals_a: [i32; 4], vals_b: [i64; 4]) -> VortexResult<ArrayRef> {
        let col_a = PrimitiveArray::from_iter(vals_a);
        let col_b = PrimitiveArray::from_iter(vals_b);
        Ok(StructArray::try_new(
            FieldNames::from(["a", "b"]),
            vec![col_a.into_array(), col_b.into_array()],
            4,
            Validity::NonNullable,
        )?
        .into_array())
    }

    #[test]
    fn collects_multiple_failures_without_stopping_early() -> VortexResult<()> {
        let stored_dir = tempfile::tempdir()?;
        let passing_arr = make_two_col_struct([1, 2, 3, 4], [10, 20, 30, 40])?;
        let mismatch_expected = make_two_col_struct([1, 2, 3, 4], [10, 20, 30, 40])?;
        let mismatch_stored = make_two_col_struct([1, 99, 3, 4], [10, 20, 30, 40])?;
        let corrupt_expected = make_two_col_struct([5, 6, 7, 8], [50, 60, 70, 80])?;

        adapter::write_file(
            &stored_dir.path().join("passing.vortex"),
            passing_arr.clone(),
        )?;
        adapter::write_file(&stored_dir.path().join("mismatch.vortex"), mismatch_stored)?;
        fs::write(
            stored_dir.path().join("corrupt.vortex"),
            b"not a vortex file",
        )?;
        adapter::write_file(
            &stored_dir.path().join("error_expected.vortex"),
            passing_arr.clone(),
        )?;
        adapter::write_file(
            &stored_dir.path().join("panic_expected.vortex"),
            passing_arr.clone(),
        )?;

        let fixtures: Vec<Box<dyn Fixture>> = vec![
            Box::new(StubFixture {
                name: "passing.vortex",
                behavior: ExpectedBehavior::Array(passing_arr),
            }),
            Box::new(StubFixture {
                name: "mismatch.vortex",
                behavior: ExpectedBehavior::Array(mismatch_expected),
            }),
            Box::new(StubFixture {
                name: "corrupt.vortex",
                behavior: ExpectedBehavior::Array(corrupt_expected),
            }),
            Box::new(StubFixture {
                name: "error_expected.vortex",
                behavior: ExpectedBehavior::Error("simulated builder error"),
            }),
            Box::new(StubFixture {
                name: "panic_expected.vortex",
                behavior: ExpectedBehavior::Panic("simulated builder panic"),
            }),
        ];

        let result = check_fixtures(stored_dir.path(), Mode::Exact, &fixtures)?;

        assert_eq!(result.passed, ["passing.vortex"]);
        assert!(result.skipped.is_empty());

        let failed_names: Vec<&str> = result
            .failed
            .iter()
            .map(|item| item.name.as_str())
            .collect();
        assert_eq!(
            failed_names,
            [
                "error_expected.vortex",
                "panic_expected.vortex",
                "corrupt.vortex",
                "mismatch.vortex",
            ]
        );
        assert!(result.failed[0].error.contains("simulated builder error"));
        assert!(result.failed[1].error.contains("simulated builder panic"));
        assert!(result.failed[2].error.contains("layout tree invalid"));
        assert!(result.failed[3].error.contains("arrays do not match"));

        Ok(())
    }

    #[test]
    fn handles_mode_mismatches_across_all_files() -> VortexResult<()> {
        let stored_dir = tempfile::tempdir()?;
        let arr = make_two_col_struct([1, 2, 3, 4], [10, 20, 30, 40])?;
        adapter::write_file(&stored_dir.path().join("extra_one.vortex"), arr.clone())?;
        adapter::write_file(&stored_dir.path().join("extra_two.vortex"), arr.clone())?;

        let fixtures: Vec<Box<dyn Fixture>> = vec![
            Box::new(StubFixture {
                name: "missing_one.vortex",
                behavior: ExpectedBehavior::Array(arr.clone()),
            }),
            Box::new(StubFixture {
                name: "missing_two.vortex",
                behavior: ExpectedBehavior::Array(arr),
            }),
        ];

        let exact = check_fixtures(stored_dir.path(), Mode::Exact, &fixtures)?;
        assert_eq!(exact.failed.len(), 4);

        let superset = check_fixtures(stored_dir.path(), Mode::Superset, &fixtures)?;
        assert_eq!(superset.failed.len(), 2);
        assert_eq!(
            superset.skipped,
            ["missing_one.vortex", "missing_two.vortex"]
        );

        let subset = check_fixtures(stored_dir.path(), Mode::Subset, &fixtures)?;
        assert_eq!(subset.failed.len(), 2);
        assert_eq!(subset.skipped, ["extra_one.vortex", "extra_two.vortex"]);

        Ok(())
    }
}
