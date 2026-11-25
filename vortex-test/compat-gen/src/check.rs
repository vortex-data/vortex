// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::collections::BTreeSet;
use std::fmt::Display;
use std::fs;
use std::io;
use std::ops::Range;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;
use std::path::Path;

use clap::ValueEnum;
use serde::Serialize;
use vortex::VortexSessionDefault;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::all_non_distinct::all_non_distinct;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::adapter;
use crate::fixtures::Fixture;
use crate::fixtures::all_fixtures;

const MAX_REPORTED_MISMATCHES: usize = 5;

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
/// validates layout trees, full scans, projected scans, and row-range scans, and compares
/// arrays without stopping at the first failure.
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

    let session = VortexSession::default();
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
    let stored_array = adapter::read_file(stored_bytes.clone())
        .map_err(|err| format!("failed to decode stored vortex file: {err}"))?;
    let fresh_array = adapter::read_file(fresh_bytes)
        .map_err(|err| format!("failed to decode fresh vortex file: {err}"))?;

    let mut ctx = session.create_execution_ctx();
    verify_arrays_eq(&stored_array, &fresh_array, &mut ctx, "full scan")?;

    let mut subcheck_errors = Vec::new();
    if let Err(err) = verify_projected_scan(&stored_bytes, &fresh_array, &mut ctx) {
        subcheck_errors.push(err);
    }
    if let Err(err) = verify_row_range_scan(stored_bytes, &fresh_array, &mut ctx) {
        subcheck_errors.push(err);
    }

    if !subcheck_errors.is_empty() {
        return Err(subcheck_errors.join("; "));
    }

    Ok(())
}

fn verify_projected_scan(
    stored_bytes: &ByteBuffer,
    fresh_array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> Result<(), String> {
    let Some(fields) = select_projection_fields(fresh_array.dtype()) else {
        return Ok(());
    };

    let stored_projected = adapter::read_projected(stored_bytes.clone(), &fields)
        .map_err(|err| format!("projected scan failed: {err}"))?;
    let expected_projected = fresh_array
        .clone()
        .execute::<StructArray>(ctx)
        .and_then(|struct_array| struct_array.project(&fields))
        .map(IntoArray::into_array)
        .map_err(|err| format!("projected scan expected build failed: {err}"))?;

    verify_arrays_eq(
        &stored_projected,
        &expected_projected,
        ctx,
        "projected scan",
    )
}

fn verify_row_range_scan(
    stored_bytes: ByteBuffer,
    fresh_array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> Result<(), String> {
    let Some(row_range) = select_row_range(fresh_array.len())? else {
        return Ok(());
    };

    let start = usize::try_from(row_range.start)
        .map_err(|err| format!("row range start overflow: {err}"))?;
    let end =
        usize::try_from(row_range.end).map_err(|err| format!("row range end overflow: {err}"))?;

    let stored_slice = adapter::read_row_range(stored_bytes, row_range)
        .map_err(|err| format!("row-range scan ({start}..{end}) failed: {err}"))?;
    let expected_slice = fresh_array
        .slice(start..end)
        .map_err(|err| format!("row-range scan ({start}..{end}) expected slice failed: {err}"))?;

    let label = format!("row-range scan ({start}..{end})");
    verify_arrays_eq(&stored_slice, &expected_slice, ctx, &label)
}

fn select_projection_fields(dtype: &DType) -> Option<Vec<FieldName>> {
    let DType::Struct(fields, _) = dtype else {
        return None;
    };
    let names = fields.names();
    match names.len() {
        0 => None,
        1 => Some(vec![names[0].clone()]),
        2 => Some(vec![names[1].clone()]),
        len => Some(vec![names[0].clone(), names[len - 1].clone()]),
    }
}

fn select_row_range(len: usize) -> Result<Option<Range<u64>>, String> {
    if len < 2 {
        return Ok(None);
    }
    let margin = (len / 4).max(1);
    let start = margin;
    let end = (len - margin).max(start + 1).min(len);
    let start_u64 =
        u64::try_from(start).map_err(|err| format!("row range start overflow: {err}"))?;
    let end_u64 = u64::try_from(end).map_err(|err| format!("row range end overflow: {err}"))?;
    Ok(Some(start_u64..end_u64))
}

fn verify_arrays_eq(
    left: &ArrayRef,
    right: &ArrayRef,
    ctx: &mut ExecutionCtx,
    label: &str,
) -> Result<(), String> {
    if left.dtype() != right.dtype() {
        return Err(format!(
            "{label}: arrays differ in type: {} != {}",
            left.dtype(),
            right.dtype(),
        ));
    }

    if left.len() != right.len() {
        return Err(format!(
            "{label}: arrays differ in length: {} != {}",
            left.len(),
            right.len(),
        ));
    }

    let executed = left
        .clone()
        .execute::<RecursiveCanonical>(ctx)
        .map_err(|err| format!("{label}: canonical execution failed: {err}"))?
        .0
        .into_array();

    let left_right_same = all_non_distinct(left, right, ctx)
        .map_err(|err| format!("{label}: failed to compare arrays: {err}"))?;
    let executed_right_same = all_non_distinct(&executed, right, ctx)
        .map_err(|err| format!("{label}: failed to compare executed arrays: {err}"))?;

    if !left_right_same || !executed_right_same {
        let mut details = Vec::new();
        if !left_right_same {
            let mismatches = sample_mismatches(left, right, ctx);
            details.push(format!("left != right ({mismatches})"));
        }
        if !executed_right_same {
            let mismatches = sample_mismatches(&executed, right, ctx);
            details.push(format!("executed != right ({mismatches})"));
        }
        return Err(format!(
            "{label}: arrays do not match: {}",
            details.join("; ")
        ));
    }

    Ok(())
}

fn sample_mismatches(left: &ArrayRef, right: &ArrayRef, ctx: &mut ExecutionCtx) -> String {
    let mut diffs = Vec::new();
    for idx in 0..left.len() {
        match (
            left.execute_scalar(idx, ctx),
            right.execute_scalar(idx, ctx),
        ) {
            (Ok(lhs_val), Ok(rhs_val)) if lhs_val != rhs_val => {
                diffs.push(format!("row {idx}: got {lhs_val}, expected {rhs_val}"));
            }
            (Err(err), _) => {
                diffs.push(format!("row {idx}: left scalar error: {err}"));
            }
            (_, Err(err)) => {
                diffs.push(format!("row {idx}: right scalar error: {err}"));
            }
            _ => {}
        }
        if diffs.len() >= MAX_REPORTED_MISMATCHES {
            break;
        }
    }

    if diffs.is_empty() {
        "values differ".to_string()
    } else {
        diffs.join(", ")
    }
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

    use vortex::VortexSessionDefault;
    use vortex_array::ArrayRef;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldName;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_session::VortexSession;

    use super::Mode;
    use super::check_fixtures;
    use super::select_projection_fields;
    use super::select_row_range;
    use super::verify_arrays_eq;
    use super::verify_projected_scan;
    use super::verify_row_range_scan;
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

    #[test]
    fn verify_arrays_eq_reports_mismatches_without_panicking() -> VortexResult<()> {
        let session = VortexSession::default();
        let mut ctx = session.create_execution_ctx();
        let left = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
        let right_diff_val = PrimitiveArray::from_iter([1i32, 99, 3]).into_array();
        let right_diff_len = PrimitiveArray::from_iter([1i32, 2]).into_array();
        let right_diff_type = PrimitiveArray::from_iter([1i64, 2, 3]).into_array();

        assert!(verify_arrays_eq(&left, &left, &mut ctx, "match").is_ok());

        let Err(val_err) = verify_arrays_eq(&left, &right_diff_val, &mut ctx, "val") else {
            vortex_bail!("expected value mismatch error");
        };
        assert!(val_err.contains("row 1: got 2i32, expected 99i32"));

        let Err(len_err) = verify_arrays_eq(&left, &right_diff_len, &mut ctx, "len") else {
            vortex_bail!("expected length mismatch error");
        };
        assert!(len_err.contains("differ in length: 3 != 2"));

        let Err(type_err) = verify_arrays_eq(&left, &right_diff_type, &mut ctx, "dtype") else {
            vortex_bail!("expected dtype mismatch error");
        };
        assert!(type_err.contains("differ in type"));

        Ok(())
    }

    #[test]
    fn select_projection_and_row_range_helpers() {
        let i32_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let i64_dtype = DType::Primitive(PType::I64, Nullability::NonNullable);

        assert_eq!(select_projection_fields(&i32_dtype), None);

        let two_fields = DType::Struct(
            StructFields::from_iter([("first", i32_dtype.clone()), ("second", i64_dtype.clone())]),
            Nullability::NonNullable,
        );
        assert_eq!(
            select_projection_fields(&two_fields),
            Some(vec![FieldName::from("second")])
        );

        let three_fields = DType::Struct(
            StructFields::from_iter([
                ("first", i32_dtype.clone()),
                ("middle", i32_dtype),
                ("last", i64_dtype),
            ]),
            Nullability::NonNullable,
        );
        assert_eq!(
            select_projection_fields(&three_fields),
            Some(vec![FieldName::from("first"), FieldName::from("last")])
        );

        assert_eq!(select_row_range(0), Ok(None));
        assert_eq!(select_row_range(1), Ok(None));
        assert_eq!(select_row_range(2), Ok(Some(1..2)));
        assert_eq!(select_row_range(3), Ok(Some(1..2)));
        assert_eq!(select_row_range(4), Ok(Some(1..3)));
        assert_eq!(select_row_range(1024), Ok(Some(256..768)));
    }

    #[test]
    fn projected_and_row_range_scans() -> VortexResult<()> {
        let session = VortexSession::default();
        let mut ctx = session.create_execution_ctx();

        let arr = make_two_col_struct([1, 2, 3, 4], [10, 20, 30, 40])?;
        let bytes = adapter::write_file_to_bytes(arr.clone())?;
        let projected = adapter::read_projected(bytes.clone(), &[FieldName::from("b")])?;
        let expected_projected = arr
            .clone()
            .execute::<StructArray>(&mut ctx)?
            .project(&[FieldName::from("b")])?
            .into_array();
        assert_arrays_eq!(projected, expected_projected, &mut ctx);

        let sliced = adapter::read_row_range(bytes.clone(), 1..3)?;
        let expected_sliced = arr.slice(1..3)?;
        assert_arrays_eq!(sliced, expected_sliced, &mut ctx);

        assert!(verify_projected_scan(&bytes, &arr, &mut ctx).is_ok());
        assert!(verify_row_range_scan(bytes.clone(), &arr, &mut ctx).is_ok());

        let diff_col_b = make_two_col_struct([1, 2, 3, 4], [10, 99, 30, 40])?;
        let Err(proj_err) = verify_projected_scan(&bytes, &diff_col_b, &mut ctx) else {
            vortex_bail!("expected projected scan mismatch error");
        };
        assert!(proj_err.contains("projected scan: arrays do not match"));

        let Err(slice_err) = verify_row_range_scan(bytes, &diff_col_b, &mut ctx) else {
            vortex_bail!("expected row-range scan mismatch error");
        };
        assert!(slice_err.contains("row-range scan (1..3): arrays do not match"));

        Ok(())
    }
}
