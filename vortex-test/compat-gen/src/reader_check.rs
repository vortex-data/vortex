// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Forward-compatibility check: files written by this build must decode to the same values on
//! the oldest reader an edition promises to support.
//!
//! The old reader is a separately built binary pinned to a released `vortex` crate. It reads
//! every fixture in a directory and dumps each one as an Arrow IPC file (`<name>.arrow`), or
//! records the read error in `<name>.error`. This module decodes the same fixtures with the
//! current reader and compares the two in Arrow space, which is stable across Vortex versions.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::Array;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::RecordBatch;
use arrow_array::StructArray;
use arrow_ipc::reader::FileReader;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_select::concat::concat_batches;
use serde::Serialize;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_arrow::ArrowSessionExt;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::adapter;

#[derive(Serialize)]
struct CheckResult {
    passed: Vec<String>,
    failed: Vec<FailedFixture>,
}

#[derive(Serialize)]
struct FailedFixture {
    name: String,
    error: String,
}

/// Decode a fixture with the current reader and export it as a record batch, mirroring what the
/// old reader does on its side.
fn read_to_batch(path: &Path) -> VortexResult<RecordBatch> {
    let bytes = std::fs::read(path).map_err(|e| vortex_err!("read {}: {e}", path.display()))?;
    let array = adapter::read_file(ByteBuffer::from(bytes))?;
    let session = array_session();
    let mut ctx = session.create_execution_ctx();
    let arrow = session.arrow().execute_arrow(array, None, &mut ctx)?;
    to_batch(arrow)
}

fn to_batch(arrow: ArrowArrayRef) -> VortexResult<RecordBatch> {
    match arrow.as_any().downcast_ref::<StructArray>() {
        Some(s) => Ok(RecordBatch::from(s)),
        None => {
            let field = Field::new("value", arrow.data_type().clone(), arrow.is_nullable());
            RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![arrow])
                .map_err(|e| vortex_err!("{e}"))
        }
    }
}

fn read_ipc(path: &Path) -> VortexResult<RecordBatch> {
    let file = File::open(path).map_err(|e| vortex_err!("open {}: {e}", path.display()))?;
    let reader = FileReader::try_new(file, None).map_err(|e| vortex_err!("{e}"))?;
    let schema = reader.schema();
    let batches = reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| vortex_err!("{e}"))?;
    concat_batches(&schema, &batches).map_err(|e| vortex_err!("{e}"))
}

/// Compare the two decodings column by column, naming the first difference.
fn compare(current: &RecordBatch, old: &RecordBatch) -> VortexResult<()> {
    if current.num_rows() != old.num_rows() {
        vortex_bail!(
            "row count differs: current reader {} vs old reader {}",
            current.num_rows(),
            old.num_rows()
        );
    }
    if current.num_columns() != old.num_columns() {
        vortex_bail!(
            "column count differs: current reader {} vs old reader {}",
            current.num_columns(),
            old.num_columns()
        );
    }
    for (i, (cur, old_col)) in current.columns().iter().zip(old.columns()).enumerate() {
        let name = current.schema().field(i).name().clone();
        // A newer release may prefer a different Arrow representation for the same values, so
        // cast to the old reader's type before comparing.
        let cur = if cur.data_type() == old_col.data_type() {
            Arc::clone(cur)
        } else {
            arrow_cast::cast(cur.as_ref(), old_col.data_type()).map_err(|e| {
                vortex_err!(
                    "column {name}: type differs ({} vs {}) and cannot be cast: {e}",
                    cur.data_type(),
                    old_col.data_type()
                )
            })?
        };
        if cur.as_ref() != old_col.as_ref() {
            let row = (0..cur.len())
                .find(|&r| cur.slice(r, 1).as_ref() != old_col.slice(r, 1).as_ref())
                .unwrap_or_default();
            vortex_bail!(
                "column {name}: values differ, first at row {row}: current reader {:?} vs old \
                 reader {:?}",
                cur.slice(row, 1),
                old_col.slice(row, 1)
            );
        }
    }
    Ok(())
}

/// Check that the old reader's dumps in `arrow_dir` match the current reader's decoding of the
/// fixtures in `dir`.
pub fn check_reader(dir: &Path, arrow_dir: &Path) -> VortexResult<()> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .map_err(|e| vortex_err!("failed to read dir {}: {e}", dir.display()))?
        .filter_map(|entry| {
            let name = entry.ok()?.file_name().to_string_lossy().to_string();
            name.ends_with(".vortex").then_some(name)
        })
        .collect();
    names.sort();

    let mut result = CheckResult {
        passed: Vec::new(),
        failed: Vec::new(),
    };

    for name in names {
        eprintln!("  checking {name}...");
        let outcome = (|| -> VortexResult<()> {
            let error_path = arrow_dir.join(format!("{name}.error"));
            if error_path.exists() {
                let msg = std::fs::read_to_string(&error_path).map_err(|e| vortex_err!("{e}"))?;
                vortex_bail!("old reader failed to read the file: {msg}");
            }
            let arrow_path = arrow_dir.join(format!("{name}.arrow"));
            if !arrow_path.exists() {
                vortex_bail!("old reader produced no output for this fixture");
            }
            let current = read_to_batch(&dir.join(&name))?;
            let old = read_ipc(&arrow_path)?;
            compare(&current, &old)
        })();
        match outcome {
            Ok(()) => {
                eprintln!("  pass {name}");
                result.passed.push(name);
            }
            Err(e) => {
                eprintln!("  FAIL {name}: {e}");
                result.failed.push(FailedFixture {
                    name,
                    error: e.to_string(),
                });
            }
        }
    }

    let json = serde_json::to_string_pretty(&result)
        .map_err(|e| vortex_err!("failed to serialize result: {e}"))?;
    println!("{json}");
    eprintln!(
        "\nresult: {} passed, {} failed",
        result.passed.len(),
        result.failed.len()
    );
    if !result.failed.is_empty() {
        vortex_bail!("{} fixture(s) failed on the old reader", result.failed.len());
    }
    Ok(())
}
