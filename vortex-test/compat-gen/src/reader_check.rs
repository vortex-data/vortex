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
use arrow_array::make_array;
use arrow_schema::DataType;
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

use tokio::runtime::Runtime;
use vortex::VortexSessionDefault;
use vortex::file::OpenOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_array::stream::ArrayStreamExt;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

use crate::adapter;
use crate::queries;
use crate::queries::Query;
use crate::queries::QueryFile;

#[derive(Serialize)]
struct CheckResult {
    passed: Vec<String>,
    failed: Vec<FailedFixture>,
    queries_passed: Vec<String>,
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

/// Run one query with the current reader and export the result as a record batch.
pub fn query_batch(path: &Path, query: &Query) -> VortexResult<RecordBatch> {
    let bytes = std::fs::read(path).map_err(|e| vortex_err!("read {}: {e}", path.display()))?;
    let session = VortexSession::default().with_tokio();
    let file = session.open_options().open_buffer(ByteBuffer::from(bytes))?;
    let dtype = file.dtype().clone();
    let mut scan = file.scan()?;
    if let Some(filter) = &query.filter {
        scan = scan.with_filter(queries::filter_expression(filter)?.bind(&dtype)?);
    }
    if let Some(projection) = &query.projection {
        let names = FieldNames::from_iter(projection.iter().map(|n| FieldName::from(n.as_str())));
        scan = scan.with_projection(select(names, root()).bind(&dtype)?);
    }
    if let Some(indices) = &query.indices {
        scan = scan.with_row_indices(StrictSortedBuffer::try_new(Buffer::from(indices.clone()))?);
    }
    if let Some(limit) = query.limit {
        scan = scan.with_limit(limit);
    }
    let runtime = Runtime::new().map_err(|e| vortex_err!("failed to create tokio runtime: {e}"))?;
    let array = runtime.block_on(scan.into_array_stream()?.read_all())?;
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

/// Clear Arrow metadata that does not affect the values. The Map `keys_sorted` flag is dropped
/// by some IPC implementations, and it does not change the physical layout.
fn normalize_type(data_type: &DataType) -> DataType {
    match data_type {
        DataType::Map(field, _) => DataType::Map(normalize_field(field), false),
        DataType::Struct(fields) => DataType::Struct(fields.iter().map(normalize_field).collect()),
        DataType::List(field) => DataType::List(normalize_field(field)),
        DataType::LargeList(field) => DataType::LargeList(normalize_field(field)),
        DataType::ListView(field) => DataType::ListView(normalize_field(field)),
        DataType::LargeListView(field) => DataType::LargeListView(normalize_field(field)),
        DataType::FixedSizeList(field, n) => DataType::FixedSizeList(normalize_field(field), *n),
        other => other.clone(),
    }
}

fn normalize_field(field: &Arc<Field>) -> Arc<Field> {
    Arc::new(field.as_ref().clone().with_data_type(normalize_type(field.data_type())))
}

fn normalize(array: &ArrowArrayRef) -> VortexResult<ArrowArrayRef> {
    let normalized = normalize_type(array.data_type());
    if &normalized == array.data_type() {
        return Ok(Arc::clone(array));
    }
    let data = array
        .to_data()
        .into_builder()
        .data_type(normalized)
        .build()
        .map_err(|e| vortex_err!("{e}"))?;
    Ok(make_array(data))
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
        let cur = &normalize(cur)?;
        let old_col = &normalize(old_col)?;
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
        queries_passed: Vec::new(),
    };
    let queries_path = dir.join(queries::QUERIES_FILE);
    let queries: QueryFile = if queries_path.exists() {
        let text = std::fs::read_to_string(&queries_path).map_err(|e| vortex_err!("{e}"))?;
        serde_json::from_str(&text).map_err(|e| vortex_err!("bad {}: {e}", queries::QUERIES_FILE))?
    } else {
        QueryFile::new()
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
            // A panic in the current reader is a finding too, so record it instead of ending the run.
            let current = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                read_to_batch(&dir.join(&name))
            }))
            .unwrap_or_else(|panic| {
                let msg = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "unknown panic".to_string());
                Err(vortex_err!("current reader panicked: {msg}"))
            })?;
            let old = read_ipc(&arrow_path)?;
            compare(&current, &old)
        })();
        match outcome {
            Ok(()) => {
                eprintln!("  pass {name}");
                result.passed.push(name.clone());
            }
            Err(e) => {
                eprintln!("  FAIL {name}: {e}");
                result.failed.push(FailedFixture {
                    name: name.clone(),
                    error: e.to_string(),
                });
            }
        }

        for query in queries.get(&name).map(Vec::as_slice).unwrap_or_default() {
            let label = format!("{name}#q{}", query.id);
            let outcome = (|| -> VortexResult<()> {
                let error_path = arrow_dir.join(format!("{name}.q{}.error", query.id));
                if error_path.exists() {
                    let msg = std::fs::read_to_string(&error_path).map_err(|e| vortex_err!("{e}"))?;
                    vortex_bail!("old reader failed the query: {msg}");
                }
                let arrow_path = arrow_dir.join(format!("{name}.q{}.arrow", query.id));
                if !arrow_path.exists() {
                    vortex_bail!("old reader produced no output for this query");
                }
                let current = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    query_batch(&dir.join(&name), query)
                }))
                .unwrap_or_else(|panic| {
                    let msg = panic
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_else(|| "unknown panic".to_string());
                    Err(vortex_err!("current reader panicked: {msg}"))
                })?;
                let old = read_ipc(&arrow_path)?;
                compare(&current, &old)
            })();
            match outcome {
                Ok(()) => result.queries_passed.push(label),
                Err(e) => {
                    let detail = serde_json::to_string(query).unwrap_or_default();
                    eprintln!("  FAIL {label}: {e}  query={detail}");
                    result.failed.push(FailedFixture {
                        name: label,
                        error: format!("{e}  query={detail}"),
                    });
                }
            }
        }
    }

    let json = serde_json::to_string_pretty(&result)
        .map_err(|e| vortex_err!("failed to serialize result: {e}"))?;
    println!("{json}");
    eprintln!(
        "\nresult: {} passed, {} failed, {} queries passed",
        result.passed.len(),
        result.failed.len(),
        result.queries_passed.len()
    );
    if !result.failed.is_empty() {
        vortex_bail!("{} fixture(s) failed on the old reader", result.failed.len());
    }
    Ok(())
}
