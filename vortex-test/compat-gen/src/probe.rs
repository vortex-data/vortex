// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Debug helper: evaluate one comparison against an f64 column three ways and print the counts.

use std::path::Path;

use arrow_array::Array;
use arrow_array::Float64Array;
use arrow_array::cast::AsArray;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::adapter;
use crate::queries::Filter;
use crate::queries::LiteralDType;
use crate::queries::Query;
use crate::reader_check::query_batch;

pub fn probe(path: &Path, column: &str, op: &str, value: f64) -> VortexResult<()> {
    let filter = Filter {
        column: column.to_string(),
        op: op.to_string(),
        value: serde_json::Number::from_f64(value)
            .map(serde_json::Value::Number)
            .ok_or_else(|| vortex_err!("bad literal"))?,
        dtype: LiteralDType {
            kind: "float".into(),
            width: Some(64),
            nullable: false,
        },
    };
    // (a) scan with the filter pushed down, before and after a full read of the same file in
    // this process, to expose state carried between scans.
    let query = Query {
        id: 0,
        projection: None,
        filter: Some(filter.clone()),
        indices: None,
        limit: None,
    };
    let first = query_batch(path, &query)?.num_rows();
    {
        let bytes = std::fs::read(path).map_err(|e| vortex_err!("{e}"))?;
        let full = adapter::read_file(ByteBuffer::from(bytes))?;
        println!("full read: {} rows", full.len());
    }
    let second = query_batch(path, &query)?.num_rows();
    println!("filtered scan before full read: {first}, after: {second}");
    let scanned = query_batch(
        path,
        &Query {
            id: 0,
            projection: None,
            filter: Some(filter.clone()),
            indices: None,
            limit: None,
        },
    )?;
    // (b) full scan, compare in plain Rust on the Arrow export.
    let bytes = std::fs::read(path).map_err(|e| vortex_err!("{e}"))?;
    let array = adapter::read_file(ByteBuffer::from(bytes))?;
    let session = vortex_array::array_session();
    let mut ctx = {
        use vortex_array::VortexSessionExecute;
        session.create_execution_ctx()
    };
    let arrow = {
        use vortex_arrow::ArrowSessionExt;
        session.arrow().execute_arrow(array, None, &mut ctx)?
    };
    let structs = arrow.as_struct();
    let col = structs
        .column_by_name(column)
        .ok_or_else(|| vortex_err!("no column {column}"))?;
    let col: &Float64Array = col.as_primitive();
    let plain = (0..col.len())
        .filter(|&i| col.is_valid(i))
        .filter(|&i| {
            let x = col.value(i);
            match op {
                "eq" => x == value,
                "not_eq" => x != value,
                "gt" => x > value,
                "gt_eq" => x >= value,
                "lt" => x < value,
                "lt_eq" => x <= value,
                _ => false,
            }
        })
        .count();
    println!(
        "{}: {column} {op} {value:e}: scan={} plain={} rows={}",
        path.display(),
        scanned.num_rows(),
        plain,
        col.len()
    );
    Ok(())
}
