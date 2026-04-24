// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execute SQL queries against Vortex files using DataFusion.

use std::path::PathBuf;

use arrow_array::RecordBatch;
use serde::Serialize;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::session::VortexSession;

use crate::datafusion_helper::arrow_value_to_json;
use crate::datafusion_helper::execute_vortex_query;

/// Command-line arguments for the query command.
#[derive(Debug, clap::Parser)]
pub struct QueryArgs {
    /// Path to the Vortex file
    pub file: PathBuf,

    /// SQL query to execute. The table is available as 'data'.
    /// Example: "SELECT * FROM data WHERE col > 10 LIMIT 100"
    #[arg(long, short)]
    pub sql: String,
}

#[derive(Serialize)]
struct QueryOutput {
    schema: SchemaInfo,
    total_rows: u64,
    rows: Vec<serde_json::Value>,
}

#[derive(Serialize)]
struct SchemaInfo {
    fields: Vec<FieldInfo>,
}

#[derive(Serialize)]
struct FieldInfo {
    name: String,
    dtype: String,
    nullable: bool,
}

/// Execute a SQL query against a Vortex file.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or the query fails.
pub async fn exec_query(session: &VortexSession, args: QueryArgs) -> VortexResult<()> {
    let file_path = args
        .file
        .to_str()
        .ok_or_else(|| vortex_err!("Path is not valid UTF-8"))?;

    let batches: Vec<RecordBatch> = execute_vortex_query(session, file_path, &args.sql)
        .await
        .map_err(|e| vortex_err!("{e}"))?;

    // Build schema info from the result
    let schema = if let Some(batch) = batches.first() {
        build_schema_from_arrow(batch.schema().as_ref())
    } else {
        SchemaInfo { fields: vec![] }
    };

    // Convert batches to JSON rows
    let mut rows = Vec::new();
    for batch in &batches {
        batch_to_json_rows(batch, &mut rows)?;
    }

    let total_rows = rows.len() as u64;

    let output = QueryOutput {
        schema,
        total_rows,
        rows,
    };

    let json_output = serde_json::to_string_pretty(&output)
        .map_err(|e| vortex_err!("Failed to serialize JSON: {e}"))?;
    println!("{json_output}");

    Ok(())
}

fn build_schema_from_arrow(schema: &arrow_schema::Schema) -> SchemaInfo {
    let fields = schema
        .fields()
        .iter()
        .map(|f| FieldInfo {
            name: f.name().clone(),
            dtype: f.data_type().to_string(),
            nullable: f.is_nullable(),
        })
        .collect();

    SchemaInfo { fields }
}

fn batch_to_json_rows(batch: &RecordBatch, rows: &mut Vec<serde_json::Value>) -> VortexResult<()> {
    let schema = batch.schema();

    for row_idx in 0..batch.num_rows() {
        let mut obj = serde_json::Map::new();

        for (col_idx, field) in schema.fields().iter().enumerate() {
            let column = batch.column(col_idx);
            let value = arrow_value_to_json(column.as_ref(), row_idx);
            obj.insert(field.name().clone(), value);
        }

        rows.push(serde_json::Value::Object(obj));
    }

    Ok(())
}
