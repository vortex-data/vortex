// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared DataFusion query execution utilities for both CLI and TUI.

use std::sync::Arc;

use arrow_array::Array as ArrowArray;
use arrow_array::RecordBatch;
use datafusion::datasource::listing::ListingOptions;
use datafusion::datasource::listing::ListingTable;
use datafusion::datasource::listing::ListingTableConfig;
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::prelude::SessionContext;
use vortex::session::VortexSession;
use vortex_datafusion::VortexFormat;

/// Execute a SQL query against a Vortex file.
///
/// The file is registered as a table named "data".
/// Returns the result as a vector of RecordBatches.
///
/// # Errors
///
/// Returns an error if the query fails to parse or execute.
pub async fn execute_vortex_query(
    session: &VortexSession,
    file_path: &str,
    sql: &str,
) -> Result<Vec<RecordBatch>, String> {
    let ctx = create_vortex_context(session, file_path).await?;

    let df = ctx.sql(sql).await.map_err(|e| format!("SQL error: {e}"))?;

    df.collect()
        .await
        .map_err(|e| format!("Query execution error: {e}"))
}

/// Create a DataFusion SessionContext with a Vortex file registered as "data".
///
/// # Errors
///
/// Returns an error if the context cannot be created.
pub async fn create_vortex_context(
    session: &VortexSession,
    file_path: &str,
) -> Result<SessionContext, String> {
    let ctx = SessionContext::new();
    let format = Arc::new(VortexFormat::new(session.clone()));

    let table_url =
        ListingTableUrl::parse(file_path).map_err(|e| format!("Failed to parse file path: {e}"))?;

    let config = ListingTableConfig::new(table_url)
        .with_listing_options(
            ListingOptions::new(format).with_session_config_options(ctx.state().config()),
        )
        .infer_schema(&ctx.state())
        .await
        .map_err(|e| format!("Failed to infer schema: {e}"))?;

    let listing_table = Arc::new(
        ListingTable::try_new(config).map_err(|e| format!("Failed to create table: {e}"))?,
    );

    ctx.register_table("data", listing_table)
        .map_err(|e| format!("Failed to register table: {e}"))?;

    Ok(ctx)
}

/// Convert an Arrow array value at a given index to a JSON value.
///
/// # Panics
///
/// Panics if the array type doesn't match the expected Arrow array type during downcast.
/// This should not happen for well-formed Arrow arrays.
#[expect(clippy::unwrap_used)]
pub fn arrow_value_to_json(array: &dyn ArrowArray, idx: usize) -> serde_json::Value {
    use arrow_array::*;
    use arrow_schema::DataType;

    if array.is_null(idx) {
        return serde_json::Value::Null;
    }

    match array.data_type() {
        DataType::Null => serde_json::Value::Null,
        DataType::Boolean => {
            let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            serde_json::Value::Bool(arr.value(idx))
        }
        DataType::Int8 => {
            let arr = array.as_any().downcast_ref::<Int8Array>().unwrap();
            serde_json::json!(arr.value(idx))
        }
        DataType::Int16 => {
            let arr = array.as_any().downcast_ref::<Int16Array>().unwrap();
            serde_json::json!(arr.value(idx))
        }
        DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
            serde_json::json!(arr.value(idx))
        }
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            serde_json::json!(arr.value(idx))
        }
        DataType::UInt8 => {
            let arr = array.as_any().downcast_ref::<UInt8Array>().unwrap();
            serde_json::json!(arr.value(idx))
        }
        DataType::UInt16 => {
            let arr = array.as_any().downcast_ref::<UInt16Array>().unwrap();
            serde_json::json!(arr.value(idx))
        }
        DataType::UInt32 => {
            let arr = array.as_any().downcast_ref::<UInt32Array>().unwrap();
            serde_json::json!(arr.value(idx))
        }
        DataType::UInt64 => {
            let arr = array.as_any().downcast_ref::<UInt64Array>().unwrap();
            serde_json::json!(arr.value(idx))
        }
        DataType::Float16 => {
            let arr = array.as_any().downcast_ref::<Float16Array>().unwrap();
            serde_json::json!(arr.value(idx).to_f32())
        }
        DataType::Float32 => {
            let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
            serde_json::json!(arr.value(idx))
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            serde_json::json!(arr.value(idx))
        }
        DataType::Utf8 => {
            let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
            serde_json::Value::String(arr.value(idx).to_string())
        }
        DataType::LargeUtf8 => {
            let arr = array.as_any().downcast_ref::<LargeStringArray>().unwrap();
            serde_json::Value::String(arr.value(idx).to_string())
        }
        DataType::Utf8View => {
            let arr = array.as_any().downcast_ref::<StringViewArray>().unwrap();
            serde_json::Value::String(arr.value(idx).to_string())
        }
        DataType::Binary => {
            let arr = array.as_any().downcast_ref::<BinaryArray>().unwrap();
            let hex: String = arr.value(idx).iter().map(|b| format!("{b:02x}")).collect();
            serde_json::Value::String(hex)
        }
        DataType::LargeBinary => {
            let arr = array.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
            let hex: String = arr.value(idx).iter().map(|b| format!("{b:02x}")).collect();
            serde_json::Value::String(hex)
        }
        DataType::BinaryView => {
            let arr = array.as_any().downcast_ref::<BinaryViewArray>().unwrap();
            let hex: String = arr.value(idx).iter().map(|b| format!("{b:02x}")).collect();
            serde_json::Value::String(hex)
        }
        DataType::Date32 => {
            let arr = array.as_any().downcast_ref::<Date32Array>().unwrap();
            serde_json::json!(arr.value(idx))
        }
        DataType::Date64 => {
            let arr = array.as_any().downcast_ref::<Date64Array>().unwrap();
            serde_json::json!(arr.value(idx))
        }
        DataType::Timestamp(..) => {
            if let Some(arr) = array.as_any().downcast_ref::<TimestampMicrosecondArray>() {
                serde_json::json!(arr.value(idx))
            } else if let Some(arr) = array.as_any().downcast_ref::<TimestampMillisecondArray>() {
                serde_json::json!(arr.value(idx))
            } else if let Some(arr) = array.as_any().downcast_ref::<TimestampSecondArray>() {
                serde_json::json!(arr.value(idx))
            } else if let Some(arr) = array.as_any().downcast_ref::<TimestampNanosecondArray>() {
                serde_json::json!(arr.value(idx))
            } else {
                serde_json::Value::String("<timestamp>".to_string())
            }
        }
        DataType::Decimal128(..) => {
            let arr = array.as_any().downcast_ref::<Decimal128Array>().unwrap();
            serde_json::Value::String(arr.value_as_string(idx))
        }
        DataType::Decimal256(..) => {
            let arr = array.as_any().downcast_ref::<Decimal256Array>().unwrap();
            serde_json::Value::String(arr.value_as_string(idx))
        }
        DataType::List(_) => {
            let arr = array.as_any().downcast_ref::<ListArray>().unwrap();
            let value_arr = arr.value(idx);
            let elements: Vec<serde_json::Value> = (0..value_arr.len())
                .map(|i| arrow_value_to_json(value_arr.as_ref(), i))
                .collect();
            serde_json::Value::Array(elements)
        }
        DataType::LargeList(_) => {
            let arr = array.as_any().downcast_ref::<LargeListArray>().unwrap();
            let value_arr = arr.value(idx);
            let elements: Vec<serde_json::Value> = (0..value_arr.len())
                .map(|i| arrow_value_to_json(value_arr.as_ref(), i))
                .collect();
            serde_json::Value::Array(elements)
        }
        DataType::Struct(_) => {
            let arr = array.as_any().downcast_ref::<StructArray>().unwrap();
            let mut obj = serde_json::Map::new();
            for (i, field) in arr.fields().iter().enumerate() {
                let col = arr.column(i);
                obj.insert(field.name().clone(), arrow_value_to_json(col.as_ref(), idx));
            }
            serde_json::Value::Object(obj)
        }
        _ => {
            // Fallback for unsupported types
            serde_json::Value::String(format!("<{}>", array.data_type()))
        }
    }
}

/// Format a JSON value for display in the TUI.
///
/// - Null becomes "NULL"
/// - Strings are displayed without quotes
/// - Other values use their JSON string representation
pub fn json_value_to_display(value: serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "NULL".to_string(),
        serde_json::Value::String(s) => s,
        other => other.to_string(),
    }
}
