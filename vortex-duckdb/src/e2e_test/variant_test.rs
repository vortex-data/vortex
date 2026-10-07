// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! DuckDB reads fields of shredded Variant columns by extracting them in the Vortex scan.

use std::slice;
use std::sync::Arc;

use anyhow::Result;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::StringArray;
use arrow_schema::DataType;
use num_traits::AsPrimitive;
use parquet_variant_compute::ShreddedSchemaBuilder;
use parquet_variant_compute::json_to_variant;
use parquet_variant_compute::shred_variant;
use tempfile::NamedTempFile;
use vortex::array::IntoArray;
use vortex::array::arrays::StructArray;
use vortex::arrow::ArrowSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::runtime::BlockingRuntime;

use crate::RUNTIME;
use crate::SESSION;
use crate::cpp;
use crate::cpp::duckdb_string_t;
use crate::duckdb::Connection;
use crate::duckdb::Database;

fn database_connection() -> Connection {
    let db = Database::open_in_memory().unwrap();
    crate::initialize(&db).unwrap();
    db.connect().unwrap()
}

/// A Vortex file with one shredded Variant column `data`.
fn write_variant_file() -> Result<NamedTempFile> {
    let json: ArrowArrayRef = Arc::new(StringArray::from(vec![
        r#"{"kind": "commit", "commit": {"collection": "post", "rev": 1}}"#,
        r#"{"kind": "identity"}"#,
        r#"{"kind": "commit", "commit": {"collection": 7}}"#,
        r#"{"kind": "commit", "commit": {"collection": "like"}}"#,
        r#"[1, 2]"#,
    ]));
    let schema = ShreddedSchemaBuilder::new()
        .with_path("kind", &DataType::Utf8)?
        .with_path("commit.collection", &DataType::Utf8)?
        .build();
    let variant = shred_variant(&json_to_variant(&json)?, &schema)?;
    let field = variant.field("data");
    let data = SESSION
        .arrow()
        .from_arrow_array(ArrowArrayRef::from(variant), &field)?;
    let table = StructArray::try_from_iter([("data", data)])?.into_array();

    let file = NamedTempFile::with_suffix(".vortex")?;
    RUNTIME.block_on(async {
        let mut output = async_fs::File::create(file.path()).await?;
        SESSION
            .write_options()
            .write(&mut output, table.to_array_stream())
            .await?;
        anyhow::Ok(())
    })?;
    Ok(file)
}

/// The values of column `column` of a query returning only non-null strings.
fn query_strings(conn: &Connection, query: &str, column: usize) -> Result<Vec<String>> {
    let mut values = Vec::new();
    for mut chunk in conn.query(query)? {
        let len = chunk.len().as_();
        for value in unsafe {
            chunk
                .get_vector_mut(column)
                .as_slice_mut::<duckdb_string_t>(len)
        } {
            let bytes = unsafe {
                slice::from_raw_parts(
                    cpp::duckdb_string_t_data(&raw mut *value) as *const u8,
                    cpp::duckdb_string_t_length(*value) as usize,
                )
            };
            values.push(String::from_utf8_lossy(bytes).into_owned());
        }
    }
    Ok(values)
}

#[test]
fn variant_fields_are_extracted_by_the_scan() -> Result<()> {
    let file = write_variant_file()?;
    let conn = database_connection();
    // Through a view, as tables are commonly registered.
    conn.query(&format!(
        "CREATE VIEW t AS SELECT * FROM read_vortex('{}')",
        file.path().display()
    ))?;

    let query = "SELECT CAST(data.\"commit\".\"collection\" AS VARCHAR) AS c FROM t \
                 WHERE CAST(data.kind AS VARCHAR) = 'commit' AND c IS NOT NULL ORDER BY c";
    assert_eq!(query_strings(&conn, query, 0)?, ["like", "post"]);

    let plan = query_strings(&conn, &format!("EXPLAIN {query}"), 1)?.join("\n");
    assert!(
        plan.contains("data.commit.collection") && plan.contains("data.kind"),
        "the scan must extract the queried fields:\n{plan}"
    );
    Ok(())
}
