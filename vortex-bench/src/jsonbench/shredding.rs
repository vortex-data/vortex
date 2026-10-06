// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Infers a Variant shredding schema from a sample of JSON documents.
//!
//! The inference is generic and query-agnostic: it shreds every object path that is present in a
//! large enough fraction of the sampled documents and almost always holds the same scalar type.
//! Paths that hold mixed types (e.g. a string in some documents and an object in others) and
//! arrays stay in the residual Variant `value`.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use serde_json::Value;

/// Paths present in fewer than this fraction of the sampled documents are not shredded.
const MIN_PRESENCE: f64 = 0.1;
/// A path is shredded as a type only if at least this fraction of its values have that type.
const MIN_TYPE_CONSISTENCY: f64 = 0.99;
/// Object nesting depth beyond which paths are not shredded.
const MAX_DEPTH: usize = 4;

#[derive(Default, Debug)]
struct PathStats {
    count: u64,
    string: u64,
    int: u64,
    float: u64,
    boolean: u64,
    object: u64,
    children: BTreeMap<String, PathStats>,
}

impl PathStats {
    fn observe(&mut self, value: &Value, depth: usize) {
        self.count += 1;
        match value {
            Value::String(_) => self.string += 1,
            Value::Number(n) if n.is_i64() => self.int += 1,
            Value::Number(_) => self.float += 1,
            Value::Bool(_) => self.boolean += 1,
            Value::Object(map) => {
                self.object += 1;
                if depth < MAX_DEPTH {
                    for (key, child) in map {
                        self.children
                            .entry(key.clone())
                            .or_default()
                            .observe(child, depth + 1);
                    }
                }
            }
            Value::Array(_) | Value::Null => {}
        }
    }

    /// The Arrow type to shred this path as, if any.
    fn shredded_type(&self, total: u64) -> Option<DataType> {
        if (self.count as f64) < MIN_PRESENCE * total as f64 {
            return None;
        }
        let consistent = |n: u64| n as f64 >= MIN_TYPE_CONSISTENCY * self.count as f64;
        if consistent(self.string) {
            Some(DataType::Utf8)
        } else if consistent(self.int) {
            Some(DataType::Int64)
        } else if consistent(self.int + self.float) {
            Some(DataType::Float64)
        } else if consistent(self.boolean) {
            Some(DataType::Boolean)
        } else if consistent(self.object) {
            self.object_type(total)
        } else {
            None
        }
    }

    fn object_type(&self, total: u64) -> Option<DataType> {
        let fields: Fields = self
            .children
            .iter()
            .filter_map(|(name, child)| {
                child
                    .shredded_type(total)
                    .map(|dtype| Arc::new(Field::new(name, dtype, true)))
            })
            .collect();
        (!fields.is_empty()).then_some(DataType::Struct(fields))
    }
}

/// Infers the `typed_value` type to shred documents into, or `None` if nothing is worth
/// shredding.
pub fn infer_shredding_schema<'a>(
    documents: impl IntoIterator<Item = &'a str>,
) -> Option<DataType> {
    let mut root = PathStats::default();
    for document in documents {
        if let Ok(value) = serde_json::from_str::<Value>(document) {
            root.observe(&value, 0);
        }
    }
    root.object_type(root.count)
}

/// Renders a shredding type as a DuckDB `STRUCT(...)` type, for the DuckDB `SHREDDING` option.
pub fn duckdb_shredding_type(dtype: &DataType) -> String {
    match dtype {
        DataType::Utf8 => "VARCHAR".to_string(),
        DataType::Int64 => "BIGINT".to_string(),
        DataType::Float64 => "DOUBLE".to_string(),
        DataType::Boolean => "BOOLEAN".to_string(),
        DataType::Struct(fields) => format!(
            "STRUCT({})",
            fields
                .iter()
                .map(|f| format!("\"{}\" {}", f.name(), duckdb_shredding_type(f.data_type())))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => unreachable!("unexpected shredding type {other}"),
    }
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;

    use super::*;

    #[test]
    fn infers_consistent_paths() {
        let docs = [
            r#"{"a": 1, "b": {"c": "x", "d": 1}, "e": [1]}"#,
            r#"{"a": 2, "b": {"c": "y", "d": "mixed"}}"#,
            r#"{"a": 3, "b": {"c": "z"}}"#,
        ];
        let schema = infer_shredding_schema(docs).unwrap();
        assert_eq!(
            duckdb_shredding_type(&schema),
            r#"STRUCT("a" BIGINT, "b" STRUCT("c" VARCHAR))"#
        );
        assert!(matches!(schema, DataType::Struct(_)));
    }
}
