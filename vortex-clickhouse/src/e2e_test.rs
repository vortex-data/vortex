// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end tests for vortex-clickhouse.
//!
//! These tests verify the full pipeline of reading and writing Vortex files
//! through ClickHouse.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex::dtype::Nullability::{NonNullable, Nullable};
    use vortex::dtype::{DType, FieldName, FieldNames, PType, StructFields};

    use crate::convert::dtype::{clickhouse_type_to_vortex, vortex_to_clickhouse_type};

    #[test]
    fn test_type_roundtrip() {
        // Test that types can be converted back and forth
        let test_types = vec![
            ("Int32", DType::Primitive(PType::I32, NonNullable)),
            ("UInt64", DType::Primitive(PType::U64, NonNullable)),
            ("Float64", DType::Primitive(PType::F64, NonNullable)),
            ("String", DType::Utf8(NonNullable)),
            ("Bool", DType::Bool(NonNullable)),
        ];

        for (ch_type, expected_dtype) in test_types {
            let converted = clickhouse_type_to_vortex(ch_type).unwrap();
            assert_eq!(
                converted, expected_dtype,
                "Failed for ClickHouse type: {}",
                ch_type
            );
        }
    }

    #[test]
    fn test_vortex_to_clickhouse_types() {
        assert_eq!(
            vortex_to_clickhouse_type(&DType::Primitive(PType::I32, NonNullable)).unwrap(),
            "Int32"
        );
        assert_eq!(
            vortex_to_clickhouse_type(&DType::Primitive(PType::I32, Nullable)).unwrap(),
            "Nullable(Int32)"
        );
        assert_eq!(
            vortex_to_clickhouse_type(&DType::Utf8(NonNullable)).unwrap(),
            "String"
        );
    }

    /// Test a realistic table schema typical for analytics workloads.
    #[test]
    fn test_analytics_table_schema() {
        // Simulate a typical ClickHouse analytics table schema
        let ch_schema = vec![
            ("event_id", "UInt64"),
            ("event_time", "DateTime64(3)"),
            ("user_id", "Nullable(UInt64)"),
            ("event_type", "LowCardinality(String)"),
            ("properties", "String"),
            ("amount", "Nullable(Float64)"),
        ];

        for (col_name, ch_type) in ch_schema {
            let result = clickhouse_type_to_vortex(ch_type);
            assert!(
                result.is_ok(),
                "Failed to convert column '{}' with type '{}'",
                col_name,
                ch_type
            );
        }
    }

    /// Test a realistic log table schema.
    #[test]
    fn test_log_table_schema() {
        let ch_schema = vec![
            ("timestamp", "DateTime64(6, 'UTC')"),
            ("level", "LowCardinality(String)"),
            ("service", "String"),
            ("trace_id", "UUID"),
            ("message", "String"),
            ("tags", "Array(String)"),
            (
                "metadata",
                "Tuple(host String, pod String, container String)",
            ),
        ];

        for (col_name, ch_type) in ch_schema {
            let result = clickhouse_type_to_vortex(ch_type);
            assert!(
                result.is_ok(),
                "Failed to convert column '{}' with type '{}': {:?}",
                col_name,
                ch_type,
                result.err()
            );
        }
    }

    /// Test time-series data schema.
    #[test]
    fn test_timeseries_schema() {
        let ch_schema = vec![
            ("metric_name", "String"),
            ("timestamp", "DateTime"),
            ("value", "Float64"),
            ("tags", "Array(String)"),
        ];

        for (col_name, ch_type) in ch_schema {
            let result = clickhouse_type_to_vortex(ch_type);
            assert!(
                result.is_ok(),
                "Failed to convert column '{}' with type '{}'",
                col_name,
                ch_type
            );
        }
    }

    /// Test building a full Vortex schema and converting to ClickHouse.
    #[test]
    fn test_full_schema_conversion() {
        // Build a Vortex schema representing a user events table
        let names = FieldNames::from(vec![
            FieldName::from("user_id"),
            FieldName::from("event_name"),
            FieldName::from("timestamp"),
            FieldName::from("properties"),
            FieldName::from("tags"),
        ]);

        let dtypes = vec![
            DType::Primitive(PType::I64, NonNullable),
            DType::Utf8(NonNullable),
            DType::Primitive(PType::I64, NonNullable), // Unix timestamp
            DType::Utf8(Nullable),                     // JSON properties
            DType::List(Arc::new(DType::Utf8(NonNullable)), Nullable),
        ];

        let schema = DType::Struct(StructFields::new(names, dtypes), NonNullable);

        let ch_type = vortex_to_clickhouse_type(&schema).unwrap();
        assert!(ch_type.starts_with("Tuple("));
        assert!(ch_type.contains("user_id Int64"));
        assert!(ch_type.contains("event_name String"));
        assert!(ch_type.contains("tags Nullable(Array(String))"));
    }

    /// Test nested struct conversion for hierarchical data.
    #[test]
    fn test_nested_data_schema() {
        // Address nested struct
        let address_names =
            FieldNames::from(vec![FieldName::from("city"), FieldName::from("country")]);
        let address_dtypes = vec![DType::Utf8(NonNullable), DType::Utf8(NonNullable)];
        let address_struct = DType::Struct(
            StructFields::new(address_names, address_dtypes),
            NonNullable,
        );

        // User struct containing address
        let user_names = FieldNames::from(vec![
            FieldName::from("id"),
            FieldName::from("name"),
            FieldName::from("address"),
        ]);
        let user_dtypes = vec![
            DType::Primitive(PType::I64, NonNullable),
            DType::Utf8(NonNullable),
            address_struct,
        ];
        let user_struct = DType::Struct(StructFields::new(user_names, user_dtypes), NonNullable);

        let ch_type = vortex_to_clickhouse_type(&user_struct).unwrap();
        assert_eq!(
            ch_type,
            "Tuple(id Int64, name String, address Tuple(city String, country String))"
        );
    }

    /// Test array of structs (common pattern in ClickHouse).
    #[test]
    fn test_array_of_structs() {
        // Item struct
        let item_names = FieldNames::from(vec![
            FieldName::from("product_id"),
            FieldName::from("quantity"),
            FieldName::from("price"),
        ]);
        let item_dtypes = vec![
            DType::Primitive(PType::I64, NonNullable),
            DType::Primitive(PType::I32, NonNullable),
            DType::Primitive(PType::F64, NonNullable),
        ];
        let item_struct = DType::Struct(StructFields::new(item_names, item_dtypes), NonNullable);

        // Array of items
        let items_list = DType::List(Arc::new(item_struct), NonNullable);

        let ch_type = vortex_to_clickhouse_type(&items_list).unwrap();
        assert_eq!(
            ch_type,
            "Array(Tuple(product_id Int64, quantity Int32, price Float64))"
        );
    }

    /// Test all primitive types roundtrip.
    #[test]
    fn test_all_primitives_roundtrip() {
        let ch_primitives = vec![
            "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float32",
            "Float64", "Bool", "String",
        ];

        for ch_type in ch_primitives {
            let vortex_dtype = clickhouse_type_to_vortex(ch_type).unwrap();
            let back_to_ch = vortex_to_clickhouse_type(&vortex_dtype).unwrap();
            // NonNullable types roundtrip back to the same type string
            assert_eq!(
                back_to_ch, ch_type,
                "Roundtrip failed for {}: got {}",
                ch_type, back_to_ch
            );
        }
    }

    /// Test empty struct handling.
    #[test]
    fn test_empty_struct() {
        let empty_struct = DType::Struct(
            StructFields::new(FieldNames::from(Vec::<FieldName>::new()), vec![]),
            NonNullable,
        );
        let ch_type = vortex_to_clickhouse_type(&empty_struct).unwrap();
        assert_eq!(ch_type, "Tuple()");
    }

    /// Test single-field struct.
    #[test]
    fn test_single_field_struct() {
        let names = FieldNames::from(vec![FieldName::from("value")]);
        let dtypes = vec![DType::Primitive(PType::I64, NonNullable)];
        let single_struct = DType::Struct(StructFields::new(names, dtypes), NonNullable);

        let ch_type = vortex_to_clickhouse_type(&single_struct).unwrap();
        assert_eq!(ch_type, "Tuple(value Int64)");
    }

    /// Test deeply nested array.
    #[test]
    fn test_deeply_nested_array() {
        // Array(Array(Array(Int32)))
        let dtype = clickhouse_type_to_vortex("Array(Array(Array(Int32)))").unwrap();

        // Verify it's three levels deep
        if let DType::List(l1, _) = dtype {
            if let DType::List(l2, _) = l1.as_ref() {
                if let DType::List(l3, _) = l2.as_ref() {
                    assert!(matches!(
                        l3.as_ref(),
                        DType::Primitive(PType::I32, NonNullable)
                    ));
                } else {
                    panic!("Expected third level to be List");
                }
            } else {
                panic!("Expected second level to be List");
            }
        } else {
            panic!("Expected first level to be List");
        }
    }
}
