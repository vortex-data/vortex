use std::ffi::CString;

use vortex::dtype::{DType, PType};
use vortex::error::{VortexError, VortexResult, vortex_err};

use crate::cpp::{self, duckdb_logical_type};
use crate::duckdb::LogicalType;

impl LogicalType {
    fn struct_type<T, N>(child_types: T, child_names: N) -> VortexResult<LogicalType>
    where
        T: IntoIterator<Item = LogicalType>,
        N: IntoIterator<Item = CString>,
    {
        let child_types: Vec<LogicalType> = child_types.into_iter().collect();
        let child_names: Vec<CString> = child_names.into_iter().collect();

        let mut child_type_ptrs: Vec<duckdb_logical_type> =
            child_types.iter().map(|lt| lt.as_ptr()).collect();

        let mut child_name_ptrs: Vec<*const std::ffi::c_char> =
            child_names.iter().map(|name| name.as_ptr()).collect();

        let struct_type_ptr = unsafe {
            cpp::duckdb_create_struct_type(
                child_type_ptrs.as_mut_ptr(),
                child_name_ptrs.as_mut_ptr(),
                child_types.len() as _,
            )
        };

        if struct_type_ptr.is_null() {
            return Err(vortex_err!("Failed to create struct logical type"));
        }

        Ok(unsafe { LogicalType::own(struct_type_ptr) })
    }
}

impl TryFrom<&DType> for LogicalType {
    type Error = VortexError;

    fn try_from(dtype: &DType) -> Result<Self, Self::Error> {
        let duckdb_type = match dtype {
            DType::Null => cpp::DUCKDB_TYPE::DUCKDB_TYPE_SQLNULL,
            DType::Bool(_) => cpp::DUCKDB_TYPE::DUCKDB_TYPE_BOOLEAN,
            DType::Primitive(ptype, _) => match ptype {
                PType::I8 => cpp::DUCKDB_TYPE::DUCKDB_TYPE_TINYINT,
                PType::I16 => cpp::DUCKDB_TYPE::DUCKDB_TYPE_SMALLINT,
                PType::I32 => cpp::DUCKDB_TYPE::DUCKDB_TYPE_INTEGER,
                PType::I64 => cpp::DUCKDB_TYPE::DUCKDB_TYPE_BIGINT,
                PType::U8 => cpp::DUCKDB_TYPE::DUCKDB_TYPE_UTINYINT,
                PType::U16 => cpp::DUCKDB_TYPE::DUCKDB_TYPE_USMALLINT,
                PType::U32 => cpp::DUCKDB_TYPE::DUCKDB_TYPE_UINTEGER,
                PType::U64 => cpp::DUCKDB_TYPE::DUCKDB_TYPE_UBIGINT,
                PType::F32 => cpp::DUCKDB_TYPE::DUCKDB_TYPE_FLOAT,
                PType::F64 => cpp::DUCKDB_TYPE::DUCKDB_TYPE_DOUBLE,
                PType::F16 => return Err(vortex_err!("F16 type not supported in DuckDB")),
            },
            DType::Utf8(_) => cpp::DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR,
            DType::Binary(_) => cpp::DUCKDB_TYPE::DUCKDB_TYPE_BLOB,
            DType::Struct(struct_type, _) => {
                let child_types: Vec<LogicalType> = struct_type
                    .fields()
                    .map(|field_dtype| LogicalType::try_from(&field_dtype))
                    .collect::<Result<_, _>>()?;

                let child_names: Vec<CString> = struct_type
                    .names()
                    .iter()
                    .map(|field_name| {
                        CString::new(field_name.as_ref())
                            .map_err(|e| vortex_err!("Invalid field name '{field_name}': {e}"))
                    })
                    .collect::<Result<_, _>>()?;

                return LogicalType::struct_type(child_types, child_names);
            }
            // TODO: Add support for Decimal, Extension, List
            _ => {
                return Err(vortex_err!(
                    "Unsupported DType for DuckDB conversion: {:?}",
                    dtype
                ));
            }
        };

        Ok(LogicalType::new(duckdb_type))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex::dtype::{DType, FieldName, FieldNames, Nullability, PType, StructFields};

    use crate::cpp;
    use crate::duckdb::LogicalType;

    #[test]
    fn test_null_type() {
        let dtype = DType::Null;
        let logical_type = LogicalType::try_from(&dtype).unwrap();
        assert_eq!(
            logical_type.as_type_id(),
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_SQLNULL
        );
    }

    #[test]
    fn test_bool_type() {
        let dtype = DType::Bool(Nullability::NonNullable);
        let logical_type = LogicalType::try_from(&dtype).unwrap();
        assert_eq!(
            logical_type.as_type_id(),
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_BOOLEAN
        );

        let dtype_nullable = DType::Bool(Nullability::Nullable);
        let logical_type_nullable = LogicalType::try_from(&dtype_nullable).unwrap();
        assert_eq!(
            logical_type_nullable.as_type_id(),
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_BOOLEAN
        );
    }

    #[test]
    fn test_integer_types() {
        // Test signed and unsigned integers
        let test_cases = [
            (PType::I8, cpp::DUCKDB_TYPE::DUCKDB_TYPE_TINYINT),
            (PType::I16, cpp::DUCKDB_TYPE::DUCKDB_TYPE_SMALLINT),
            (PType::I32, cpp::DUCKDB_TYPE::DUCKDB_TYPE_INTEGER),
            (PType::I64, cpp::DUCKDB_TYPE::DUCKDB_TYPE_BIGINT),
            (PType::U8, cpp::DUCKDB_TYPE::DUCKDB_TYPE_UTINYINT),
            (PType::U16, cpp::DUCKDB_TYPE::DUCKDB_TYPE_USMALLINT),
            (PType::U32, cpp::DUCKDB_TYPE::DUCKDB_TYPE_UINTEGER),
            (PType::U64, cpp::DUCKDB_TYPE::DUCKDB_TYPE_UBIGINT),
        ];

        for (ptype, expected_duckdb_type) in test_cases {
            let dtype = DType::Primitive(ptype, Nullability::NonNullable);
            let logical_type = LogicalType::try_from(&dtype).unwrap();
            assert_eq!(logical_type.as_type_id(), expected_duckdb_type);
        }
    }

    #[test]
    fn test_float_types() {
        let float_test_cases = [
            (PType::F32, cpp::DUCKDB_TYPE::DUCKDB_TYPE_FLOAT),
            (PType::F64, cpp::DUCKDB_TYPE::DUCKDB_TYPE_DOUBLE),
        ];

        for (ptype, expected_duckdb_type) in float_test_cases {
            let dtype = DType::Primitive(ptype, Nullability::NonNullable);
            let logical_type = LogicalType::try_from(&dtype).unwrap();
            assert_eq!(logical_type.as_type_id(), expected_duckdb_type);
        }
    }

    #[test]
    fn test_f16_unsupported() {
        let dtype = DType::Primitive(PType::F16, Nullability::NonNullable);
        let result = LogicalType::try_from(&dtype);
        assert!(result.is_err());
    }

    #[test]
    fn test_string_type() {
        let dtype = DType::Utf8(Nullability::NonNullable);
        let logical_type = LogicalType::try_from(&dtype).unwrap();
        assert_eq!(
            logical_type.as_type_id(),
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR
        );

        let dtype_nullable = DType::Utf8(Nullability::Nullable);
        let logical_type_nullable = LogicalType::try_from(&dtype_nullable).unwrap();
        assert_eq!(
            logical_type_nullable.as_type_id(),
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR
        );
    }

    #[test]
    fn test_binary_type() {
        let dtype = DType::Binary(Nullability::NonNullable);
        let logical_type = LogicalType::try_from(&dtype).unwrap();
        assert_eq!(
            logical_type.as_type_id(),
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_BLOB
        );

        let dtype_nullable = DType::Binary(Nullability::Nullable);
        let logical_type_nullable = LogicalType::try_from(&dtype_nullable).unwrap();
        assert_eq!(
            logical_type_nullable.as_type_id(),
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_BLOB
        );
    }

    #[test]
    fn test_struct_type() {
        // Create a simple struct with two fields
        let field_names = FieldNames::from([FieldName::from("field1"), FieldName::from("field2")]);
        let field_types = vec![
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Utf8(Nullability::NonNullable),
        ];
        let struct_fields = StructFields::new(field_names, field_types);
        let dtype = DType::Struct(Arc::new(struct_fields), Nullability::NonNullable);
        let logical_type = LogicalType::try_from(&dtype).unwrap();

        assert_eq!(
            logical_type.as_type_id(),
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_STRUCT
        );
    }

    #[test]
    fn test_struct_with_invalid_field_name() {
        let field_names = FieldNames::from([FieldName::from("field\0with_null")]);
        let field_types = vec![DType::Primitive(PType::I32, Nullability::NonNullable)];
        let struct_fields = StructFields::new(field_names, field_types);
        let dtype = DType::Struct(Arc::new(struct_fields), Nullability::NonNullable);

        let result = LogicalType::try_from(&dtype);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_struct() {
        let struct_fields = StructFields::new([].into(), [].into());
        let dtype = DType::Struct(Arc::new(struct_fields), Nullability::NonNullable);

        let logical_type = LogicalType::try_from(&dtype).unwrap();
        assert_eq!(
            logical_type.as_type_id(),
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_STRUCT
        );
    }

    #[test]
    fn test_struct_with_different_types() {
        let field_names_expected = FieldNames::from([
            FieldName::from("i8_field"),
            FieldName::from("u16_field"),
            FieldName::from("f64_field"),
            FieldName::from("bool_field"),
            FieldName::from("string_field"),
            FieldName::from("binary_field"),
        ]);

        let field_types_expected = vec![
            DType::Primitive(PType::I8, Nullability::NonNullable),
            DType::Primitive(PType::U16, Nullability::NonNullable),
            DType::Primitive(PType::F64, Nullability::NonNullable),
            DType::Bool(Nullability::NonNullable),
            DType::Utf8(Nullability::NonNullable),
            DType::Binary(Nullability::NonNullable),
        ];

        let struct_fields =
            StructFields::new(field_names_expected.clone(), field_types_expected.clone());
        let dtype = DType::Struct(Arc::new(struct_fields), Nullability::NonNullable);

        assert_eq!(
            LogicalType::try_from(&dtype).unwrap().as_type_id(),
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_STRUCT
        );

        // Verify that the original struct fields are preserved.
        if let DType::Struct(ref struct_ref, _) = dtype {
            assert_eq!(struct_ref.names().len(), 6);
            assert_eq!(struct_ref.fields().len(), 6);

            for (idx, expected_name) in field_names_expected.iter().enumerate() {
                assert_eq!(struct_ref.names()[idx].as_ref(), expected_name.as_ref());
            }

            let field_types_actual: Vec<DType> = struct_ref.fields().collect();
            assert_eq!(field_types_expected, field_types_actual);
        }
    }

    #[test]
    fn test_nullable_vs_non_nullable() {
        // Test that nullability doesn't affect the logical type conversion.
        let nullable_i32 = DType::Primitive(PType::I32, Nullability::Nullable);
        let non_nullable_i32 = DType::Primitive(PType::I32, Nullability::NonNullable);

        let nullable_logical_type = LogicalType::try_from(&nullable_i32).unwrap();
        let non_nullable_logical_type = LogicalType::try_from(&non_nullable_i32).unwrap();

        assert_eq!(
            nullable_logical_type.as_type_id(),
            non_nullable_logical_type.as_type_id()
        );
        assert_eq!(
            nullable_logical_type.as_type_id(),
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_INTEGER
        );
    }
}
