// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Logical type conversion between Vortex and DuckDB.
//!
//! This module provides functionality to convert Vortex data types (`DType`) to DuckDB logical types.
//!
//! Note that nullability of Vortex logical types is not transferred to DuckDB logical types.
//!
//! # Supported Type Mappings
//!
//! | Vortex Type | DuckDB Type |
//! |-------------|-------------|
//! | `Null` | `SQLNULL` |
//! | `Bool` | `BOOLEAN` |
//! | `I8/U8` | `TINYINT/UTINYINT` |
//! | `I16/U16` | `SMALLINT/USMALLINT` |
//! | `I32/U32` | `INTEGER/UINTEGER` |
//! | `I64/U64` | `BIGINT/UBIGINT` |
//! | `F32` | `FLOAT` |
//! | `F64` | `DOUBLE` |
//! | `Utf8` | `VARCHAR` |
//! | `Binary` | `BLOB` |
//! | `Struct` | `STRUCT` |
//! | `Decimal` | `DECIMAL` |
//! | `List` | `LIST` |
//! | `Date` | `DATE` |
//! | `Time` | `TIME` |
//! | `Timestamp` | `TIMESTAMP` |

use std::ffi::CString;
use std::sync::Arc;

use vortex::dtype::PType::{F32, F64, I8, I16, I32, I64, U8, U16, U32, U64};
use vortex::dtype::datetime::{DATE_ID, TIME_ID, TIMESTAMP_ID, TemporalMetadata, TimeUnit};
use vortex::dtype::{
    DType, DecimalDType, ExtDType, FieldName, Nullability, PType, StructFields, datetime,
};
use vortex::error::{VortexError, VortexResult, vortex_bail, vortex_err};

use crate::cpp::{self, DUCKDB_TYPE, duckdb_logical_type};
use crate::duckdb::LogicalType;

impl LogicalType {
    /// Creates a DuckDB struct logical type from child types and field names.
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

        Ok(unsafe { Self::own(struct_type_ptr) })
    }

    /// Creates a DuckDB decimal logical type with the specified precision and scale.
    fn decimal_type(precision: u8, scale: u8) -> VortexResult<Self> {
        assert!(
            precision <= 38,
            "DuckDB decimal type precision must be <= 38. precision: {precision}"
        );

        unsafe {
            let ptr = cpp::duckdb_create_decimal_type(precision, scale);
            if ptr.is_null() {
                return Err(vortex_err!("Failed to create decimal type"));
            }
            Ok(Self::own(ptr))
        }
    }

    /// Creates a DuckDB list logical type with the specified element type.
    fn list_type(element_type: LogicalType) -> VortexResult<Self> {
        unsafe {
            let ptr = cpp::duckdb_create_list_type(element_type.as_ptr());
            if ptr.is_null() {
                return Err(vortex_err!("Failed to create list type"));
            }
            Ok(Self::own(ptr))
        }
    }

    /// Converts temporal extension types to corresponding DuckDB types.
    ///
    /// # Arguments
    ///
    /// * `ext_dtype` - A reference to the extension data type containing temporal metadata.
    ///
    /// # Supported Temporal Types
    ///
    /// - **Date**: Must use `TimeUnit::D`
    /// - **Time**: Must use `TimeUnit::Us`
    /// - **Timestamp**: Supports `TimeUnit::Ns`, `Us`, `Ms`, `S`
    fn temporal_type(ext_dtype: &ExtDType) -> VortexResult<Self> {
        use vortex::dtype::datetime::{TemporalMetadata, TimeUnit};

        let temporal_metadata = TemporalMetadata::try_from(ext_dtype)
            .map_err(|e| vortex_err!("Failed to extract temporal metadata: {}", e))?;

        let duckdb_type = match temporal_metadata {
            TemporalMetadata::Date(TimeUnit::D) => DUCKDB_TYPE::DUCKDB_TYPE_DATE,
            TemporalMetadata::Date(time_unit) => {
                return Err(vortex_err!("Invalid TimeUnit {} for date", time_unit));
            }
            TemporalMetadata::Time(TimeUnit::Us) => DUCKDB_TYPE::DUCKDB_TYPE_TIME,
            TemporalMetadata::Time(time_unit) => {
                return Err(vortex_err!("Invalid TimeUnit {} for time", time_unit));
            }
            TemporalMetadata::Timestamp(time_unit, tz) => {
                if tz.is_some() {
                    return Err(vortex_err!("Timestamp with timezone is not yet supported"));
                }
                match time_unit {
                    TimeUnit::Ns => DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_NS,
                    TimeUnit::Us => DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP,
                    TimeUnit::Ms => DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_MS,
                    TimeUnit::S => DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_S,
                    _ => return Err(vortex_err!("Invalid TimeUnit {} for timestamp", time_unit)),
                }
            }
        };

        Ok(Self::new(duckdb_type))
    }
}

pub trait FromLogicalType {
    fn from_logical_type(
        logical_type: LogicalType,
        nullability: Nullability,
    ) -> VortexResult<DType>;
}

impl FromLogicalType for DType {
    fn from_logical_type(
        logical_type: LogicalType,
        nullability: Nullability,
    ) -> VortexResult<DType> {
        Ok(match logical_type.as_type_id() {
            DUCKDB_TYPE::DUCKDB_TYPE_INVALID => vortex_bail!("invalid duckdb type"),
            DUCKDB_TYPE::DUCKDB_TYPE_SQLNULL => DType::Null,
            DUCKDB_TYPE::DUCKDB_TYPE_BOOLEAN => DType::Bool(nullability),
            DUCKDB_TYPE::DUCKDB_TYPE_TINYINT => DType::Primitive(I8, nullability),
            DUCKDB_TYPE::DUCKDB_TYPE_SMALLINT => DType::Primitive(I16, nullability),
            DUCKDB_TYPE::DUCKDB_TYPE_INTEGER => DType::Primitive(I32, nullability),
            DUCKDB_TYPE::DUCKDB_TYPE_BIGINT => DType::Primitive(I64, nullability),
            DUCKDB_TYPE::DUCKDB_TYPE_UTINYINT => DType::Primitive(U8, nullability),
            DUCKDB_TYPE::DUCKDB_TYPE_USMALLINT => DType::Primitive(U16, nullability),
            DUCKDB_TYPE::DUCKDB_TYPE_UINTEGER => DType::Primitive(U32, nullability),
            DUCKDB_TYPE::DUCKDB_TYPE_UBIGINT => DType::Primitive(U64, nullability),
            DUCKDB_TYPE::DUCKDB_TYPE_HUGEINT => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_UHUGEINT => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_FLOAT => DType::Primitive(F32, nullability),
            DUCKDB_TYPE::DUCKDB_TYPE_DOUBLE => DType::Primitive(F64, nullability),

            DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR => DType::Utf8(nullability),
            DUCKDB_TYPE::DUCKDB_TYPE_BLOB => DType::Binary(nullability),
            DUCKDB_TYPE::DUCKDB_TYPE_DECIMAL => {
                let (width, scale) = logical_type.as_decimal();
                DType::Decimal(DecimalDType::new(width, scale.try_into()?), nullability)
            }
            DUCKDB_TYPE::DUCKDB_TYPE_DATE => DType::Extension(Arc::new(ExtDType::new(
                DATE_ID.clone(),
                Arc::new(DType::Primitive(I32, nullability)),
                Some(TemporalMetadata::Date(TimeUnit::D).into()),
            ))),
            DUCKDB_TYPE::DUCKDB_TYPE_TIME => DType::Extension(Arc::new(ExtDType::new(
                TIME_ID.clone(),
                Arc::new(DType::Primitive(I64, nullability)),
                Some(TemporalMetadata::Date(TimeUnit::Us).into()),
            ))),
            DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_S => DType::Extension(Arc::new(ExtDType::new(
                TIMESTAMP_ID.clone(),
                Arc::new(DType::Primitive(I32, nullability)),
                Some(TemporalMetadata::Timestamp(TimeUnit::S, None).into()),
            ))),
            DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_MS => DType::Extension(Arc::new(ExtDType::new(
                TIMESTAMP_ID.clone(),
                Arc::new(DType::Primitive(I32, nullability)),
                Some(TemporalMetadata::Timestamp(TimeUnit::Ms, None).into()),
            ))),
            DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP => DType::Extension(Arc::new(ExtDType::new(
                TIMESTAMP_ID.clone(),
                Arc::new(DType::Primitive(I64, nullability)),
                Some(TemporalMetadata::Timestamp(TimeUnit::Us, None).into()),
            ))),
            DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_NS => DType::Extension(Arc::new(ExtDType::new(
                TIMESTAMP_ID.clone(),
                Arc::new(DType::Primitive(I64, nullability)),
                Some(TemporalMetadata::Timestamp(TimeUnit::Ns, None).into()),
            ))),
            DUCKDB_TYPE::DUCKDB_TYPE_INTERVAL => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_ENUM => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_LIST => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_STRUCT => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_MAP => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_ARRAY => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_UUID => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_UNION => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_BIT => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_TIME_TZ => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_TZ => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_ANY => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_VARINT => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_STRING_LITERAL => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_INTEGER_LITERAL => todo!(),
        })
    }
}

pub fn from_duckdb_table<I, S>(iter: I) -> VortexResult<StructFields>
where
    I: Iterator<Item = (S, LogicalType, Nullability)>,
    S: AsRef<str>,
{
    iter.map(|(name, type_, nullability)| {
        Ok((
            FieldName::from(name.as_ref()),
            DType::from_logical_type(type_, nullability)?,
        ))
    })
    .collect::<VortexResult<StructFields>>()
}

impl TryFrom<&DType> for LogicalType {
    type Error = VortexError;

    /// Converts a Vortex data type to a DuckDB logical type.
    ///
    /// This is the main conversion function that handles all supported Vortex data types
    /// and maps them to their corresponding DuckDB logical types.
    ///
    /// # Arguments
    ///
    /// * `dtype` - A reference to the Vortex data type to convert
    ///
    /// # Returns
    ///
    /// A `Result` containing the DuckDB logical type or a conversion error.
    fn try_from(dtype: &DType) -> Result<Self, Self::Error> {
        let duckdb_type = match dtype {
            DType::Null => DUCKDB_TYPE::DUCKDB_TYPE_SQLNULL,
            DType::Bool(_) => DUCKDB_TYPE::DUCKDB_TYPE_BOOLEAN,
            DType::Primitive(ptype, _) => match ptype {
                I8 => DUCKDB_TYPE::DUCKDB_TYPE_TINYINT,
                I16 => DUCKDB_TYPE::DUCKDB_TYPE_SMALLINT,
                I32 => DUCKDB_TYPE::DUCKDB_TYPE_INTEGER,
                I64 => DUCKDB_TYPE::DUCKDB_TYPE_BIGINT,
                U8 => DUCKDB_TYPE::DUCKDB_TYPE_UTINYINT,
                U16 => DUCKDB_TYPE::DUCKDB_TYPE_USMALLINT,
                U32 => DUCKDB_TYPE::DUCKDB_TYPE_UINTEGER,
                U64 => DUCKDB_TYPE::DUCKDB_TYPE_UBIGINT,
                F32 => DUCKDB_TYPE::DUCKDB_TYPE_FLOAT,
                F64 => DUCKDB_TYPE::DUCKDB_TYPE_DOUBLE,
                PType::F16 => return Err(vortex_err!("F16 type not supported in DuckDB")),
            },
            DType::Utf8(_) => DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR,
            DType::Binary(_) => DUCKDB_TYPE::DUCKDB_TYPE_BLOB,
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
            DType::Decimal(decimal_dtype, _) => {
                return LogicalType::decimal_type(
                    decimal_dtype.precision(),
                    decimal_dtype.scale().try_into()?,
                );
            }
            DType::List(element_dtype, _) => {
                let element_logical_type = LogicalType::try_from(element_dtype.as_ref())?;
                return LogicalType::list_type(element_logical_type);
            }
            DType::Extension(ext_dtype) => {
                if datetime::is_temporal_ext_type(ext_dtype.id()) {
                    return LogicalType::temporal_type(ext_dtype);
                } else {
                    vortex_bail!("Unsupported extension type \"{}\"", ext_dtype.id())
                }
            }
        };

        Ok(LogicalType::new(duckdb_type))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
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
    }

    #[rstest]
    #[case(PType::I8, cpp::DUCKDB_TYPE::DUCKDB_TYPE_TINYINT)]
    #[case(PType::I16, cpp::DUCKDB_TYPE::DUCKDB_TYPE_SMALLINT)]
    #[case(PType::I32, cpp::DUCKDB_TYPE::DUCKDB_TYPE_INTEGER)]
    #[case(PType::I64, cpp::DUCKDB_TYPE::DUCKDB_TYPE_BIGINT)]
    #[case(PType::U8, cpp::DUCKDB_TYPE::DUCKDB_TYPE_UTINYINT)]
    #[case(PType::U16, cpp::DUCKDB_TYPE::DUCKDB_TYPE_USMALLINT)]
    #[case(PType::U32, cpp::DUCKDB_TYPE::DUCKDB_TYPE_UINTEGER)]
    #[case(PType::U64, cpp::DUCKDB_TYPE::DUCKDB_TYPE_UBIGINT)]
    #[case(PType::F32, cpp::DUCKDB_TYPE::DUCKDB_TYPE_FLOAT)]
    #[case(PType::F64, cpp::DUCKDB_TYPE::DUCKDB_TYPE_DOUBLE)]
    fn test_primitive_types(#[case] ptype: PType, #[case] expected_duckdb_type: cpp::DUCKDB_TYPE) {
        let dtype = DType::Primitive(ptype, Nullability::NonNullable);
        let logical_type = LogicalType::try_from(&dtype).unwrap();
        assert_eq!(logical_type.as_type_id(), expected_duckdb_type);
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
    }

    #[test]
    fn test_binary_type() {
        let dtype = DType::Binary(Nullability::NonNullable);
        let logical_type = LogicalType::try_from(&dtype).unwrap();
        assert_eq!(
            logical_type.as_type_id(),
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_BLOB
        );
    }

    #[test]
    fn test_struct_type() {
        let field_names = FieldNames::from([FieldName::from("field1"), FieldName::from("field2")]);
        let field_types = vec![
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Utf8(Nullability::NonNullable),
        ];
        let struct_fields = StructFields::new(field_names, field_types);
        let dtype = DType::Struct(struct_fields, Nullability::NonNullable);
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
        let dtype = DType::Struct(struct_fields, Nullability::NonNullable);

        let result = LogicalType::try_from(&dtype);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_struct() {
        let struct_fields = StructFields::new(FieldNames::default(), [].into());
        let dtype = DType::Struct(struct_fields, Nullability::NonNullable);

        let logical_type = LogicalType::try_from(&dtype).unwrap();
        assert_eq!(
            logical_type.as_type_id(),
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_STRUCT
        );
    }

    #[test]
    fn test_decimal_type() {
        use vortex::dtype::DecimalDType;
        let decimal_dtype = DecimalDType::new(18, 4);
        let dtype = DType::Decimal(decimal_dtype, Nullability::NonNullable);
        let logical_type = LogicalType::try_from(&dtype).unwrap();

        assert_eq!(
            logical_type.as_type_id(),
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_DECIMAL
        );
    }

    #[test]
    fn test_list_type() {
        let element_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let dtype = DType::List(Arc::new(element_dtype), Nullability::NonNullable);
        let logical_type = LogicalType::try_from(&dtype).unwrap();

        assert_eq!(
            logical_type.as_type_id(),
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_LIST
        );
    }

    #[test]
    fn test_date_extension_type() {
        use std::sync::Arc;

        use vortex::dtype::datetime::{DATE_ID, TemporalMetadata, TimeUnit};
        use vortex::dtype::{ExtDType, PType};

        let ext_dtype = ExtDType::new(
            DATE_ID.clone(),
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            Some(TemporalMetadata::Date(TimeUnit::D).into()),
        );
        let dtype = DType::Extension(Arc::new(ext_dtype));
        let logical_type = LogicalType::try_from(&dtype).unwrap();

        assert_eq!(
            logical_type.as_type_id(),
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_DATE
        );
    }

    #[test]
    fn test_time_extension_type() {
        use std::sync::Arc;

        use vortex::dtype::datetime::{TIME_ID, TemporalMetadata, TimeUnit};
        use vortex::dtype::{ExtDType, PType};

        let ext_dtype = ExtDType::new(
            TIME_ID.clone(),
            Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
            Some(TemporalMetadata::Time(TimeUnit::Us).into()),
        );
        let dtype = DType::Extension(Arc::new(ext_dtype));
        let logical_type = LogicalType::try_from(&dtype).unwrap();

        assert_eq!(
            logical_type.as_type_id(),
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_TIME
        );
    }

    #[test]
    fn test_timestamp_extension_types() {
        use std::sync::Arc;

        use vortex::dtype::datetime::{TIMESTAMP_ID, TemporalMetadata, TimeUnit};
        use vortex::dtype::{ExtDType, PType};

        let test_cases = [
            (TimeUnit::Ns, cpp::DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_NS),
            (TimeUnit::Us, cpp::DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP),
            (TimeUnit::Ms, cpp::DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_MS),
            (TimeUnit::S, cpp::DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_S),
        ];

        for (time_unit, expected_type) in test_cases {
            let ext_dtype = ExtDType::new(
                TIMESTAMP_ID.clone(),
                Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
                Some(TemporalMetadata::Timestamp(time_unit, None).into()),
            );
            let dtype = DType::Extension(Arc::new(ext_dtype));
            let logical_type = LogicalType::try_from(&dtype).unwrap();

            assert_eq!(logical_type.as_type_id(), expected_type);
        }
    }

    #[test]
    fn test_temporal_extension_invalid_time_units() {
        use std::sync::Arc;

        use vortex::dtype::datetime::{DATE_ID, TIME_ID, TemporalMetadata, TimeUnit};
        use vortex::dtype::{ExtDType, PType};

        // Invalid DATE time unit
        let ext_dtype = ExtDType::new(
            DATE_ID.clone(),
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            Some(TemporalMetadata::Date(TimeUnit::Ms).into()),
        );
        let dtype = DType::Extension(Arc::new(ext_dtype));
        assert!(LogicalType::try_from(&dtype).is_err());

        // Invalid TIME time unit
        let ext_dtype = ExtDType::new(
            TIME_ID.clone(),
            Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
            Some(TemporalMetadata::Time(TimeUnit::Ms).into()),
        );
        let dtype = DType::Extension(Arc::new(ext_dtype));
        assert!(LogicalType::try_from(&dtype).is_err());
    }

    #[test]
    fn test_timestamp_with_timezone_unsupported() {
        use std::sync::Arc;

        use vortex::dtype::datetime::{TIMESTAMP_ID, TemporalMetadata, TimeUnit};
        use vortex::dtype::{ExtDType, PType};

        let ext_dtype = ExtDType::new(
            TIMESTAMP_ID.clone(),
            Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
            Some(TemporalMetadata::Timestamp(TimeUnit::Us, Some("UTC".to_string())).into()),
        );
        let dtype = DType::Extension(Arc::new(ext_dtype));

        assert!(LogicalType::try_from(&dtype).is_err());
    }

    #[test]
    fn test_unsupported_extension_type() {
        use vortex::dtype::{ExtDType, ExtID, PType};

        let ext_dtype = ExtDType::new(
            ExtID::from("unknown.extension"),
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            None,
        );
        let dtype = DType::Extension(Arc::new(ext_dtype));

        assert!(LogicalType::try_from(&dtype).is_err());
    }
}
