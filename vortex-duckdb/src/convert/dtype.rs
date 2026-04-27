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

use vortex::dtype::DType;
use vortex::dtype::DecimalDType;
use vortex::dtype::FieldName;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::dtype::PType::F32;
use vortex::dtype::PType::F64;
use vortex::dtype::PType::I8;
use vortex::dtype::PType::I16;
use vortex::dtype::PType::I32;
use vortex::dtype::PType::I64;
use vortex::dtype::PType::U8;
use vortex::dtype::PType::U16;
use vortex::dtype::PType::U32;
use vortex::dtype::PType::U64;
use vortex::dtype::StructFields;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::extension::datetime::AnyTemporal;
use vortex::extension::datetime::Date;
use vortex::extension::datetime::TemporalMetadata;
use vortex::extension::datetime::Time;
use vortex::extension::datetime::TimeUnit;
use vortex::extension::datetime::Timestamp;

use crate::cpp::DUCKDB_TYPE;
use crate::duckdb::LogicalType;
use crate::duckdb::LogicalTypeRef;

pub trait FromLogicalType {
    fn from_logical_type(
        logical_type: &LogicalTypeRef,
        nullability: Nullability,
    ) -> VortexResult<DType>;
}

impl FromLogicalType for DType {
    fn from_logical_type(
        logical_type: &LogicalTypeRef,
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
                DType::Decimal(
                    DecimalDType::try_new(width, scale.try_into()?)?,
                    nullability,
                )
            }
            DUCKDB_TYPE::DUCKDB_TYPE_DATE => {
                DType::Extension(Date::new(TimeUnit::Days, nullability).erased())
            }
            DUCKDB_TYPE::DUCKDB_TYPE_TIME => {
                DType::Extension(Time::new(TimeUnit::Microseconds, nullability).erased())
            }
            DUCKDB_TYPE::DUCKDB_TYPE_TIME_NS => {
                DType::Extension(Time::new(TimeUnit::Nanoseconds, nullability).erased())
            }
            DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_S => {
                DType::Extension(Timestamp::new(TimeUnit::Seconds, nullability).erased())
            }
            DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_MS => {
                DType::Extension(Timestamp::new(TimeUnit::Milliseconds, nullability).erased())
            }
            DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP => {
                DType::Extension(Timestamp::new(TimeUnit::Microseconds, nullability).erased())
            }
            // NOTE(ngates): DuckDB's TIMESTAMP_TZ does not actually have configurable timezones.
            //  Instead, it implies the values is timezone-aware and stored in UTC.
            DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_TZ => DType::Extension(
                Timestamp::new_with_tz(TimeUnit::Microseconds, Some("UTC".into()), nullability)
                    .erased(),
            ),
            DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_NS => {
                DType::Extension(Timestamp::new(TimeUnit::Nanoseconds, nullability).erased())
            }
            DUCKDB_TYPE::DUCKDB_TYPE_ARRAY => {
                let child_type = logical_type.array_child_type();
                DType::FixedSizeList(
                    Arc::new(DType::from_logical_type(
                        &child_type,
                        Nullability::Nullable,
                    )?),
                    logical_type.array_type_array_size(),
                    nullability,
                )
            }
            DUCKDB_TYPE::DUCKDB_TYPE_LIST => {
                let child_type = logical_type.list_child_type();
                DType::List(
                    Arc::new(DType::from_logical_type(
                        &child_type,
                        Nullability::Nullable,
                    )?),
                    nullability,
                )
            }
            DUCKDB_TYPE::DUCKDB_TYPE_STRUCT => DType::Struct(
                (0..logical_type.struct_type_child_count())
                    .map(|i| {
                        let child_name = logical_type.struct_child_name(i);
                        let child_type = logical_type.struct_child_type(i);
                        Ok((
                            child_name,
                            DType::from_logical_type(&child_type, Nullability::Nullable)?,
                        ))
                    })
                    .collect::<VortexResult<_>>()?,
                nullability,
            ),
            DUCKDB_TYPE::DUCKDB_TYPE_TIME_TZ => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_INTERVAL => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_ENUM => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_MAP => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_UUID => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_UNION => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_BIT => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_ANY => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_BIGNUM => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_STRING_LITERAL => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_INTEGER_LITERAL => todo!(),
            DUCKDB_TYPE::DUCKDB_TYPE_GEOMETRY => todo!(),
        })
    }
}

pub fn from_duckdb_table<'a, I, S>(iter: I) -> VortexResult<StructFields>
where
    I: Iterator<Item = (S, &'a LogicalTypeRef, Nullability)>,
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

impl TryFrom<DType> for LogicalType {
    type Error = VortexError;

    fn try_from(value: DType) -> Result<Self, Self::Error> {
        LogicalType::try_from(&value)
    }
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
            DType::Primitive(ptype, _) => return LogicalType::try_from(*ptype),
            DType::Utf8(_) => DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR,
            DType::Binary(_) => DUCKDB_TYPE::DUCKDB_TYPE_BLOB,
            DType::Struct(struct_type, _) => {
                return LogicalType::try_from(struct_type);
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
            DType::FixedSizeList(element_dtype, list_size, _) => {
                let element_logical_type = LogicalType::try_from(element_dtype.as_ref())?;
                return LogicalType::array_type(element_logical_type, *list_size);
            }
            DType::Variant(_) => {
                vortex_bail!("Vortex Variant array aren't supported in DuckDB")
            }
            DType::Extension(ext_dtype) => {
                let Some(temporal) = ext_dtype.metadata_opt::<AnyTemporal>() else {
                    vortex_bail!("Unsupported extension type \"{}\"", ext_dtype.id());
                };

                match temporal {
                    TemporalMetadata::Timestamp(unit, None) => match unit {
                        TimeUnit::Nanoseconds => DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_NS,
                        TimeUnit::Microseconds => DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP,
                        TimeUnit::Milliseconds => DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_MS,
                        TimeUnit::Seconds => DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_S,
                        _ => vortex_bail!("Invalid TimeUnit {} for timestamp", unit),
                    },
                    TemporalMetadata::Timestamp(unit, Some(tz)) => {
                        if tz.as_ref() != "UTC" {
                            vortex_bail!("Invalid timezone for timestamp_tz {tz}, must be UTC");
                        }
                        if unit != &TimeUnit::Microseconds {
                            vortex_bail!(
                                "Invalid TimeUnit {} for timestamp_tz, must be Microseconds",
                                unit
                            );
                        }
                        DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_TZ
                    }
                    TemporalMetadata::Date(unit) => match unit {
                        TimeUnit::Days => DUCKDB_TYPE::DUCKDB_TYPE_DATE,
                        _ => vortex_bail!("Invalid TimeUnit {} for date", unit),
                    },
                    TemporalMetadata::Time(unit) => match unit {
                        TimeUnit::Microseconds => DUCKDB_TYPE::DUCKDB_TYPE_TIME,
                        TimeUnit::Nanoseconds => DUCKDB_TYPE::DUCKDB_TYPE_TIME_NS,
                        _ => vortex_bail!("Invalid TimeUnit {} for time", unit),
                    },
                }
            }
        };

        Ok(LogicalType::new(duckdb_type))
    }
}

impl TryFrom<StructFields> for LogicalType {
    type Error = VortexError;

    fn try_from(struct_type: StructFields) -> Result<Self, Self::Error> {
        LogicalType::try_from(&struct_type)
    }
}

impl TryFrom<&StructFields> for LogicalType {
    type Error = VortexError;

    fn try_from(struct_type: &StructFields) -> Result<Self, Self::Error> {
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

        LogicalType::struct_type(child_types, child_names)
    }
}

impl TryFrom<PType> for LogicalType {
    type Error = VortexError;

    fn try_from(value: PType) -> Result<Self, Self::Error> {
        Ok(match value {
            I8 => LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_TINYINT),
            I16 => LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_SMALLINT),
            I32 => LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER),
            I64 => LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_BIGINT),
            U8 => LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_UTINYINT),
            U16 => LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_USMALLINT),
            U32 => LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_UINTEGER),
            U64 => LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_UBIGINT),
            F32 => LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_FLOAT),
            F64 => LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_DOUBLE),
            PType::F16 => return Err(vortex_err!("F16 type not supported in DuckDB")),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex::dtype::DType;
    use vortex::dtype::FieldName;
    use vortex::dtype::FieldNames;
    use vortex::dtype::Nullability;
    use vortex::dtype::PType;
    use vortex::dtype::StructFields;
    use vortex::dtype::extension::ExtDType;
    use vortex::dtype::extension::ExtId;
    use vortex::dtype::extension::ExtVTable;
    use vortex::error::VortexResult;
    use vortex::extension::EmptyMetadata;
    use vortex::extension::datetime::Date;
    use vortex::extension::datetime::Time;
    use vortex::extension::datetime::Timestamp;
    use vortex::scalar::ScalarValue;

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
        use vortex::extension::datetime::TimeUnit;

        let dtype = DType::Extension(Date::new(TimeUnit::Days, Nullability::NonNullable).erased());
        let logical_type = LogicalType::try_from(&dtype).unwrap();

        assert_eq!(
            logical_type.as_type_id(),
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_DATE
        );
    }

    #[test]
    fn test_time_extension_type() {
        use vortex::extension::datetime::TimeUnit;

        let dtype =
            DType::Extension(Time::new(TimeUnit::Microseconds, Nullability::NonNullable).erased());
        let logical_type = LogicalType::try_from(&dtype).unwrap();

        assert_eq!(
            logical_type.as_type_id(),
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_TIME
        );
    }

    #[test]
    fn test_timestamp_extension_types() {
        use vortex::extension::datetime::TimeUnit;

        let test_cases = [
            (
                TimeUnit::Nanoseconds,
                cpp::DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_NS,
            ),
            (
                TimeUnit::Microseconds,
                cpp::DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP,
            ),
            (
                TimeUnit::Milliseconds,
                cpp::DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_MS,
            ),
            (TimeUnit::Seconds, cpp::DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_S),
        ];

        for (time_unit, expected_type) in test_cases {
            let dtype =
                DType::Extension(Timestamp::new(time_unit, Nullability::NonNullable).erased());
            let logical_type = LogicalType::try_from(&dtype).unwrap();

            assert_eq!(logical_type.as_type_id(), expected_type);
        }
    }

    #[test]
    fn test_timestamp_with_timezone() {
        use vortex::extension::datetime::TimeUnit;

        let dtype = DType::Extension(
            Timestamp::new_with_tz(
                TimeUnit::Microseconds,
                Some("UTC".into()),
                Nullability::NonNullable,
            )
            .erased(),
        );

        assert_eq!(
            LogicalType::try_from(&dtype).unwrap().as_type_id(),
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_TZ
        );
    }

    #[test]
    fn test_temporal_extension_invalid_time_units() {
        use vortex::extension::datetime::TimeUnit;

        // Invalid DATE time unit
        let dtype =
            DType::Extension(Date::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased());
        assert!(LogicalType::try_from(&dtype).is_err());

        // Invalid TIME time unit
        let dtype =
            DType::Extension(Time::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased());
        assert!(LogicalType::try_from(&dtype).is_err());
    }

    #[test]
    fn test_unsupported_extension_type() {
        #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
        struct TestExt;
        impl ExtVTable for TestExt {
            type Metadata = EmptyMetadata;
            type NativeValue<'a> = &'a str;

            fn id(&self) -> ExtId {
                ExtId::new("unknown.extension")
            }

            fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
                Ok(vec![])
            }

            fn deserialize_metadata(&self, _data: &[u8]) -> VortexResult<Self::Metadata> {
                Ok(EmptyMetadata)
            }

            fn validate_dtype(_ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
                Ok(())
            }

            fn unpack_native<'a>(
                _ext_dtype: &'a ExtDType<Self>,
                _storage_value: &'a ScalarValue,
            ) -> VortexResult<Self::NativeValue<'a>> {
                Ok("")
            }
        }

        let ext_dtype = ExtDType::<TestExt>::try_new(
            EmptyMetadata,
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )
        .unwrap()
        .erased();
        let dtype = DType::Extension(ext_dtype);

        assert!(LogicalType::try_from(&dtype).is_err());
    }
}
