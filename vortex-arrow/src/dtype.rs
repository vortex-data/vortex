// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Convert between Vortex [`vortex_array::dtype::DType`] and Apache Arrow [`arrow_schema::DataType`].
//!
//! Apache Arrow's type system includes physical information, which could lead to ambiguities as
//! Vortex treats encodings as separate from logical types.
//!
//! The conversions in this module are "naive": every logical type is encoded in its simplest
//! corresponding Arrow type, and Arrow extension types (other than the builtin temporal types and
//! Parquet Variant) are not understood. The authoritative, plugin-aware conversion entry point is
//! [`ArrowSession`](crate::ArrowSession), through its `to_arrow_schema` / `to_arrow_datatype` /
//! `from_arrow_schema` / `from_arrow_field` / `from_arrow_datatype` methods.
//!
//! For this reason, it's recommended to do as much computation as possible within Vortex, and then
//! materialize an Arrow ArrayRef at the very end of the processing chain.

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::FieldRef;
use arrow_schema::Fields;
use arrow_schema::TimeUnit as ArrowTimeUnit;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::extension::datetime::AnyTemporal;
use vortex_array::extension::datetime::Date;
use vortex_array::extension::datetime::TemporalMetadata;
use vortex_array::extension::datetime::Time;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

/// Convert an Arrow [`ArrowTimeUnit`] to a Vortex [`TimeUnit`].
pub(crate) fn from_arrow_time_unit(value: ArrowTimeUnit) -> TimeUnit {
    match value {
        ArrowTimeUnit::Second => TimeUnit::Seconds,
        ArrowTimeUnit::Millisecond => TimeUnit::Milliseconds,
        ArrowTimeUnit::Microsecond => TimeUnit::Microseconds,
        ArrowTimeUnit::Nanosecond => TimeUnit::Nanoseconds,
    }
}

/// Convert a Vortex [`TimeUnit`] to an Arrow [`ArrowTimeUnit`].
///
/// # Errors
///
/// Returns an error for units with no Arrow equivalent (e.g. [`TimeUnit::Days`]).
pub(crate) fn to_arrow_time_unit(value: TimeUnit) -> VortexResult<ArrowTimeUnit> {
    Ok(match value {
        TimeUnit::Seconds => ArrowTimeUnit::Second,
        TimeUnit::Milliseconds => ArrowTimeUnit::Millisecond,
        TimeUnit::Microseconds => ArrowTimeUnit::Microsecond,
        TimeUnit::Nanoseconds => ArrowTimeUnit::Nanosecond,
        _ => vortex_bail!("Cannot convert {value} to Arrow TimeUnit"),
    })
}

/// Naive conversion of an Arrow [`DataType`] to the Vortex [`PType`] it stores.
pub(crate) fn ptype_from_arrow(value: &DataType) -> VortexResult<PType> {
    match value {
        DataType::Int8 => Ok(PType::I8),
        DataType::Int16 => Ok(PType::I16),
        DataType::Int32 => Ok(PType::I32),
        DataType::Int64 => Ok(PType::I64),
        DataType::UInt8 => Ok(PType::U8),
        DataType::UInt16 => Ok(PType::U16),
        DataType::UInt32 => Ok(PType::U32),
        DataType::UInt64 => Ok(PType::U64),
        DataType::Float16 => Ok(PType::F16),
        DataType::Float32 => Ok(PType::F32),
        DataType::Float64 => Ok(PType::F64),
        _ => Err(vortex_err!(
            "Arrow datatype {:?} cannot be converted to ptype",
            value
        )),
    }
}

/// Naive conversion of an Arrow [`DataType`] to a Vortex [`DType`], with no awareness of Arrow
/// extension plugins.
pub(crate) fn from_arrow_data_type(
    data_type: &DataType,
    nullability: Nullability,
) -> VortexResult<DType> {
    if data_type.is_integer() || data_type.is_floating() {
        return Ok(DType::Primitive(ptype_from_arrow(data_type)?, nullability));
    }

    Ok(match data_type {
        DataType::Null => DType::Null,
        DataType::Decimal32(precision, scale)
        | DataType::Decimal64(precision, scale)
        | DataType::Decimal128(precision, scale)
        | DataType::Decimal256(precision, scale) => {
            DType::Decimal(DecimalDType::try_new(*precision, *scale)?, nullability)
        }
        DataType::Boolean => DType::Bool(nullability),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => DType::Utf8(nullability),
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView => {
            DType::Binary(nullability)
        }
        DataType::Date32 => DType::Extension(Date::new(TimeUnit::Days, nullability).erased()),
        DataType::Date64 => {
            DType::Extension(Date::new(TimeUnit::Milliseconds, nullability).erased())
        }
        DataType::Time32(unit) => {
            DType::Extension(Time::new(from_arrow_time_unit(*unit), nullability).erased())
        }
        DataType::Time64(unit) => {
            DType::Extension(Time::new(from_arrow_time_unit(*unit), nullability).erased())
        }
        DataType::Timestamp(unit, tz) => DType::Extension(
            Timestamp::try_new_with_tz(from_arrow_time_unit(*unit), tz.clone(), nullability)?
                .erased(),
        ),
        DataType::List(e)
        | DataType::LargeList(e)
        | DataType::ListView(e)
        | DataType::LargeListView(e) => {
            DType::List(Arc::new(from_arrow_field_naive(e.as_ref())?), nullability)
        }
        DataType::FixedSizeList(e, size) => DType::FixedSizeList(
            Arc::new(from_arrow_field_naive(e.as_ref())?),
            *size as u32,
            nullability,
        ),
        DataType::Struct(f) => DType::Struct(from_arrow_fields_naive(f)?, nullability),
        DataType::Map(entries, keys_sorted) => {
            vortex_ensure!(
                !entries.is_nullable(),
                "Arrow map entries field must be non-nullable"
            );
            let DataType::Struct(fields) = entries.data_type() else {
                vortex_bail!(
                    "Arrow map entries field must have Struct type, got {:?}",
                    entries.data_type()
                );
            };
            vortex_ensure_eq!(
                fields.len(),
                2,
                "Arrow map entries struct must contain exactly two fields"
            );
            let key = &fields[0];
            let value = &fields[1];
            vortex_ensure!(
                !key.is_nullable(),
                "Arrow map key field must be non-nullable"
            );
            DType::map(
                from_arrow_field_naive(key.as_ref())?,
                from_arrow_field_naive(value.as_ref())?,
                *keys_sorted,
                nullability,
            )?
        }
        DataType::Dictionary(_, value_type) => {
            from_arrow_data_type(value_type.as_ref(), nullability)?
        }
        DataType::RunEndEncoded(_, value_type) => {
            from_arrow_data_type(value_type.data_type(), nullability)?
        }
        _ => vortex_bail!("Arrow data type not supported: {data_type:?}"),
    })
}

/// Naive conversion of an Arrow [`Field`] to a Vortex [`DType`]. Only the builtin
/// `arrow.parquet.variant` extension metadata is recognized.
pub(crate) fn from_arrow_field_naive(field: &Field) -> VortexResult<DType> {
    if field
        .metadata()
        .get("ARROW:extension:name")
        .map(|s| s.as_str())
        == Some("arrow.parquet.variant")
    {
        return Ok(DType::Variant(field.is_nullable().into()));
    }
    from_arrow_data_type(field.data_type(), field.is_nullable().into())
}

/// Naive conversion of Arrow [`Fields`] to Vortex [`StructFields`].
pub(crate) fn from_arrow_fields_naive(fields: &Fields) -> VortexResult<StructFields> {
    fields
        .into_iter()
        .map(|f| {
            Ok((
                FieldName::from(f.name().as_str()),
                from_arrow_field_naive(f.as_ref())?,
            ))
        })
        .collect::<VortexResult<StructFields>>()
}

/// Naive conversion from a Vortex `DType` to the nearest Arrow physical data type.
pub(crate) fn to_data_type_naive(dtype: &DType) -> VortexResult<DataType> {
    Ok(match dtype {
        DType::Null => DataType::Null,
        DType::Bool(_) => DataType::Boolean,
        DType::Primitive(ptype, _) => match ptype {
            PType::U8 => DataType::UInt8,
            PType::U16 => DataType::UInt16,
            PType::U32 => DataType::UInt32,
            PType::U64 => DataType::UInt64,
            PType::I8 => DataType::Int8,
            PType::I16 => DataType::Int16,
            PType::I32 => DataType::Int32,
            PType::I64 => DataType::Int64,
            PType::F16 => DataType::Float16,
            PType::F32 => DataType::Float32,
            PType::F64 => DataType::Float64,
        },
        DType::Decimal(dt, _) => {
            let precision = dt.precision();
            let scale = dt.scale();

            match precision {
                // This code is commented out until DataFusion improves its support for smaller decimals.
                // // DECIMAL32_MAX_PRECISION
                // 0..=9 => DataType::Decimal32(precision, scale),
                // // DECIMAL64_MAX_PRECISION
                // 10..=18 => DataType::Decimal64(precision, scale),
                // DECIMAL128_MAX_PRECISION
                0..=38 => DataType::Decimal128(precision, scale),
                // DECIMAL256_MAX_PRECISION
                39.. => DataType::Decimal256(precision, scale),
            }
        }
        DType::Utf8(_) => DataType::Utf8View,
        DType::Binary(_) => DataType::BinaryView,
        // There are four kinds of lists: List (32-bit offsets), Large List (64-bit), List View
        // (32-bit), Large List View (64-bit). We cannot both guarantee zero-copy and commit to an
        // Arrow dtype because we do not how large our offsets are.
        DType::List(elem_dtype, _) => DataType::List(FieldRef::new(Field::new_list_field(
            to_data_type_naive(elem_dtype)?,
            elem_dtype.nullability().into(),
        ))),
        DType::FixedSizeList(elem_dtype, size, _) => DataType::FixedSizeList(
            FieldRef::new(Field::new_list_field(
                to_data_type_naive(elem_dtype)?,
                elem_dtype.nullability().into(),
            )),
            *size as i32,
        ),
        DType::Map(map_dtype, _) => {
            let key = Field::new("key", to_data_type_naive(&map_dtype.key_dtype())?, false);
            let value_dtype = map_dtype.value_dtype();
            let value = Field::new(
                "value",
                to_data_type_naive(&value_dtype)?,
                value_dtype.is_nullable(),
            );
            let entries = Field::new_struct("entries", Fields::from(vec![key, value]), false);
            DataType::Map(FieldRef::new(entries), map_dtype.keys_sorted())
        }
        DType::Struct(struct_dtype, _) => {
            let mut fields = Vec::with_capacity(struct_dtype.names().len());
            for (field_name, field_dt) in struct_dtype.names().iter().zip(struct_dtype.fields()) {
                fields.push(FieldRef::from(Field::new(
                    field_name.as_ref(),
                    to_data_type_naive(&field_dt)?,
                    field_dt.is_nullable(),
                )));
            }

            DataType::Struct(Fields::from(fields))
        }
        DType::Union(..) => vortex_bail!("TODO(connor)[Union]: unimplemented"),
        DType::Variant(_) => vortex_bail!(
            "DType::Variant requires Arrow Field metadata; use to_arrow_schema or a Field helper"
        ),
        DType::Extension(ext_dtype) => {
            // NOTE: Temporal are the only builtin and default-loaded extension types, and they map
            // directly onto non-extension Arrow physical encodings. For this reason, we
            // choose to special-case them as part of this function rather than implementing them
            // as an import/export VTable.
            if let Some(temporal) = ext_dtype.metadata_opt::<AnyTemporal>() {
                return Ok(match temporal {
                    TemporalMetadata::Timestamp(unit, tz) => {
                        DataType::Timestamp(to_arrow_time_unit(*unit)?, tz.clone())
                    }
                    TemporalMetadata::Date(unit) => match unit {
                        TimeUnit::Days => DataType::Date32,
                        TimeUnit::Milliseconds => DataType::Date64,
                        TimeUnit::Nanoseconds | TimeUnit::Microseconds | TimeUnit::Seconds => {
                            vortex_panic!("Invalid TimeUnit {} for {}", unit, ext_dtype.id())
                        }
                    },
                    TemporalMetadata::Time(unit) => match unit {
                        TimeUnit::Seconds => DataType::Time32(ArrowTimeUnit::Second),
                        TimeUnit::Milliseconds => DataType::Time32(ArrowTimeUnit::Millisecond),
                        TimeUnit::Microseconds => DataType::Time64(ArrowTimeUnit::Microsecond),
                        TimeUnit::Nanoseconds => DataType::Time64(ArrowTimeUnit::Nanosecond),
                        TimeUnit::Days => {
                            vortex_panic!("Invalid TimeUnit {} for {}", unit, ext_dtype.id())
                        }
                    },
                });
            };

            vortex_bail!("Unsupported extension type \"{}\"", ext_dtype.id())
        }
    })
}

#[cfg(test)]
mod test {
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::FieldRef;
    use arrow_schema::Fields;
    use arrow_schema::Schema;
    use rstest::fixture;
    use rstest::rstest;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldName;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;

    use super::*;
    use crate::ArrowSession;

    #[test]
    fn test_dtype_conversion_success() {
        assert_eq!(to_data_type_naive(&DType::Null).unwrap(), DataType::Null);

        assert_eq!(
            to_data_type_naive(&DType::Bool(Nullability::NonNullable)).unwrap(),
            DataType::Boolean
        );

        assert_eq!(
            to_data_type_naive(&DType::Primitive(PType::U64, Nullability::NonNullable)).unwrap(),
            DataType::UInt64
        );

        assert_eq!(
            to_data_type_naive(&DType::Utf8(Nullability::NonNullable)).unwrap(),
            DataType::Utf8View
        );

        assert_eq!(
            to_data_type_naive(&DType::Binary(Nullability::NonNullable)).unwrap(),
            DataType::BinaryView
        );

        assert_eq!(
            to_data_type_naive(&DType::struct_(
                [
                    ("field_a", DType::Bool(false.into())),
                    ("field_b", DType::Utf8(true.into()))
                ],
                Nullability::NonNullable,
            ))
            .unwrap(),
            DataType::Struct(Fields::from(vec![
                FieldRef::from(Field::new("field_a", DataType::Boolean, false)),
                FieldRef::from(Field::new("field_b", DataType::Utf8View, true)),
            ]))
        );
    }

    #[test]
    fn test_reject_timestamp_invalid_timezone() {
        let dtype = DataType::Timestamp(ArrowTimeUnit::Second, Some("Not/A/Timezone".into()));
        assert!(from_arrow_data_type(&dtype, Nullability::Nullable).is_err());
    }

    #[rstest]
    #[case(1, DataType::Decimal128(1, 0))]
    #[case(38, DataType::Decimal128(38, 0))]
    #[case(39, DataType::Decimal256(39, 0))]
    #[case(76, DataType::Decimal256(76, 0))]
    fn test_decimal_dtype_to_arrow(#[case] precision: u8, #[case] expected: DataType) {
        use vortex_array::dtype::DecimalDType;

        let dtype = DType::Decimal(DecimalDType::new(precision, 0), Nullability::NonNullable);
        assert_eq!(to_data_type_naive(&dtype).unwrap(), expected);
    }

    #[rstest]
    #[case::decimal32(DataType::Decimal32(1, 2))]
    #[case::decimal64(DataType::Decimal64(1, 2))]
    #[case::decimal128(DataType::Decimal128(1, 2))]
    #[case::decimal256(DataType::Decimal256(1, 2))]
    #[case::zero_precision(DataType::Decimal128(0, 0))]
    #[case::excessive_precision(DataType::Decimal256(77, 0))]
    fn test_malformed_decimal_dtype_from_arrow(#[case] data_type: DataType) {
        assert!(from_arrow_data_type(&data_type, Nullability::Nullable).is_err());
    }

    #[test]
    fn test_variant_dtype_to_arrow_dtype_errors() {
        let err = to_data_type_naive(&DType::Variant(Nullability::NonNullable))
            .unwrap_err()
            .to_string();
        assert!(err.contains("Variant"));
    }

    #[test]
    fn infer_nullable_list_element() {
        let list_non_nullable = DType::List(
            Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
            Nullability::Nullable,
        );

        let arrow_list_non_nullable = to_data_type_naive(&list_non_nullable).unwrap();

        let list_nullable = DType::List(
            Arc::new(DType::Primitive(PType::I64, Nullability::Nullable)),
            Nullability::Nullable,
        );
        let arrow_list_nullable = to_data_type_naive(&list_nullable).unwrap();

        assert_ne!(arrow_list_non_nullable, arrow_list_nullable);
        assert_eq!(
            arrow_list_nullable,
            DataType::List(Arc::new(Field::new_list_field(DataType::Int64, true))),
        );
        assert_eq!(
            arrow_list_non_nullable,
            DataType::List(Arc::new(Field::new_list_field(DataType::Int64, false))),
        );
    }

    #[fixture]
    fn the_struct() -> StructFields {
        StructFields::new(
            FieldNames::from([
                FieldName::from("field_a"),
                FieldName::from("field_b"),
                FieldName::from("field_c"),
            ]),
            vec![
                DType::Bool(Nullability::NonNullable),
                DType::Utf8(Nullability::NonNullable),
                DType::Primitive(PType::I32, Nullability::Nullable),
            ],
        )
    }

    #[rstest]
    fn test_schema_conversion(the_struct: StructFields) {
        let schema_nonnull = DType::Struct(the_struct, Nullability::NonNullable);

        assert_eq!(
            ArrowSession::default()
                .to_arrow_schema(&schema_nonnull)
                .unwrap(),
            Schema::new(Fields::from(vec![
                Field::new("field_a", DataType::Boolean, false),
                Field::new("field_b", DataType::Utf8View, false),
                Field::new("field_c", DataType::Int32, true),
            ]))
        );
    }

    #[test]
    fn test_unicode_field_names_roundtrip() {
        // Regression test for https://github.com/vortex-data/vortex/issues/5979.

        // Unicode characters in field names should survive an Arrow roundtrip without
        // double-escaping.
        let unicode_field_name = "\u{5}=A";
        let original_dtype = DType::struct_(
            [(
                unicode_field_name,
                DType::Primitive(PType::I8, Nullability::Nullable),
            )],
            Nullability::NonNullable,
        );

        let arrow_dtype = to_data_type_naive(&original_dtype).unwrap();
        assert_eq!(
            arrow_dtype,
            DataType::Struct(Fields::from(vec![Field::new(
                "\u{5}=A",
                DataType::Int8,
                true
            )]))
        );
        let roundtripped_dtype =
            from_arrow_data_type(&arrow_dtype, Nullability::NonNullable).unwrap();

        assert_eq!(original_dtype, roundtripped_dtype);
    }

    #[test]
    fn test_unicode_field_names_nested_roundtrip() {
        // Regression test for https://github.com/vortex-data/vortex/issues/5979.

        // Nested structs with unicode field names should also survive an Arrow roundtrip.
        let inner_struct = DType::struct_(
            [(
                "\u{6}=inner",
                DType::Primitive(PType::I32, Nullability::Nullable),
            )],
            Nullability::Nullable,
        );
        let original_dtype =
            DType::struct_([("\u{7}=outer", inner_struct)], Nullability::NonNullable);

        let arrow_dtype = to_data_type_naive(&original_dtype).unwrap();
        assert_eq!(
            arrow_dtype,
            DataType::Struct(Fields::from(vec![Field::new(
                "\u{7}=outer",
                DataType::Struct(Fields::from(vec![Field::new(
                    "\u{6}=inner",
                    DataType::Int32,
                    true
                )])),
                true
            )]))
        );
        let roundtripped_dtype =
            from_arrow_data_type(&arrow_dtype, Nullability::NonNullable).unwrap();

        assert_eq!(original_dtype, roundtripped_dtype);
    }

    #[test]
    fn map_dtype_roundtrip_uses_conventional_export_names_and_positional_import() -> VortexResult<()>
    {
        let dtype = DType::map(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Utf8(Nullability::Nullable),
            true,
            Nullability::Nullable,
        )?;

        let arrow = to_data_type_naive(&dtype)?;
        let DataType::Map(entries, keys_sorted) = &arrow else {
            panic!("expected Map, got {arrow:?}");
        };
        assert!(*keys_sorted);
        assert_eq!(entries.name(), "entries");
        assert!(!entries.is_nullable());
        let DataType::Struct(fields) = entries.data_type() else {
            panic!("expected map entries to be a struct");
        };
        assert_eq!(fields[0].name(), "key");
        assert!(!fields[0].is_nullable());
        assert_eq!(fields[1].name(), "value");
        assert!(fields[1].is_nullable());
        assert_eq!(from_arrow_data_type(&arrow, Nullability::Nullable)?, dtype);

        let positional = DataType::Map(
            Arc::new(Field::new_struct(
                "anything",
                Fields::from(vec![
                    Field::new("first", DataType::Int32, false),
                    Field::new("second", DataType::Utf8, true),
                ]),
                false,
            )),
            false,
        );
        assert_eq!(
            from_arrow_data_type(&positional, Nullability::NonNullable)?,
            DType::map(
                DType::Primitive(PType::I32, Nullability::NonNullable),
                DType::Utf8(Nullability::Nullable),
                false,
                Nullability::NonNullable,
            )?
        );

        Ok(())
    }

    #[test]
    fn map_dtype_import_rejects_invalid_arrow_shape() {
        let invalid_entries = [
            Field::new_struct(
                "entries",
                Fields::from(vec![
                    Field::new("key", DataType::Int32, false),
                    Field::new("value", DataType::Utf8, true),
                ]),
                true,
            ),
            Field::new("entries", DataType::Int32, false),
            Field::new_struct(
                "entries",
                Fields::from(vec![Field::new("key", DataType::Int32, false)]),
                false,
            ),
            Field::new_struct(
                "entries",
                Fields::from(vec![
                    Field::new("key", DataType::Int32, true),
                    Field::new("value", DataType::Utf8, true),
                ]),
                false,
            ),
        ];

        for entries in invalid_entries {
            let data_type = DataType::Map(Arc::new(entries), false);
            assert!(from_arrow_data_type(&data_type, Nullability::NonNullable).is_err());
        }
    }

    // Regression test for https://github.com/vortex-data/vortex/issues/8346: unsupported Arrow
    // types must return an error instead of panicking with `unimplemented!`.
    #[rstest]
    #[case::duration(DataType::Duration(ArrowTimeUnit::Microsecond))]
    #[case::interval(DataType::Interval(arrow_schema::IntervalUnit::DayTime))]
    #[case::fixed_size_binary(DataType::FixedSizeBinary(3))]
    fn test_try_from_arrow_unsupported_type_errors(#[case] data_type: DataType) {
        let err = from_arrow_data_type(&data_type, Nullability::NonNullable)
            .expect_err("unsupported Arrow type should not convert")
            .to_string();
        assert!(err.contains("not supported"), "unexpected error: {err}");

        // The same unsupported type nested in a field or schema must also error cleanly.
        let field = Field::new("c0", data_type, true);
        assert!(from_arrow_field_naive(&field).is_err());
        let schema = Schema::new(vec![field]);
        assert!(ArrowSession::default().from_arrow_schema(&schema).is_err());
    }

    #[test]
    fn test_try_from_arrow_supported_type_succeeds() -> VortexResult<()> {
        let dtype = from_arrow_data_type(&DataType::Int32, Nullability::Nullable)?;
        assert_eq!(dtype, DType::Primitive(PType::I32, Nullability::Nullable));
        Ok(())
    }
}
