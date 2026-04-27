// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use chrono::Timelike;
use parquet_variant::Variant as PqVariant;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::extension::datetime::Date;
use vortex_array::extension::datetime::Time;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ParquetVariantArrayExt;
use crate::vtable::ParquetVariant;

impl OperationsVTable<ParquetVariant> for ParquetVariant {
    /// Resolves a single variant value according to the Parquet Variant shredding spec:
    ///
    /// | value    | typed_value | Meaning                                              |
    /// |----------|-------------|------------------------------------------------------|
    /// | NULL     | NULL        | Missing (only valid for shredded object fields)       |
    /// | non-NULL | NULL        | Un-shredded: decode from metadata + value bytes       |
    /// | NULL     | non-NULL    | Perfectly shredded: use typed_value directly           |
    /// | non-NULL | non-NULL    | Partially shredded object (typed_value takes priority) |
    fn scalar_at(
        array: ArrayView<'_, ParquetVariant>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        if array.validity()?.is_null(index)? {
            return Ok(Scalar::null(DType::Variant(Nullability::Nullable)));
        }

        let metadata = array
            .metadata_array()
            .execute_scalar(index, ctx)?
            .as_binary()
            .value()
            .cloned()
            .vortex_expect("non-null metadata row must have binary value");
        let inner = scalar_from_variant_storage(
            metadata.as_ref(),
            array.value_array(),
            array.typed_value_array(),
            index,
            ctx,
        )?;

        Scalar::try_new(
            array.dtype().clone(),
            Some(ScalarValue::Variant(Box::new(inner))),
        )
    }
}

fn scalar_from_variant_storage(
    metadata: &[u8],
    value: Option<&ArrayRef>,
    typed_value: Option<&ArrayRef>,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    if let Some(typed_value) = typed_value
        && typed_value.is_valid(index, ctx)?
    {
        return scalar_from_typed_value_array(metadata, value, typed_value, index, ctx);
    }

    if let Some(value) = value
        && value.is_valid(index, ctx)?
    {
        return scalar_from_unshredded_value(metadata, &value.execute_scalar(index, ctx)?);
    }

    Ok(Scalar::null(DType::Null))
}

fn scalar_from_typed_value_array(
    metadata: &[u8],
    value: Option<&ArrayRef>,
    typed_value: &ArrayRef,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    let value_scalar = match value {
        Some(value) if value.is_valid(index, ctx)? => Some(value.execute_scalar(index, ctx)?),
        _ => None,
    };
    scalar_from_typed_value_scalar(
        metadata,
        value_scalar,
        typed_value.execute_scalar(index, ctx)?,
    )
}

fn scalar_from_typed_value_scalar(
    metadata: &[u8],
    value: Option<Scalar>,
    typed_value: Scalar,
) -> VortexResult<Scalar> {
    match typed_value.dtype() {
        DType::List(..) => {
            let list = typed_value.as_list();
            let children = list
                .elements()
                .unwrap_or_default()
                .into_iter()
                .map(|element| {
                    let nested = scalar_from_shredded_field_scalar(metadata, element)?;
                    Ok(Scalar::variant(nested))
                })
                .collect::<VortexResult<Vec<_>>>()?;
            Ok(Scalar::list(
                DType::Variant(Nullability::NonNullable),
                children,
                Nullability::NonNullable,
            ))
        }
        DType::Struct(..) => scalar_from_shredded_object_scalar(metadata, value, typed_value),
        _ => Ok(typed_value),
    }
}

fn scalar_from_shredded_field_scalar(
    metadata: &[u8],
    field_scalar: Scalar,
) -> VortexResult<Scalar> {
    if field_scalar.is_null() {
        return Ok(Scalar::null(DType::Null));
    }

    let field = field_scalar.as_struct();
    scalar_from_field_scalars(metadata, field.field("value"), field.field("typed_value"))
}

fn scalar_from_field_scalars(
    metadata: &[u8],
    value: Option<Scalar>,
    typed_value: Option<Scalar>,
) -> VortexResult<Scalar> {
    if let Some(typed_value) = typed_value
        && !typed_value.is_null()
    {
        return scalar_from_typed_value_scalar(metadata, value, typed_value);
    }

    if let Some(value) = value
        && !value.is_null()
    {
        return scalar_from_unshredded_value(metadata, &value);
    }

    Ok(Scalar::null(DType::Null))
}

fn scalar_from_shredded_object_scalar(
    metadata: &[u8],
    value: Option<Scalar>,
    typed_value: Scalar,
) -> VortexResult<Scalar> {
    let typed_value = typed_value.as_struct();
    let mut names = Vec::new();
    let mut dtypes = Vec::new();
    let mut field_values = Vec::new();

    for name in typed_value.names().iter() {
        let nested = scalar_from_shredded_field_scalar(
            metadata,
            typed_value
                .field(name.as_ref())
                .vortex_expect("typed struct field must exist"),
        )?;
        names.push(FieldName::from(name.as_ref()));
        dtypes.push(DType::Variant(Nullability::NonNullable));
        field_values.push(Scalar::variant(nested).into_value());
    }

    if let Some(value) = value
        && !value.is_null()
    {
        let unshredded = scalar_from_unshredded_value(metadata, &value)?;
        if !unshredded.is_null() {
            let unshredded = unshredded.as_struct();
            for name in unshredded.names().iter() {
                if typed_value.field(name.as_ref()).is_some() {
                    continue;
                }
                let field = unshredded
                    .field(name.as_ref())
                    .vortex_expect("unshredded struct field must exist");
                names.push(FieldName::from(name.as_ref()));
                dtypes.push(DType::Variant(Nullability::NonNullable));
                field_values.push(field.into_value());
            }
        }
    }

    let fields = StructFields::new(FieldNames::from(names), dtypes);
    Scalar::try_new(
        DType::Struct(fields, Nullability::NonNullable),
        Some(ScalarValue::Tuple(field_values)),
    )
}

fn scalar_from_unshredded_value(metadata: &[u8], value: &Scalar) -> VortexResult<Scalar> {
    let value = value
        .as_binary()
        .value()
        .cloned()
        .vortex_expect("non-null value row must have binary value");
    parquet_variant_to_scalar(PqVariant::try_new(metadata, value.as_ref())?)
}

fn parquet_variant_to_scalar(variant: PqVariant<'_, '_>) -> VortexResult<Scalar> {
    let nn = Nullability::NonNullable;

    Ok(match variant {
        PqVariant::Null => Scalar::null(DType::Null),
        PqVariant::Int8(v) => Scalar::primitive(v, nn),
        PqVariant::Int16(v) => Scalar::primitive(v, nn),
        PqVariant::Int32(v) => Scalar::primitive(v, nn),
        PqVariant::Int64(v) => Scalar::primitive(v, nn),
        PqVariant::Float(v) => Scalar::primitive(v, nn),
        PqVariant::Double(v) => Scalar::primitive(v, nn),
        PqVariant::BooleanTrue => Scalar::bool(true, nn),
        PqVariant::BooleanFalse => Scalar::bool(false, nn),
        PqVariant::Decimal4(v) => Scalar::decimal(
            v.integer().into(),
            DecimalDType::new(9, v.scale() as i8),
            nn,
        ),
        PqVariant::Decimal8(v) => Scalar::decimal(
            v.integer().into(),
            DecimalDType::new(18, v.scale() as i8),
            nn,
        ),
        PqVariant::Decimal16(v) => Scalar::decimal(
            v.integer().into(),
            DecimalDType::new(38, v.scale() as i8),
            nn,
        ),
        PqVariant::Binary(v) => Scalar::binary(v.to_vec(), nn),
        PqVariant::String(v) => Scalar::utf8(v, nn),
        PqVariant::ShortString(v) => Scalar::utf8(v.as_str(), nn),
        PqVariant::Date(v) => {
            let dtype = DType::Extension(Date::new(TimeUnit::Days, nn).erased());
            Scalar::try_new(
                dtype,
                Some(ScalarValue::Primitive(PValue::I32(v.to_epoch_days()))),
            )?
        }
        PqVariant::TimestampMicros(v) => {
            let dtype = DType::Extension(
                Timestamp::new_with_tz(TimeUnit::Microseconds, Some(Arc::from("UTC")), nn).erased(),
            );
            Scalar::try_new(
                dtype,
                Some(ScalarValue::Primitive(PValue::I64(v.timestamp_micros()))),
            )?
        }
        PqVariant::TimestampNtzMicros(v) => {
            let dtype = DType::Extension(Timestamp::new(TimeUnit::Microseconds, nn).erased());
            Scalar::try_new(
                dtype,
                Some(ScalarValue::Primitive(PValue::I64(
                    v.and_utc().timestamp_micros(),
                ))),
            )?
        }
        PqVariant::TimestampNanos(v) => {
            let dtype = DType::Extension(
                Timestamp::new_with_tz(TimeUnit::Nanoseconds, Some(Arc::from("UTC")), nn).erased(),
            );
            let nanos = v
                .timestamp_nanos_opt()
                .ok_or_else(|| vortex_err!("Timestamp nanoseconds value out of i64 range"))?;
            Scalar::try_new(dtype, Some(ScalarValue::Primitive(PValue::I64(nanos))))?
        }
        PqVariant::TimestampNtzNanos(v) => {
            let dtype = DType::Extension(Timestamp::new(TimeUnit::Nanoseconds, nn).erased());
            let nanos = v
                .and_utc()
                .timestamp_nanos_opt()
                .ok_or_else(|| vortex_err!("Timestamp nanoseconds value out of i64 range"))?;
            Scalar::try_new(dtype, Some(ScalarValue::Primitive(PValue::I64(nanos))))?
        }
        PqVariant::Time(v) => {
            // Parquet Variant spec stores Time as microseconds since midnight.
            let micros =
                v.num_seconds_from_midnight() as i64 * 1_000_000 + v.nanosecond() as i64 / 1_000;
            let dtype = DType::Extension(Time::new(TimeUnit::Microseconds, nn).erased());
            Scalar::try_new(dtype, Some(ScalarValue::Primitive(PValue::I64(micros))))?
        }
        // TODO: Should this depend on the new UUID-extension type? leaving this for now.
        PqVariant::Uuid(v) => Scalar::utf8(v.to_string(), nn),
        PqVariant::List(values) => {
            let children = values
                .iter()
                .map(|v| parquet_variant_to_scalar(v).map(Scalar::variant))
                .collect::<VortexResult<Vec<_>>>()?;
            Scalar::list(DType::Variant(nn), children, nn)
        }
        PqVariant::Object(values) => {
            let mut names = Vec::new();
            let mut dtypes = Vec::new();
            let mut field_values = Vec::new();
            for (name, value) in values.iter() {
                names.push(FieldName::from(name));
                dtypes.push(DType::Variant(nn));
                field_values.push(Some(ScalarValue::Variant(Box::new(
                    parquet_variant_to_scalar(value)?,
                ))));
            }
            let fields = StructFields::new(FieldNames::from(names), dtypes);
            Scalar::try_new(
                DType::Struct(fields, nn),
                Some(ScalarValue::Tuple(field_values)),
            )?
        }
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::Array as _;
    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::Int32Array;
    use arrow_array::ListArray;
    use arrow_array::StructArray;
    use arrow_array::builder::BinaryViewBuilder;
    use arrow_buffer::NullBuffer;
    use arrow_buffer::OffsetBuffer;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use parquet_variant::Variant as PqVariant;
    use parquet_variant::VariantBuilder;
    use parquet_variant_compute::VariantArray as ArrowVariantArray;
    use parquet_variant_compute::VariantArrayBuilder;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Variant;
    use vortex_array::arrays::variant::VariantArrayExt;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar::ScalarValue;
    use vortex_error::VortexResult;

    use crate::ParquetVariant;
    use crate::ParquetVariantArrayExt;
    use crate::ParquetVariantData;
    use crate::operations::parquet_variant_to_scalar;

    fn binary_view_array(values: &[&[u8]]) -> ArrowArrayRef {
        let mut builder = BinaryViewBuilder::new();
        for value in values {
            builder.append_value(*value);
        }
        Arc::new(builder.finish())
    }

    fn assert_scalar_at_matches_arrow_try_value(
        arrow_variant: &ArrowVariantArray,
        rows: impl IntoIterator<Item = usize>,
    ) -> VortexResult<()> {
        let vortex_arr = ParquetVariantData::from_arrow_variant(arrow_variant)?;

        for index in rows {
            let expected_inner = parquet_variant_to_scalar(arrow_variant.try_value(index)?)?;
            let expected = Scalar::try_new(
                vortex_arr.dtype().clone(),
                Some(ScalarValue::Variant(Box::new(expected_inner))),
            )?;
            assert_eq!(
                vortex_arr.execute_scalar(index, &mut LEGACY_SESSION.create_execution_ctx())?,
                expected
            );
        }

        Ok(())
    }

    #[test]
    fn test_from_arrow_variant_nullable_validity() -> VortexResult<()> {
        let mut builder = VariantArrayBuilder::new(3);
        builder.append_variant(PqVariant::from(42i32));
        builder.append_variant(PqVariant::from("hello"));
        builder.append_variant(PqVariant::from(true));
        let inner = builder.build().into_inner();

        let null_struct = StructArray::try_new(
            inner.fields().clone(),
            inner.columns().to_vec(),
            Some(NullBuffer::from(vec![true, false, true])),
        )?;

        let arrow_variant = ArrowVariantArray::try_new(&null_struct)?;
        let vortex_arr = ParquetVariantData::from_arrow_variant(&arrow_variant)?;

        assert_eq!(vortex_arr.dtype(), &DType::Variant(Nullability::Nullable));

        let variant = vortex_arr.as_opt::<Variant>().unwrap();
        assert!(variant.dtype().is_nullable());

        assert!(
            vortex_arr
                .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );
        assert!(
            !vortex_arr
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );
        assert!(
            !vortex_arr
                .execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );

        Ok(())
    }

    #[test]
    fn test_from_arrow_variant_all_nulls() -> VortexResult<()> {
        let mut builder = VariantArrayBuilder::new(2);
        builder.append_variant(PqVariant::from(1i32));
        builder.append_variant(PqVariant::from(2i32));
        let inner = builder.build().into_inner();

        let null_struct = StructArray::try_new(
            inner.fields().clone(),
            inner.columns().to_vec(),
            Some(NullBuffer::from(vec![false, false])),
        )?;

        let arrow_variant = ArrowVariantArray::try_new(&null_struct)?;
        let vortex_arr = ParquetVariantData::from_arrow_variant(&arrow_variant)?;

        assert_eq!(vortex_arr.dtype(), &DType::Variant(Nullability::Nullable));
        assert!(
            vortex_arr
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );
        assert!(
            vortex_arr
                .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );

        let variant_view = vortex_arr.as_opt::<Variant>().unwrap();
        let child = variant_view.child();
        let inner_pv = child.as_opt::<ParquetVariant>().unwrap();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let roundtripped = inner_pv.to_arrow(&mut ctx)?;
        assert_eq!(roundtripped.inner().null_count(), 2);

        Ok(())
    }

    #[test]
    fn test_from_arrow_variant_non_nullable() -> VortexResult<()> {
        let mut builder = VariantArrayBuilder::new(2);
        builder.append_variant(PqVariant::from(1i32));
        builder.append_variant(PqVariant::from(2i32));
        let arrow_variant = builder.build();

        let vortex_arr = ParquetVariantData::from_arrow_variant(&arrow_variant)?;

        assert_eq!(
            vortex_arr.dtype(),
            &DType::Variant(Nullability::NonNullable)
        );
        assert!(
            !vortex_arr
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );
        assert!(
            !vortex_arr
                .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );
        assert_scalar_at_matches_arrow_try_value(&arrow_variant, [0, 1])?;

        Ok(())
    }

    #[test]
    fn test_scalar_at_matches_arrow_try_value_unshredded_mixed_values() -> VortexResult<()> {
        let mut builder = VariantArrayBuilder::new(3);
        builder.append_variant(PqVariant::from(42i32));
        builder.append_variant(PqVariant::from("hello"));
        builder.append_variant(PqVariant::from(true));
        let arrow_variant = builder.build();

        assert_scalar_at_matches_arrow_try_value(&arrow_variant, [0, 1, 2])
    }

    #[test]
    fn test_scalar_at_matches_arrow_try_value_shredded_primitive() -> VortexResult<()> {
        let struct_array = StructArray::try_new(
            vec![
                Arc::new(Field::new("metadata", DataType::BinaryView, false)),
                Arc::new(Field::new("typed_value", DataType::Int32, false)),
            ]
            .into(),
            vec![
                binary_view_array(&[b"\x01\x00", b"\x01\x00", b"\x01\x00"]),
                Arc::new(Int32Array::from(vec![10, 20, 30])),
            ],
            None,
        )?;

        let arrow_variant = ArrowVariantArray::try_new(&struct_array)?;
        assert_scalar_at_matches_arrow_try_value(&arrow_variant, [0, 1, 2])
    }

    #[test]
    fn test_scalar_at_matches_arrow_try_value_imperfectly_shredded_primitive() -> VortexResult<()> {
        let (metadata0, value0) = VariantBuilder::new().with_value("fallback-0").finish();
        let (metadata1, _value1) = VariantBuilder::new().with_value(20i32).finish();
        let (metadata2, value2) = VariantBuilder::new().with_value("fallback-2").finish();

        let metadata = binary_view_array(&[
            metadata0.as_slice(),
            metadata1.as_slice(),
            metadata2.as_slice(),
        ]);
        let mut value_builder = BinaryViewBuilder::new();
        value_builder.append_value(value0.as_slice());
        value_builder.append_null();
        value_builder.append_value(value2.as_slice());
        let value = Arc::new(value_builder.finish());

        let typed_value = Arc::new(Int32Array::from(vec![None, Some(20), None]));
        let struct_array = StructArray::try_new(
            vec![
                Arc::new(Field::new("metadata", DataType::BinaryView, false)),
                Arc::new(Field::new("value", DataType::BinaryView, true)),
                Arc::new(Field::new("typed_value", DataType::Int32, true)),
            ]
            .into(),
            vec![metadata, value, typed_value],
            None,
        )?;

        let arrow_variant = ArrowVariantArray::try_new(&struct_array)?;
        assert_scalar_at_matches_arrow_try_value(&arrow_variant, [0, 1, 2])
    }

    #[test]
    fn test_scalar_at_recursive_shredded_list() -> VortexResult<()> {
        // Spec basis: for arrays, "value must be null" when the value is an array, and array
        // elements cannot be missing.
        // Source: https://github.com/apache/parquet-format/blob/master/VariantShredding.md
        let element_struct = StructArray::try_new(
            vec![Arc::new(Field::new("typed_value", DataType::Int32, false))].into(),
            vec![Arc::new(Int32Array::from(vec![10, 20, 30]))],
            None,
        )?;

        let typed_value: ArrowArrayRef = Arc::new(ListArray::try_new(
            Arc::new(Field::new(
                "element",
                element_struct.data_type().clone(),
                false,
            )),
            OffsetBuffer::from_lengths([2, 1]),
            Arc::new(element_struct),
            None,
        )?);

        let struct_array = StructArray::try_new(
            vec![
                Arc::new(Field::new("metadata", DataType::BinaryView, false)),
                Arc::new(Field::new(
                    "typed_value",
                    typed_value.data_type().clone(),
                    false,
                )),
            ]
            .into(),
            vec![binary_view_array(&[b"\x01\x00", b"\x01\x00"]), typed_value],
            None,
        )?;

        let arrow_variant = ArrowVariantArray::try_new(&struct_array)?;
        let vortex_arr = ParquetVariantData::from_arrow_variant(&arrow_variant)?;

        let row0 = vortex_arr.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?;
        let row0 = row0.as_variant().value().unwrap().as_list();
        assert_eq!(row0.len(), 2);
        assert_eq!(
            row0.element(0)
                .unwrap()
                .as_variant()
                .value()
                .unwrap()
                .as_primitive()
                .typed_value::<i32>(),
            Some(10)
        );
        assert_eq!(
            row0.element(1)
                .unwrap()
                .as_variant()
                .value()
                .unwrap()
                .as_primitive()
                .typed_value::<i32>(),
            Some(20)
        );

        let row1 = vortex_arr.execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())?;
        let row1 = row1.as_variant().value().unwrap().as_list();
        assert_eq!(row1.len(), 1);
        assert_eq!(
            row1.element(0)
                .unwrap()
                .as_variant()
                .value()
                .unwrap()
                .as_primitive()
                .typed_value::<i32>(),
            Some(30)
        );

        Ok(())
    }

    #[test]
    fn test_scalar_at_partially_shredded_object_merges_fields() -> VortexResult<()> {
        // Spec basis: non-null `value` + non-null `typed_value` means a "partially shredded
        // object", so reconstruction must merge the shredded object with the fallback object.
        // Source: https://github.com/apache/parquet-format/blob/master/VariantShredding.md
        let mut builder = VariantBuilder::new();
        builder
            .new_object()
            .with_field("a", 1i32)
            .with_field("b", "leftover")
            .finish();
        let (metadata, value) = builder.finish();

        let shredded_a = StructArray::try_new(
            vec![Arc::new(Field::new("typed_value", DataType::Int32, false))].into(),
            vec![Arc::new(Int32Array::from(vec![7]))],
            None,
        )?;
        let typed_value: ArrowArrayRef = Arc::new(StructArray::try_new(
            vec![Arc::new(Field::new(
                "a",
                shredded_a.data_type().clone(),
                false,
            ))]
            .into(),
            vec![Arc::new(shredded_a)],
            None,
        )?);

        let struct_array = StructArray::try_new(
            vec![
                Arc::new(Field::new("metadata", DataType::BinaryView, false)),
                Arc::new(Field::new("value", DataType::BinaryView, true)),
                Arc::new(Field::new(
                    "typed_value",
                    typed_value.data_type().clone(),
                    false,
                )),
            ]
            .into(),
            vec![
                binary_view_array(&[metadata.as_slice()]),
                binary_view_array(&[value.as_slice()]),
                typed_value,
            ],
            None,
        )?;

        let arrow_variant = ArrowVariantArray::try_new(&struct_array)?;
        let vortex_arr = ParquetVariantData::from_arrow_variant(&arrow_variant)?;
        let object = vortex_arr.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?;
        let object = object.as_variant().value().unwrap().as_struct();

        assert_eq!(
            object
                .field("a")
                .unwrap()
                .as_variant()
                .value()
                .unwrap()
                .as_primitive()
                .typed_value::<i32>(),
            Some(7)
        );
        assert_eq!(
            object
                .field("b")
                .unwrap()
                .as_variant()
                .value()
                .unwrap()
                .as_utf8()
                .value()
                .map(|value| value.as_str()),
            Some("leftover")
        );

        Ok(())
    }
}
