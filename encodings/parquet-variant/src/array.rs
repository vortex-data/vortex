// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use arrow_array::Array as ArrowArray;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::Field;
use parquet_variant_compute::VariantArray as ArrowVariantArray;
use vortex_array::Array;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::TypedArrayRef;
use vortex_array::arrays::VariantArray;
use vortex_array::arrow::ArrowArrayExecutor;
use vortex_array::arrow::FromArrowArray;
use vortex_array::arrow::to_arrow_null_buffer;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use crate::ParquetVariant;

/// The validity bitmap indicating which elements are non-null.
pub(crate) const VALIDITY_SLOT: usize = 0;
/// The metadata array for the Parquet variant values.
pub(crate) const METADATA_SLOT: usize = 1;
/// The value array containing the Parquet variant data.
pub(crate) const VALUE_SLOT: usize = 2;
/// The typed value array for strongly-typed Parquet variant data.
pub(crate) const TYPED_VALUE_SLOT: usize = 3;
pub(crate) const NUM_SLOTS: usize = 4;
pub(crate) const SLOT_NAMES: [&str; NUM_SLOTS] = ["validity", "metadata", "value", "typed_value"];

/// Array storage for Arrow's canonical `arrow.parquet.variant` extension type.
///
/// `ParquetVariantArray` preserves semi-structured data stored as Parquet Variant values in a
/// lossless form and supports both unshredded and shredded layouts. Its storage matches the
/// canonical extension type contract:
/// - `metadata` is always present and non-nullable.
/// - `value` stores unshredded variant bytes when present
/// - `typed_value` stores shredded data when present
///
/// At least one of `value` or `typed_value` must be present. `typed_value` may be a primitive,
/// list, or struct, with nested shredded children following the same recursive rules as the
/// Arrow canonical extension type docs.
///
/// # Nullability
///
/// There are three independent levels of nullability in this encoding:
///
/// 1. **Outer validity** (`validity`): controls which *rows* of the variant column are null.
///    This drives the `DType::Variant(Nullable | NonNullable)` of the enclosing `VariantArray`.
///
/// 2. **Value child nullability**: the `value` child is `Binary` and may be nullable or
///    non-nullable. In partially-shredded layouts some rows have their data in `typed_value`
///    instead, so the corresponding `value` slot is null — making the child nullable.
///
/// 3. **Typed-value child nullability**: the `typed_value` child carries its own `DType`
///    (which includes nullability).
#[derive(Clone, Debug)]
pub struct ParquetVariantData;

impl Display for ParquetVariantData {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl ParquetVariant {
    pub fn try_new(
        validity: Validity,
        metadata: ArrayRef,
        value: Option<ArrayRef>,
        typed_value: Option<ArrayRef>,
    ) -> VortexResult<Array<Self>> {
        let len = metadata.len();
        let dtype = DType::Variant(validity.nullability());
        validate_parts(
            &validity,
            &metadata,
            value.as_ref(),
            typed_value.as_ref(),
            &dtype,
            len,
        )?;
        let slots = vec![
            validity_to_child(&validity, len),
            Some(metadata),
            value,
            typed_value,
        ];
        let data = ParquetVariantData;
        Array::try_from_parts(ArrayParts::new(ParquetVariant, dtype, len, data).with_slots(slots))
    }
}

impl ParquetVariantData {
    pub(crate) fn validate_parts(
        validity: &Validity,
        metadata: &ArrayRef,
        value: Option<&ArrayRef>,
        typed_value: Option<&ArrayRef>,
        dtype: &DType,
        len: usize,
    ) -> VortexResult<()> {
        vortex_ensure!(
            matches!(dtype, DType::Variant(_)),
            "Expected Variant DType, found {dtype}"
        );
        vortex_ensure!(
            value.is_some() || typed_value.is_some(),
            "at least one of value or typed_value must be present"
        );

        vortex_ensure_eq!(
            dtype.nullability(),
            validity.nullability(),
            "variant dtype nullability must match validity nullability"
        );
        vortex_ensure_eq!(
            metadata.dtype(),
            &DType::Binary(Nullability::NonNullable),
            "metadata dtype must be non-nullable binary"
        );
        vortex_ensure_eq!(
            metadata.len(),
            len,
            "metadata length must match array length"
        );

        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure_eq!(validity_len, len, "validity length must match array length");
        }
        if let Some(v) = value {
            vortex_ensure!(
                matches!(v.dtype(), DType::Binary(_)),
                "value dtype must be binary, found {}",
                v.dtype()
            );
            vortex_ensure_eq!(v.len(), len, "value length must match array length");
        }
        if let Some(tv) = typed_value {
            vortex_ensure_eq!(tv.len(), len, "typed_value length must match array length");
        }
        Ok(())
    }

    /// Converts an Arrow `parquet_variant_compute::VariantArray` into a Vortex `ArrayRef`
    /// wrapping `VariantArray(ParquetVariantArray(...))`.
    pub fn from_arrow_variant(arrow_variant: &ArrowVariantArray) -> VortexResult<ArrayRef> {
        let storage = arrow_variant.inner();
        let value_nullable = storage
            .fields()
            .iter()
            .find(|field| field.name() == "value")
            .map(|field| field.is_nullable())
            .unwrap_or(false);
        let typed_value_nullable = storage
            .fields()
            .iter()
            .find(|field| field.name() == "typed_value")
            .map(|field| field.is_nullable())
            .unwrap_or(false);
        let validity = arrow_variant
            .nulls()
            .map(|nulls| {
                if nulls.null_count() == nulls.len() {
                    Validity::AllInvalid
                } else {
                    Validity::from(BitBuffer::from(nulls.inner().clone()))
                }
            })
            .unwrap_or(Validity::NonNullable);
        let metadata =
            ArrayRef::from_arrow(arrow_variant.metadata_field() as &dyn ArrowArray, false)?;

        let value = arrow_variant
            .value_field()
            .map(|v| ArrayRef::from_arrow(v as &dyn ArrowArray, value_nullable))
            .transpose()?;

        let typed_value = arrow_variant
            .typed_value_field()
            .map(|tv| ArrayRef::from_arrow(tv.as_ref(), typed_value_nullable))
            .transpose()?;

        let pv = ParquetVariant::try_new(validity, metadata, value, typed_value)?;
        Ok(VariantArray::new(pv.into_array()).into_array())
    }
}

pub(crate) fn validate_parts(
    validity: &Validity,
    metadata: &ArrayRef,
    value: Option<&ArrayRef>,
    typed_value: Option<&ArrayRef>,
    dtype: &DType,
    len: usize,
) -> VortexResult<()> {
    ParquetVariantData::validate_parts(validity, metadata, value, typed_value, dtype, len)
}

pub trait ParquetVariantArrayExt: TypedArrayRef<ParquetVariant> {
    fn metadata_array(&self) -> &ArrayRef {
        self.as_ref().slots()[METADATA_SLOT]
            .as_ref()
            .vortex_expect("ParquetVariantArray metadata slot")
    }

    fn validity(&self) -> Validity {
        child_to_validity(
            &self.as_ref().slots()[VALIDITY_SLOT],
            self.as_ref().dtype().nullability(),
        )
    }

    fn value_array(&self) -> Option<&ArrayRef> {
        self.as_ref().slots()[VALUE_SLOT].as_ref()
    }

    fn typed_value_array(&self) -> Option<&ArrayRef> {
        self.as_ref().slots()[TYPED_VALUE_SLOT].as_ref()
    }

    fn to_arrow(&self, ctx: &mut ExecutionCtx) -> VortexResult<ArrowVariantArray> {
        let metadata = self.metadata_array();
        let len = metadata.len();
        let nulls = to_arrow_null_buffer(self.validity(), len, ctx)?;

        let mut fields = Vec::with_capacity(3);
        let mut arrays: Vec<ArrowArrayRef> = Vec::with_capacity(3);

        let metadata_arrow = metadata.clone().execute_arrow(None, ctx)?;
        fields.push(Arc::new(Field::new(
            "metadata",
            metadata_arrow.data_type().clone(),
            false,
        )));
        arrays.push(metadata_arrow);

        if let Some(value) = self.value_array() {
            let value_arrow = value.clone().execute_arrow(None, ctx)?;
            fields.push(Arc::new(Field::new(
                "value",
                value_arrow.data_type().clone(),
                value.dtype().is_nullable(),
            )));
            arrays.push(value_arrow);
        }

        if let Some(typed_value) = self.typed_value_array() {
            let tv_arrow = typed_value.clone().execute_arrow(None, ctx)?;
            fields.push(Arc::new(Field::new(
                "typed_value",
                tv_arrow.data_type().clone(),
                typed_value.dtype().is_nullable(),
            )));
            arrays.push(tv_arrow);
        }

        let struct_array = arrow_array::StructArray::try_new(fields.into(), arrays, nulls)?;
        Ok(ArrowVariantArray::try_new(&struct_array)?)
    }
}

impl<T: TypedArrayRef<ParquetVariant>> ParquetVariantArrayExt for T {}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::Array as _;
    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::Int32Array;
    use arrow_array::StructArray;
    use arrow_array::builder::BinaryViewBuilder;
    use arrow_buffer::NullBuffer;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Fields;
    use parquet_variant::Variant as PqVariant;
    use parquet_variant_compute::VariantArray as ArrowVariantArray;
    use parquet_variant_compute::VariantArrayBuilder;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::Variant;
    use vortex_array::arrays::variant::VariantArrayExt;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ParquetVariant;
    use crate::ParquetVariantData;
    use crate::array::ParquetVariantArrayExt;

    fn assert_arrow_variant_storage_roundtrip(struct_array: StructArray) -> VortexResult<()> {
        let arrow_variant = ArrowVariantArray::try_new(&struct_array).unwrap();
        let vortex_arr = ParquetVariantData::from_arrow_variant(&arrow_variant)?;
        let variant_view = vortex_arr.as_opt::<Variant>().unwrap();
        let child = variant_view.child();
        let inner = child.as_opt::<ParquetVariant>().unwrap();

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let roundtripped = inner.to_arrow(&mut ctx)?;
        let roundtripped = roundtripped.inner();

        assert_eq!(struct_array.len(), roundtripped.len());
        assert_eq!(struct_array.column_names(), roundtripped.column_names());
        assert_eq!(struct_array.nulls(), roundtripped.nulls());
        assert_eq!(struct_array.fields().len(), roundtripped.fields().len());

        for (expected, actual) in struct_array
            .fields()
            .iter()
            .zip(roundtripped.fields().iter())
        {
            assert_eq!(expected.name(), actual.name());
            assert_eq!(expected.data_type(), actual.data_type());
            assert_eq!(expected.is_nullable(), actual.is_nullable());
        }

        for (expected, actual) in struct_array
            .columns()
            .iter()
            .zip(roundtripped.columns().iter())
        {
            assert_eq!(expected.to_data(), actual.to_data());
        }

        Ok(())
    }

    fn binary_view_array<const N: usize>(values: [&[u8]; N]) -> ArrowArrayRef {
        let mut builder = BinaryViewBuilder::new();
        for value in values {
            builder.append_value(value);
        }
        Arc::new(builder.finish())
    }

    #[test]
    fn test_from_arrow_variant_basic() -> VortexResult<()> {
        let mut builder = VariantArrayBuilder::new(3);
        builder.append_variant(PqVariant::from(42i32));
        builder.append_variant(PqVariant::from("hello"));
        builder.append_variant(PqVariant::from(true));
        let arrow_variant = builder.build();

        let vortex_arr = ParquetVariantData::from_arrow_variant(&arrow_variant)?;

        assert_eq!(vortex_arr.len(), 3);
        assert_eq!(
            vortex_arr.dtype(),
            &DType::Variant(Nullability::NonNullable)
        );

        Ok(())
    }

    #[test]
    fn test_from_arrow_variant_with_shredded_typed_value() -> VortexResult<()> {
        let mut metadata_builder = BinaryViewBuilder::new();
        let min_metadata = [1u8, 0];
        for _ in 0..3 {
            metadata_builder.append_value(min_metadata);
        }
        let metadata = metadata_builder.finish();

        let typed_value: ArrowArrayRef = Arc::new(Int32Array::from(vec![10, 20, 30]));

        let struct_fields: Fields = vec![
            Arc::new(Field::new("metadata", DataType::BinaryView, false)),
            Arc::new(Field::new("typed_value", DataType::Int32, false)),
        ]
        .into();
        let struct_array =
            StructArray::try_new(struct_fields, vec![Arc::new(metadata), typed_value], None)
                .unwrap();

        let arrow_variant = ArrowVariantArray::try_new(&struct_array).unwrap();

        let vortex_arr = ParquetVariantData::from_arrow_variant(&arrow_variant)?;
        assert_eq!(vortex_arr.len(), 3);
        assert_eq!(
            vortex_arr.dtype(),
            &DType::Variant(Nullability::NonNullable)
        );

        let variant_arr = vortex_arr.as_opt::<Variant>().unwrap();
        let inner = variant_arr.child().as_opt::<ParquetVariant>().unwrap();
        assert!(inner.typed_value_array().is_some());

        Ok(())
    }

    #[test]
    fn test_to_arrow_basic() -> VortexResult<()> {
        let metadata = VarBinViewArray::from_iter_bin([b"\x01\x00", b"\x01\x00"]).into_array();
        let value = VarBinViewArray::from_iter_bin([b"\x10", b"\x11"]).into_array();
        let pv_array = ParquetVariant::try_new(Validity::NonNullable, metadata, Some(value), None)?;

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let variant_arr = pv_array.to_arrow(&mut ctx)?;
        let struct_arr = variant_arr.inner();

        assert_eq!(struct_arr.num_columns(), 2);
        assert_eq!(struct_arr.column_names(), &["metadata", "value"]);

        Ok(())
    }

    #[test]
    fn test_to_arrow_with_typed_value() -> VortexResult<()> {
        let metadata = VarBinViewArray::from_iter_bin([b"\x01\x00", b"\x01\x00"]).into_array();
        let value = VarBinViewArray::from_iter_bin([b"\x10", b"\x11"]).into_array();
        let typed_value = buffer![1i32, 2].into_array();
        let pv_array = ParquetVariant::try_new(
            Validity::NonNullable,
            metadata,
            Some(value),
            Some(typed_value),
        )?;

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let variant_arr = pv_array.to_arrow(&mut ctx)?;
        let struct_arr = variant_arr.inner();

        assert_eq!(struct_arr.num_columns(), 3);
        assert_eq!(
            struct_arr.column_names(),
            &["metadata", "value", "typed_value"]
        );

        Ok(())
    }

    #[test]
    fn test_arrow_variant_roundtrip_unshredded_storage() -> VortexResult<()> {
        let mut builder = VariantArrayBuilder::new(3);
        builder.append_variant(PqVariant::from(42i32));
        builder.append_variant(PqVariant::from("hello"));
        builder.append_variant(PqVariant::from(true));

        assert_arrow_variant_storage_roundtrip(builder.build().into_inner())
    }

    #[test]
    fn test_arrow_variant_roundtrip_typed_value_only_storage() -> VortexResult<()> {
        let metadata = binary_view_array([b"\x01\x00", b"\x01\x00", b"\x01\x00"]);
        let typed_value: ArrowArrayRef = Arc::new(Int32Array::from(vec![10, 20, 30]));

        let struct_array = StructArray::try_new(
            vec![
                Arc::new(Field::new("metadata", DataType::BinaryView, false)),
                Arc::new(Field::new("typed_value", DataType::Int32, false)),
            ]
            .into(),
            vec![metadata, typed_value],
            None,
        )
        .unwrap();

        assert_arrow_variant_storage_roundtrip(struct_array)
    }

    #[test]
    fn test_arrow_variant_roundtrip_value_and_typed_value_storage() -> VortexResult<()> {
        let metadata = binary_view_array([b"\x01\x00", b"\x01\x00"]);
        let value = binary_view_array([b"\x10", b"\x11"]);
        let typed_value: ArrowArrayRef = Arc::new(Int32Array::from(vec![1, 2]));

        let struct_array = StructArray::try_new(
            vec![
                Arc::new(Field::new("metadata", DataType::BinaryView, false)),
                Arc::new(Field::new("value", DataType::BinaryView, true)),
                Arc::new(Field::new("typed_value", DataType::Int32, false)),
            ]
            .into(),
            vec![metadata, value, typed_value],
            None,
        )
        .unwrap();

        assert_arrow_variant_storage_roundtrip(struct_array)
    }

    #[test]
    fn test_arrow_variant_roundtrip_with_outer_nulls() -> VortexResult<()> {
        let metadata = binary_view_array([b"\x01\x00", b"\x01\x00", b"\x01\x00"]);
        let value = binary_view_array([b"\x10", b"\x00", b"\x11"]);
        let struct_array = StructArray::try_new(
            vec![
                Arc::new(Field::new("metadata", DataType::BinaryView, false)),
                Arc::new(Field::new("value", DataType::BinaryView, true)),
            ]
            .into(),
            vec![metadata, value],
            Some(NullBuffer::from(vec![true, false, true])),
        )
        .unwrap();

        assert_arrow_variant_storage_roundtrip(struct_array)
    }
}
