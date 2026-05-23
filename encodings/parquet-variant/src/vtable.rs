// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::EmptyArrayData;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::arrays::VariantArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::serde::ArrayChildren;
use vortex_array::smallvec::smallvec;
use vortex_array::validity::Validity;
use vortex_array::vtable::VTable;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_proto::dtype as pb;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::array::METADATA_SLOT;
use crate::array::NUM_SLOTS;
use crate::array::ParquetVariantArrayExt;
use crate::array::SLOT_NAMES;
use crate::array::TYPED_VALUE_SLOT;
use crate::array::VALIDITY_SLOT;
use crate::array::VALUE_SLOT;
use crate::array::core_storage_without_typed_value;
use crate::array::logical_shredded_from_parquet_typed_value;
use crate::kernel::PARENT_KERNELS;

/// VTable for Arrow's canonical `arrow.parquet.variant` extension storage.
///
/// `ParquetVariantArray` preserves semi-structured data stored as Parquet Variant values in a
/// lossless form and supports both unshredded and shredded layouts. Its storage matches the
/// canonical extension type contract:
/// - `metadata` is always present and non-nullable.
/// - `value` stores unshredded variant bytes when present.
/// - `typed_value` stores shredded data when present.
///
/// At least one of `value` or `typed_value` must be present. `typed_value` may be a primitive,
/// list, or struct, with nested shredded children following the same recursive rules as the
/// Arrow canonical extension type docs.
///
/// Row values are interpreted according to the [Parquet Variant shredding] rules:
///
/// | `value`  | `typed_value` | Meaning |
/// |----------|---------------|---------|
/// | null     | null          | Missing value; only valid for shredded object fields. |
/// | non-null | null          | Unshredded value decoded from `metadata` and `value`. |
/// | null     | non-null      | Perfectly shredded value decoded from `typed_value`. |
/// | non-null | non-null      | Partially shredded object; merge shredded fields with raw-only fields from `value`. |
///
/// The final row is only valid for object shredding. Duplicate field names between `value` and
/// `typed_value` are invalid writer output.
///
/// [Parquet Variant shredding]: https://github.com/apache/parquet-format/blob/master/VariantShredding.md#value-shredding
#[derive(Debug, Clone)]
pub struct ParquetVariant;

#[derive(Clone, prost::Message)]
struct ParquetVariantMetadataProto {
    /// Whether the un-shredded `value` child is present.
    #[prost(bool, tag = "1")]
    pub has_value: bool,
    /// DType of the shredded `typed_value`, if present.
    #[prost(message, optional, tag = "2")]
    pub typed_value_dtype: Option<pb::DType>,
    /// Whether the `value` child is nullable.
    #[prost(bool, tag = "3")]
    pub value_nullable: bool,
}

/// A [`ParquetVariant`]-encoded Vortex array.
pub type ParquetVariantArray = Array<ParquetVariant>;

impl VTable for ParquetVariant {
    type TypedArrayData = EmptyArrayData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.parquet.variant");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "ParquetVariantArray expects {NUM_SLOTS} slots, got {}",
            slots.len()
        );
        let validity = child_to_validity(slots[VALIDITY_SLOT].as_ref(), dtype.nullability());
        let metadata = slots[METADATA_SLOT]
            .as_ref()
            .ok_or_else(|| vortex_err!("ParquetVariantArray metadata slot"))?;
        let value = slots[VALUE_SLOT].as_ref();
        let typed_value = slots[TYPED_VALUE_SLOT].as_ref();

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
        if let Some(value) = value {
            vortex_ensure!(
                matches!(value.dtype(), DType::Binary(_)),
                "value dtype must be binary, found {}",
                value.dtype()
            );
            vortex_ensure_eq!(value.len(), len, "value length must match array length");
        }
        if let Some(typed_value) = typed_value {
            vortex_ensure_eq!(
                typed_value.len(),
                len,
                "typed_value length must match array length"
            );
        }
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("ParquetVariantArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let typed_value_dtype = array
            .typed_value_array()
            .map(|tv| tv.dtype().try_into())
            .transpose()?;
        Ok(Some(
            ParquetVariantMetadataProto {
                has_value: array.value_array().is_some(),
                typed_value_dtype,
                value_nullable: array.value_array().is_some_and(|v| v.dtype().is_nullable()),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(
            buffers.is_empty(),
            "ParquetVariantArray expects 0 buffers, got {}",
            buffers.len()
        );

        let proto = ParquetVariantMetadataProto::decode(metadata)?;
        let typed_value_dtype = match proto.typed_value_dtype.as_ref() {
            Some(dtype) => Some(DType::from_proto(dtype, session)?),
            None => None,
        };

        vortex_ensure!(matches!(dtype, DType::Variant(_)), "Expected Variant DType");
        let has_typed_value = typed_value_dtype.is_some();
        vortex_ensure!(
            proto.has_value || has_typed_value,
            "At least one of value or typed_value must be present"
        );

        let expected_children = 1 + proto.has_value as usize + has_typed_value as usize;
        vortex_ensure!(
            children.len() == expected_children || children.len() == expected_children + 1,
            "Expected {} or {} children, got {}",
            expected_children,
            expected_children + 1,
            children.len()
        );

        let (validity, mut child_idx) = if children.len() == expected_children {
            (Validity::from(dtype.nullability()), 0)
        } else {
            (Validity::Array(children.get(0, &Validity::DTYPE, len)?), 1)
        };
        let variant_metadata =
            children.get(child_idx, &DType::Binary(Nullability::NonNullable), len)?;
        child_idx += 1;

        let value = if proto.has_value {
            let v = children.get(child_idx, &DType::Binary(proto.value_nullable.into()), len)?;
            child_idx += 1;
            Some(v)
        } else {
            None
        };

        let typed_value = if has_typed_value {
            // typed_value can be any type — primitive, list, struct, etc.
            let dtype = typed_value_dtype
                .ok_or_else(|| vortex_err!("typed_value_dtype missing for typed_value child"))?;
            let tv = children.get(child_idx, &dtype, len)?;
            Some(tv)
        } else {
            None
        };

        let slots = smallvec![
            validity_to_child(&validity, len),
            Some(variant_metadata),
            value,
            typed_value,
        ];
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, EmptyArrayData).with_slots(slots))
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let shredded = array
            .typed_value_array()
            .cloned()
            .map(|typed_value| {
                logical_shredded_from_parquet_typed_value(array.metadata_array(), typed_value, ctx)
            })
            .transpose()?;
        let core_storage = core_storage_without_typed_value(&array)?;
        Ok(ExecutionResult::done(
            VariantArray::try_new(core_storage, shredded)?.into_array(),
        ))
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::Int32Array;
    use arrow_array::StructArray;
    use arrow_array::builder::BinaryViewBuilder;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use parquet_variant_compute::VariantArray as ArrowVariantArray;
    use rstest::fixture;
    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::ArrayEq;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::Precision;
    use vortex_array::VTable;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::VariantArray;
    use vortex_array::arrays::variant::VariantArrayExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_array::session::ArraySession;
    use vortex_array::session::ArraySessionExt;
    use vortex_array::stream::ArrayStreamExt;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_file::OpenOptionsSessionExt;
    use vortex_file::WriteOptionsSessionExt;
    use vortex_io::session::RuntimeSession;
    use vortex_layout::LayoutStrategy;
    use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
    use vortex_layout::session::LayoutSession;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use crate::ParquetVariant;
    use crate::array::ParquetVariantArrayExt;

    fn roundtrip(array: ArrayRef) -> VortexResult<ArrayRef> {
        let dtype = array.dtype().clone();
        let len = array.len();

        let session = VortexSession::empty().with::<ArraySession>();
        session.arrays().register(ParquetVariant);

        let ctx = ArrayContext::empty();
        let serialized = array.serialize(&ctx, &session, &SerializeOptions::default())?;

        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let concat = concat.freeze();

        let parts = SerializedArray::try_from(concat)?;
        parts.decode(&dtype, len, &ReadContext::new(ctx.to_ids()), &session)
    }

    #[fixture]
    fn typed_value_variant_array() -> VortexResult<ArrayRef> {
        let mut metadata = BinaryViewBuilder::new();
        for _ in 0..3 {
            metadata.append_value(b"\x01\x00");
        }
        let metadata: ArrowArrayRef = Arc::new(metadata.finish());
        let typed_value: ArrowArrayRef = Arc::new(Int32Array::from(vec![10, 20, 30]));
        let arrow_storage = StructArray::try_new(
            vec![
                Arc::new(Field::new("metadata", DataType::BinaryView, false)),
                Arc::new(Field::new("typed_value", DataType::Int32, false)),
            ]
            .into(),
            vec![metadata, typed_value],
            None,
        )?;

        ParquetVariant::from_arrow_variant(&ArrowVariantArray::try_new(&arrow_storage)?)
    }

    #[fixture]
    fn parquet_variant_file_session() -> VortexSession {
        let session = VortexSession::empty()
            .with::<ArraySession>()
            .with::<LayoutSession>()
            .with::<RuntimeSession>();
        vortex_file::register_default_encodings(&session);
        session.arrays().register(ParquetVariant);
        session
    }

    #[fixture]
    fn write_strategy() -> Arc<dyn LayoutStrategy> {
        let mut allowed = vortex_file::ALLOWED_ENCODINGS.clone();
        allowed.insert(ParquetVariant.id());
        vortex_file::WriteStrategyBuilder::default()
            .with_allow_encodings(allowed)
            .build()
    }

    #[test]
    fn test_execute_exposes_typed_value_as_canonical_shredded() -> VortexResult<()> {
        let metadata =
            VarBinViewArray::from_iter_bin([b"\x01\x00", b"\x01\x00", b"\x01\x00"]).into_array();
        let typed_value =
            PrimitiveArray::from_option_iter([Some(10i32), None, Some(30)]).into_array();

        let parquet_variant =
            ParquetVariant::try_new(Validity::NonNullable, metadata, None, Some(typed_value))?;
        assert!(parquet_variant.typed_value_array().is_some());
        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        let Canonical::Variant(variant) = parquet_variant
            .into_array()
            .execute::<Canonical>(&mut ctx)?
        else {
            return Err(vortex_err!("expected canonical variant"));
        };

        let core_storage = variant
            .core_storage()
            .as_opt::<ParquetVariant>()
            .ok_or_else(|| vortex_err!("expected parquet variant core storage"))?;
        assert!(core_storage.typed_value_array().is_none());
        let shredded = variant
            .shredded()
            .ok_or_else(|| vortex_err!("expected canonical shredded child"))?;
        assert_eq!(
            shredded.dtype(),
            &DType::Primitive(PType::I32, Nullability::Nullable)
        );
        let shredded = shredded.clone().execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(
            shredded,
            PrimitiveArray::from_option_iter([Some(10), None, Some(30)])
        );

        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn test_file_roundtrip_typed_value_variant_with_statistics(
        #[from(typed_value_variant_array)] expected: VortexResult<ArrayRef>,
        parquet_variant_file_session: VortexSession,
    ) -> VortexResult<()> {
        let expected = expected?;

        let mut bytes = ByteBufferMut::empty();
        parquet_variant_file_session
            .write_options()
            .with_strategy(Arc::new(FlatLayoutStrategy::default()))
            .write(&mut bytes, expected.to_array_stream())
            .await?;

        let actual = parquet_variant_file_session
            .open_options()
            .open_buffer(bytes)?
            .scan()?
            .into_array_stream()?
            .read_all()
            .await?;

        assert_arrays_eq!(expected, actual);
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn test_file_roundtrip_typed_value_variant_with_zoned_strategy(
        #[from(typed_value_variant_array)] expected: VortexResult<ArrayRef>,
        parquet_variant_file_session: VortexSession,
        write_strategy: Arc<dyn LayoutStrategy>,
    ) -> VortexResult<()> {
        let expected = expected?;

        let mut bytes = ByteBufferMut::empty();
        parquet_variant_file_session
            .write_options()
            .with_strategy(write_strategy)
            .write(&mut bytes, expected.to_array_stream())
            .await?;

        let actual = parquet_variant_file_session
            .open_options()
            .open_buffer(bytes)?
            .scan()?
            .into_array_stream()?
            .read_all()
            .await?;

        assert_arrays_eq!(expected, actual);
        Ok(())
    }

    #[rstest]
    fn test_serde_roundtrip_typed_value_variant() -> VortexResult<()> {
        let outer_metadata =
            VarBinViewArray::from_iter_bin([b"\x01\x00", b"\x01\x00", b"\x01\x00"]).into_array();

        let inner_metadata =
            VarBinViewArray::from_iter_bin([b"\x01\x00", b"\x01\x00", b"\x01\x00"]).into_array();
        let inner_value = VarBinViewArray::from_iter_bin([b"\x02", b"\x03", b"\x04"]).into_array();
        let inner_pv = ParquetVariant::try_new(
            Validity::NonNullable,
            inner_metadata,
            Some(inner_value),
            None,
        )?;
        let typed_value = VariantArray::try_new(inner_pv.into_array(), None)?.into_array();

        let outer_pv = ParquetVariant::try_new(
            Validity::NonNullable,
            outer_metadata,
            None,
            Some(typed_value),
        )?;
        let array = outer_pv.into_array();
        let decoded = roundtrip(array.clone())?;

        assert!(array.array_eq(&decoded, Precision::Value));
        let decoded_pv = decoded
            .as_opt::<ParquetVariant>()
            .ok_or_else(|| vortex_err!("expected parquet variant array"))?;
        let typed = decoded_pv
            .typed_value_array()
            .ok_or_else(|| vortex_err!("expected typed_value child"))?;
        assert_eq!(typed.dtype(), &DType::Variant(Nullability::NonNullable));
        Ok(())
    }

    #[rstest]
    fn test_serde_roundtrip_with_nullable_validity() -> VortexResult<()> {
        let metadata =
            VarBinViewArray::from_iter_bin([b"\x01\x00", b"\x01\x00", b"\x01\x00"]).into_array();
        let value = VarBinViewArray::from_iter_bin([b"\x10", b"\x11", b"\x12"]).into_array();
        let validity = Validity::from(BitBuffer::from_iter([true, false, true]));

        let pv = ParquetVariant::try_new(validity, metadata, Some(value), None)?;
        let array = pv.into_array();
        let decoded = roundtrip(array.clone())?;

        assert!(array.array_eq(&decoded, Precision::Value));
        assert_eq!(decoded.dtype(), &DType::Variant(Nullability::Nullable));
        let decoded_pv = decoded
            .as_opt::<ParquetVariant>()
            .ok_or_else(|| vortex_err!("expected parquet variant array"))?;
        assert!(decoded_pv.value_array().is_some());
        assert!(decoded_pv.typed_value_array().is_none());
        Ok(())
    }

    #[rstest]
    fn test_serde_roundtrip_typed_value_int32() -> VortexResult<()> {
        let outer_metadata =
            VarBinViewArray::from_iter_bin([b"\x01\x00", b"\x01\x00", b"\x01\x00"]).into_array();
        let typed_value = buffer![10i32, 20, 30].into_array();

        let outer_pv = ParquetVariant::try_new(
            Validity::NonNullable,
            outer_metadata,
            None,
            Some(typed_value),
        )?;
        let array = outer_pv.into_array();
        let decoded = roundtrip(array.clone())?;

        assert!(array.array_eq(&decoded, Precision::Value));
        let decoded_pv = decoded
            .as_opt::<ParquetVariant>()
            .ok_or_else(|| vortex_err!("expected parquet variant array"))?;
        let typed = decoded_pv
            .typed_value_array()
            .ok_or_else(|| vortex_err!("expected typed_value child"))?;
        assert_eq!(
            typed.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        Ok(())
    }
}
