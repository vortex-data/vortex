// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hasher;

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::arrays::VariantArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::VTable;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_proto::dtype as pb;
use vortex_session::VortexSession;

use crate::array::METADATA_SLOT;
use crate::array::ParquetVariantArrayExt;
use crate::array::ParquetVariantData;
use crate::array::SLOT_NAMES;
use crate::array::TYPED_VALUE_SLOT;
use crate::array::VALIDITY_SLOT;
use crate::array::VALUE_SLOT;
use crate::array::validate_parts;
use crate::kernel::PARENT_KERNELS;

/// VTable for [`ParquetVariantArray`].
#[derive(Debug, Clone)]
pub struct ParquetVariant;

impl ParquetVariant {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.parquet.variant");
}

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
    type ArrayData = ParquetVariantData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn validate(
        &self,
        data: &Self::ArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let _ = data;
        let validity = child_to_validity(&slots[VALIDITY_SLOT], dtype.nullability());
        let metadata = slots[METADATA_SLOT]
            .as_ref()
            .ok_or_else(|| vortex_err!("ParquetVariantArray metadata slot"))?;
        validate_parts(
            &validity,
            metadata,
            slots[VALUE_SLOT].as_ref(),
            slots[TYPED_VALUE_SLOT].as_ref(),
            dtype,
            len,
        )
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

        ParquetVariantData::validate_parts(
            &validity,
            &variant_metadata,
            value.as_ref(),
            typed_value.as_ref(),
            dtype,
            len,
        )?;
        let slots = vec![
            validity_to_child(&validity, len),
            Some(variant_metadata),
            value,
            typed_value,
        ];
        let data = ParquetVariantData;
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            VariantArray::new(array.as_ref().clone().into_array()).into_array(),
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

impl ArrayHash for ParquetVariantData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _precision: Precision) {}
}

impl ArrayEq for ParquetVariantData {
    fn array_eq(&self, _other: &Self, _precision: Precision) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::ArrayContext;
    use vortex_array::ArrayEq;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::Precision;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::Variant;
    use vortex_array::arrays::VariantArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_array::session::ArraySessionExt;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use crate::ParquetVariant;
    use crate::array::ParquetVariantArrayExt;
    fn roundtrip(array: ArrayRef) -> ArrayRef {
        let dtype = array.dtype().clone();
        let len = array.len();

        let session = VortexSession::empty().with::<vortex_array::session::ArraySession>();
        let ctx = ArrayContext::empty();
        let serialized = array
            .serialize(&ctx, &session, &SerializeOptions::default())
            .unwrap();

        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let concat = concat.freeze();
        session.arrays().register(ParquetVariant);
        session.arrays().register(Variant);

        let parts = SerializedArray::try_from(concat).unwrap();
        parts
            .decode(&dtype, len, &ReadContext::new(ctx.to_ids()), &session)
            .unwrap()
    }

    #[test]
    fn test_serde_roundtrip_typed_value_variant() {
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
        )
        .unwrap();
        let typed_value = VariantArray::new(inner_pv.into_array()).into_array();

        let outer_pv = ParquetVariant::try_new(
            Validity::NonNullable,
            outer_metadata,
            None,
            Some(typed_value),
        )
        .unwrap();
        let array = outer_pv.into_array();
        let decoded = roundtrip(array.clone());

        assert!(array.array_eq(&decoded, Precision::Value));
        let decoded_pv = decoded.as_opt::<ParquetVariant>().unwrap();
        let typed = decoded_pv.typed_value_array().unwrap();
        assert_eq!(typed.dtype(), &DType::Variant(Nullability::NonNullable));
    }

    #[test]
    fn test_serde_roundtrip_with_nullable_validity() {
        let metadata =
            VarBinViewArray::from_iter_bin([b"\x01\x00", b"\x01\x00", b"\x01\x00"]).into_array();
        let value = VarBinViewArray::from_iter_bin([b"\x10", b"\x11", b"\x12"]).into_array();
        let validity = Validity::from(BitBuffer::from_iter([true, false, true]));

        let pv = ParquetVariant::try_new(validity, metadata, Some(value), None).unwrap();
        let array = pv.into_array();
        let decoded = roundtrip(array.clone());

        assert!(array.array_eq(&decoded, Precision::Value));
        assert_eq!(decoded.dtype(), &DType::Variant(Nullability::Nullable));
        let decoded_pv = decoded.as_opt::<ParquetVariant>().unwrap();
        assert!(decoded_pv.value_array().is_some());
        assert!(decoded_pv.typed_value_array().is_none());
    }

    #[test]
    fn test_serde_roundtrip_typed_value_int32() {
        let outer_metadata =
            VarBinViewArray::from_iter_bin([b"\x01\x00", b"\x01\x00", b"\x01\x00"]).into_array();
        let typed_value = buffer![10i32, 20, 30].into_array();

        let outer_pv = ParquetVariant::try_new(
            Validity::NonNullable,
            outer_metadata,
            None,
            Some(typed_value),
        )
        .unwrap();
        let array = outer_pv.into_array();
        let decoded = roundtrip(array.clone());

        assert!(array.array_eq(&decoded, Precision::Value));
        let decoded_pv = decoded.as_opt::<ParquetVariant>().unwrap();
        let typed = decoded_pv.typed_value_array().unwrap();
        assert_eq!(
            typed.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
    }
}
