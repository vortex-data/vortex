// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use prost::Message;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::arrays::VariantArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::StatsSetRef;
use vortex_array::validity::Validity;
use vortex_array::vtable;
use vortex_array::vtable::Array;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTableFromValidityHelper;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_proto::dtype as pb;
use vortex_session::VortexSession;

use crate::array::NUM_SLOTS;
use crate::array::ParquetVariantArray;
use crate::array::SLOT_NAMES;
use crate::array::VALIDITY_SLOT;
use crate::kernel::PARENT_KERNELS;

/// VTable for [`ParquetVariantArray`].
#[derive(Debug, Clone)]
pub struct ParquetVariant;

impl ParquetVariant {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.parquet.variant");
}

/// Serialized metadata for a [`ParquetVariantArray`].
#[derive(Clone, Debug)]
pub struct ParquetVariantMetadata {
    /// Whether the un-shredded `value` child is present.
    pub has_value: bool,
    /// Whether the `value` child is nullable.
    ///
    /// In partially-shredded layouts, rows whose data lives entirely in `typed_value` have a
    /// null `value` slot, so the Arrow field is marked nullable. This flag preserves that
    /// distinction across serialization round-trips.
    pub value_nullable: bool,
    /// DType of the shredded `typed_value`, if present.
    ///
    /// This is required to deserialize non-variant shredded children.
    pub typed_value_dtype: Option<DType>,
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

vtable!(ParquetVariant);

impl VTable for ParquetVariant {
    type Array = ParquetVariantArray;
    type Metadata = ParquetVariantMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;

    fn vtable(_array: &Self::Array) -> &Self {
        &ParquetVariant
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &ParquetVariantArray) -> usize {
        array.metadata_array().len()
    }

    fn dtype(array: &ParquetVariantArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ParquetVariantArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: Hasher>(array: &ParquetVariantArray, state: &mut H, precision: Precision) {
        array.validity.array_hash(state, precision);
        array.metadata_array().array_hash(state, precision);
        // Hash discriminators so that (value=Some, typed_value=None) and
        // (value=None, typed_value=Some) produce different hashes.
        array.value_array().is_some().hash(state);
        if let Some(value) = array.value_array() {
            value.array_hash(state, precision);
        }
        array.typed_value_array().is_some().hash(state);
        if let Some(typed_value) = array.typed_value_array() {
            typed_value.array_hash(state, precision);
        }
    }

    fn array_eq(
        array: &ParquetVariantArray,
        other: &ParquetVariantArray,
        precision: Precision,
    ) -> bool {
        if !array.validity.array_eq(&other.validity, precision)
            || !array
                .metadata_array()
                .array_eq(other.metadata_array(), precision)
        {
            return false;
        }
        match (array.value_array(), other.value_array()) {
            (Some(a), Some(b)) => {
                if !a.array_eq(b, precision) {
                    return false;
                }
            }
            (None, None) => {}
            _ => return false,
        }
        match (array.typed_value_array(), other.typed_value_array()) {
            (Some(a), Some(b)) => a.array_eq(b, precision),
            (None, None) => true,
            _ => false,
        }
    }

    fn nbuffers(_array: &ParquetVariantArray) -> usize {
        0
    }

    fn buffer(_array: &ParquetVariantArray, idx: usize) -> BufferHandle {
        vortex_panic!("ParquetVariantArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &ParquetVariantArray, _idx: usize) -> Option<String> {
        None
    }

    fn slots(array: &ParquetVariantArray) -> &[Option<ArrayRef>] {
        &array.slots
    }

    fn slot_name(_array: &ParquetVariantArray, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn metadata(array: &ParquetVariantArray) -> VortexResult<Self::Metadata> {
        Ok(ParquetVariantMetadata {
            has_value: array.value_array().is_some(),
            value_nullable: array.value_array().is_some_and(|v| v.dtype().is_nullable()),
            typed_value_dtype: array.typed_value_array().map(|tv| tv.dtype().clone()),
        })
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        let typed_value_dtype = metadata
            .typed_value_dtype
            .as_ref()
            .map(|dtype| dtype.try_into())
            .transpose()?;
        Ok(Some(
            ParquetVariantMetadataProto {
                has_value: metadata.has_value,
                typed_value_dtype,
                value_nullable: metadata.value_nullable,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        let proto = ParquetVariantMetadataProto::decode(bytes)?;
        let typed_value_dtype = match proto.typed_value_dtype.as_ref() {
            Some(dtype) => Some(DType::from_proto(dtype, session)?),
            None => None,
        };
        Ok(ParquetVariantMetadata {
            has_value: proto.has_value,
            value_nullable: proto.value_nullable,
            typed_value_dtype,
        })
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ParquetVariantArray> {
        vortex_ensure!(matches!(dtype, DType::Variant(_)), "Expected Variant DType");
        let has_typed_value = metadata.typed_value_dtype.is_some();
        vortex_ensure!(
            metadata.has_value || has_typed_value,
            "At least one of value or typed_value must be present"
        );

        let expected_children = 1 + metadata.has_value as usize + has_typed_value as usize;
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

        let value = if metadata.has_value {
            let v = children.get(
                child_idx,
                &DType::Binary(metadata.value_nullable.into()),
                len,
            )?;
            child_idx += 1;
            Some(v)
        } else {
            None
        };

        let typed_value = if has_typed_value {
            // typed_value can be any type — primitive, list, struct, etc.
            let dtype = metadata
                .typed_value_dtype
                .clone()
                .ok_or_else(|| vortex_err!("typed_value_dtype missing for typed_value child"))?;
            let tv = children.get(child_idx, &dtype, len)?;
            Some(tv)
        } else {
            None
        };

        ParquetVariantArray::try_new(validity, variant_metadata, value, typed_value)
    }

    fn with_slots(array: &mut Self::Array, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "ParquetVariantArray expects {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );
        // Update validity from the validity slot.
        if let Some(validity_child) = &slots[VALIDITY_SLOT] {
            array.validity = Validity::Array(validity_child.clone());
        }
        array.slots = slots;
        Ok(())
    }

    fn execute(array: Arc<Array<Self>>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            VariantArray::new(array.as_ref().clone().into_array()).into_array(),
        ))
    }

    fn execute_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
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
    use vortex_array::serde::ArrayParts;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::session::ArraySessionExt;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use crate::ParquetVariant;
    use crate::ParquetVariantArray;

    fn roundtrip(array: ArrayRef) -> ArrayRef {
        let dtype = array.dtype().clone();
        let len = array.len();

        let ctx = ArrayContext::empty();
        let serialized = array.serialize(&ctx, &SerializeOptions::default()).unwrap();

        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let concat = concat.freeze();

        let session = VortexSession::empty().with::<vortex_array::session::ArraySession>();
        session.arrays().register(ParquetVariant);
        session.arrays().register(Variant);

        let parts = ArrayParts::try_from(concat).unwrap();
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
        let inner_pv = ParquetVariantArray::try_new(
            Validity::NonNullable,
            inner_metadata,
            Some(inner_value),
            None,
        )
        .unwrap();
        let typed_value = VariantArray::new(inner_pv.into_array()).into_array();

        let outer_pv = ParquetVariantArray::try_new(
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

        let pv = ParquetVariantArray::try_new(validity, metadata, Some(value), None).unwrap();
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

        let outer_pv = ParquetVariantArray::try_new(
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
