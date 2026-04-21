// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod operations;
mod rules;
mod validity;

use prost::Message;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::variant::CORE_STORAGE_SLOT;
use crate::arrays::variant::NUM_SLOTS;
use crate::arrays::variant::SHREDDED_SLOT;
use crate::arrays::variant::SLOT_NAMES;
use crate::arrays::variant::VariantMetadata;
use crate::arrays::variant::try_derived_shredded_from_core_storage;
use crate::arrays::variant::vtable::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::serde::ArrayChildren;

/// A canonical Vortex array of [`DType::Variant`] values.
///
/// The array stores its mandatory `core_storage` child in slot 0 and an optional same-length
/// logical `shredded` child. See [`crate::arrays::variant`] for the canonical layout contract.
/// Derived `shredded` children are delegated from `core_storage` and are not stored as physical
/// children.
pub type VariantArray = Array<Variant>;

/// VTable for the canonical two-child [`VariantArray`] layout.
///
/// Validation and serde enforce the inline-vs-derived `shredded` contract described by
/// [`VariantMetadata`].
#[derive(Clone, Debug)]
pub struct Variant;

#[derive(Clone, prost::Message)]
struct VariantMetadataProto {
    #[prost(oneof = "variant_metadata_proto::Shredded", tags = "1, 2")]
    pub shredded: Option<variant_metadata_proto::Shredded>,
}

mod variant_metadata_proto {
    use prost::Oneof;
    use vortex_proto::dtype as pb;

    #[derive(Clone, Oneof)]
    pub enum Shredded {
        #[prost(message, tag = "1")]
        InlineDtype(pb::DType),
        #[prost(string, tag = "2")]
        DerivedSlotName(String),
    }
}

impl VTable for Variant {
    type ArrayData = VariantMetadata;

    type OperationsVTable = Self;

    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.variant");
        *ID
    }

    fn validate(
        &self,
        data: &Self::ArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "VariantArray expects {NUM_SLOTS} slots, got {}",
            slots.len()
        );
        let core_storage = slots[CORE_STORAGE_SLOT]
            .as_ref()
            .vortex_expect("validated core storage slot presence");
        vortex_ensure!(
            matches!(dtype, DType::Variant(_)),
            "Expected Variant DType, got {dtype}"
        );
        vortex_ensure!(
            core_storage.dtype() == dtype,
            "VariantArray core_storage dtype {} does not match outer dtype {}",
            core_storage.dtype(),
            dtype
        );
        vortex_ensure!(
            core_storage.len() == len,
            "VariantArray core_storage length {} does not match outer length {}",
            core_storage.len(),
            len
        );
        match data {
            VariantMetadata::None => {
                vortex_ensure!(
                    slots[SHREDDED_SLOT].is_none(),
                    "VariantArray without shredded metadata must not populate slot {SHREDDED_SLOT}"
                );
            }
            VariantMetadata::Inline { shredded_dtype } => {
                let shredded = slots[SHREDDED_SLOT].as_ref().ok_or_else(|| {
                    vortex_error::vortex_err!("VariantArray missing inline shredded slot")
                })?;
                vortex_ensure!(
                    shredded.len() == len,
                    "VariantArray shredded length {} does not match outer length {}",
                    shredded.len(),
                    len
                );
                vortex_ensure!(
                    shredded.dtype() == shredded_dtype,
                    "VariantArray inline shredded dtype {} does not match metadata {}",
                    shredded.dtype(),
                    shredded_dtype
                );
            }
            VariantMetadata::Derived { slot_name } => {
                vortex_ensure!(
                    slots[SHREDDED_SLOT].is_none(),
                    "VariantArray derived shredded child must not populate physical slot {SHREDDED_SLOT}"
                );
                let derived_shredded = try_derived_shredded_from_core_storage(core_storage, slot_name)?
                    .ok_or_else(|| vortex_error::vortex_err!(
                        "VariantArray derived shredded slot {slot_name} is not exposed by core_storage"
                    ))?;
                vortex_ensure!(
                    derived_shredded.len() == len,
                    "VariantArray shredded length {} does not match outer length {}",
                    derived_shredded.len(),
                    len
                );
            }
        }
        Ok(())
    }

    fn nchildren(array: ArrayView<'_, Self>) -> usize {
        match array.data() {
            VariantMetadata::Inline { .. } => 2,
            VariantMetadata::None | VariantMetadata::Derived { .. } => 1,
        }
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("VariantArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let metadata = match array.data() {
            VariantMetadata::None => vec![],
            VariantMetadata::Inline { shredded_dtype } => VariantMetadataProto {
                shredded: Some(variant_metadata_proto::Shredded::InlineDtype(
                    shredded_dtype.try_into()?,
                )),
            }
            .encode_to_vec(),
            VariantMetadata::Derived { slot_name } => VariantMetadataProto {
                shredded: Some(variant_metadata_proto::Shredded::DerivedSlotName(
                    slot_name.clone(),
                )),
            }
            .encode_to_vec(),
        };
        Ok(Some(metadata))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(matches!(dtype, DType::Variant(_)), "Expected Variant DType");
        let variant_metadata = if metadata.is_empty() {
            VariantMetadata::None
        } else {
            let proto = VariantMetadataProto::decode(metadata)?;
            match proto.shredded {
                Some(variant_metadata_proto::Shredded::InlineDtype(dtype)) => {
                    VariantMetadata::Inline {
                        shredded_dtype: DType::from_proto(&dtype, session)?,
                    }
                }
                Some(variant_metadata_proto::Shredded::DerivedSlotName(slot_name)) => {
                    VariantMetadata::Derived { slot_name }
                }
                None => VariantMetadata::None,
            }
        };
        let expected_children = match &variant_metadata {
            VariantMetadata::Inline { .. } => 2,
            VariantMetadata::None | VariantMetadata::Derived { .. } => 1,
        };
        vortex_ensure!(
            children.len() == expected_children,
            "Expected {expected_children} children, got {}",
            children.len()
        );
        let core_storage = children.get(CORE_STORAGE_SLOT, dtype, len)?;
        let shredded = match &variant_metadata {
            VariantMetadata::Inline { shredded_dtype } => {
                Some(children.get(SHREDDED_SLOT, shredded_dtype, len)?)
            }
            VariantMetadata::None | VariantMetadata::Derived { .. } => None,
        };
        Ok(
            ArrayParts::new(self.clone(), dtype.clone(), len, variant_metadata)
                .with_slots(vec![Some(core_storage), shredded]),
        )
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        match SLOT_NAMES.get(idx) {
            Some(name) => (*name).to_string(),
            None => vortex_panic!("VariantArray slot_name index {idx} out of bounds"),
        }
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_ensure;
    use vortex_error::vortex_panic;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;
    use vortex_session::registry::CachedId;
    use vortex_session::registry::ReadContext;

    use super::VariantArray;
    use super::VariantMetadata;
    use crate::ArrayContext;
    use crate::ArrayEq;
    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::Precision;
    use crate::array::Array;
    use crate::array::ArrayId;
    use crate::array::ArrayParts;
    use crate::array::ArrayView;
    use crate::array::EmptyArrayData;
    use crate::array::NotSupported;
    use crate::array::VTable;
    use crate::array::ValidityChild;
    use crate::array::ValidityVTableFromChild;
    use crate::arrays::ConstantArray;
    use crate::arrays::DictArray;
    use crate::arrays::FilterArray;
    use crate::arrays::MaskedArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::SliceArray;
    use crate::arrays::variant::VariantArrayExt;
    use crate::assert_arrays_eq;
    use crate::buffer::BufferHandle;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;
    use crate::serde::ArrayChildren;
    use crate::serde::SerializeOptions;
    use crate::serde::SerializedArray;
    use crate::validity::Validity;

    fn roundtrip(array: ArrayRef) -> ArrayRef {
        let dtype = array.dtype().clone();
        let len = array.len();

        let ctx = ArrayContext::empty();
        let serialized = array
            .serialize(&ctx, &LEGACY_SESSION, &SerializeOptions::default())
            .unwrap();

        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let parts = SerializedArray::try_from(concat.freeze()).unwrap();
        parts
            .decode(
                &dtype,
                len,
                &ReadContext::new(ctx.to_ids()),
                &LEGACY_SESSION,
            )
            .unwrap()
    }

    #[derive(Clone, Debug)]
    struct DelegatingCoreStorage;

    impl VTable for DelegatingCoreStorage {
        type ArrayData = EmptyArrayData;
        type OperationsVTable = NotSupported;
        type ValidityVTable = ValidityVTableFromChild;

        fn id(&self) -> ArrayId {
            static ID: CachedId = CachedId::new("vortex.variant.test.delegating_core_storage");
            *ID
        }

        fn validate(
            &self,
            _data: &EmptyArrayData,
            dtype: &DType,
            len: usize,
            slots: &[Option<ArrayRef>],
        ) -> VortexResult<()> {
            vortex_ensure!(
                matches!(dtype, DType::Variant(_)),
                "expected variant dtype, found {dtype}"
            );
            vortex_ensure!(slots.len() == 1, "expected 1 slot, got {}", slots.len());
            let delegated = slots[0]
                .as_ref()
                .ok_or_else(|| vortex_error::vortex_err!("missing delegated slot"))?;
            vortex_ensure!(
                delegated.len() == len,
                "delegated length {} does not match outer length {}",
                delegated.len(),
                len
            );
            Ok(())
        }

        fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
            0
        }

        fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
            vortex_panic!("DelegatingCoreStorage buffer index {idx} out of bounds")
        }

        fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
            None
        }

        fn serialize(
            _array: ArrayView<'_, Self>,
            _session: &VortexSession,
        ) -> VortexResult<Option<Vec<u8>>> {
            Ok(Some(vec![]))
        }

        fn deserialize(
            &self,
            _dtype: &DType,
            _len: usize,
            _metadata: &[u8],
            _buffers: &[BufferHandle],
            _children: &dyn ArrayChildren,
            _session: &VortexSession,
        ) -> VortexResult<ArrayParts<Self>> {
            unreachable!("test-only vtable is not registered for serde")
        }

        fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
            match idx {
                0 => "delegated".to_string(),
                _ => vortex_panic!("DelegatingCoreStorage slot_name index {idx} out of bounds"),
            }
        }

        fn execute(
            array: Array<Self>,
            _ctx: &mut crate::ExecutionCtx,
        ) -> VortexResult<crate::ExecutionResult> {
            Ok(crate::ExecutionResult::done(array))
        }
    }

    impl ValidityChild<DelegatingCoreStorage> for DelegatingCoreStorage {
        fn validity_child(array: ArrayView<'_, DelegatingCoreStorage>) -> ArrayRef {
            array.array().slots()[0].as_ref().unwrap().clone()
        }
    }

    fn make_delegating_core_storage(delegated: ArrayRef) -> ArrayRef {
        Array::try_from_parts(
            ArrayParts::new(
                DelegatingCoreStorage,
                DType::Variant(Nullability::NonNullable),
                delegated.len(),
                EmptyArrayData,
            )
            .with_slots(vec![Some(delegated)]),
        )
        .unwrap()
        .into_array()
    }

    #[test]
    fn test_try_new_rejects_shredded_length_mismatch() {
        let core_storage = ConstantArray::new(Scalar::variant(Scalar::from(42i32)), 3).into_array();
        let shredded = buffer![1i32, 2].into_array();

        assert!(VariantArray::try_new(core_storage, Some(shredded)).is_err());
    }

    #[test]
    fn test_validate_rejects_shredded_metadata_mismatch() {
        let core_storage = ConstantArray::new(Scalar::variant(Scalar::from(42i32)), 3).into_array();
        let shredded = buffer![1i32, 2, 3].into_array();

        let result = Array::<super::Variant>::try_from_parts(
            ArrayParts::new(
                super::Variant,
                core_storage.dtype().clone(),
                core_storage.len(),
                VariantMetadata::None,
            )
            .with_slots(vec![Some(core_storage), Some(shredded)]),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_serde_roundtrip_preserves_shredded_child() {
        let core_storage = ConstantArray::new(Scalar::variant(Scalar::from(42i32)), 3).into_array();
        let shredded = buffer![1i32, 2, 3].into_array();
        let array = VariantArray::try_new(core_storage, Some(shredded.clone()))
            .unwrap()
            .into_array();

        let decoded = roundtrip(array.clone());

        assert!(array.array_eq(&decoded, Precision::Value));
        let decoded = decoded.as_opt::<super::Variant>().unwrap();
        assert!(
            decoded
                .shredded()
                .unwrap()
                .array_eq(&shredded, Precision::Value)
        );
        assert!(matches!(
            ArrayRef::clone(decoded.as_ref())
                .try_downcast::<super::Variant>()
                .unwrap()
                .into_data(),
            VariantMetadata::Inline { shredded_dtype } if shredded_dtype == shredded.dtype().clone()
        ));
    }

    #[test]
    fn test_try_new_derived_requires_delegated_child() {
        let core_storage = ConstantArray::new(Scalar::variant(Scalar::from(42i32)), 3).into_array();

        assert!(VariantArray::try_new_derived(core_storage, "missing").is_err());
    }

    #[test]
    fn test_validate_rejects_physical_slot_for_derived_shredded() {
        let delegated = ConstantArray::new(Scalar::variant(Scalar::from(7i32)), 3).into_array();
        let core_storage = Array::try_from_parts(
            ArrayParts::new(
                DelegatingCoreStorage,
                DType::Variant(Nullability::NonNullable),
                delegated.len(),
                EmptyArrayData,
            )
            .with_slots(vec![Some(delegated.clone())]),
        )
        .unwrap()
        .into_array();

        let result = Array::<super::Variant>::try_from_parts(
            ArrayParts::new(
                super::Variant,
                core_storage.dtype().clone(),
                core_storage.len(),
                VariantMetadata::Derived {
                    slot_name: "delegated".to_string(),
                },
            )
            .with_slots(vec![Some(core_storage), Some(delegated)]),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_serde_roundtrip_preserves_derived_shredded_delegation() {
        let inner_core_storage =
            ConstantArray::new(Scalar::variant(Scalar::from(42i32)), 3).into_array();
        let delegated_core_storage = VariantArray::try_new(inner_core_storage.clone(), None)
            .unwrap()
            .into_array();
        let array = VariantArray::try_new_derived(delegated_core_storage, "core_storage")
            .unwrap()
            .into_array();

        let decoded = roundtrip(array.clone());

        assert!(array.array_eq(&decoded, Precision::Value));
        let decoded = decoded.as_opt::<super::Variant>().unwrap();
        assert!(decoded.shredded_is_derived());
        assert_eq!(decoded.derived_shredded_slot_name(), Some("core_storage"));
        assert!(decoded.as_ref().slots()[1].is_none());
        assert!(
            decoded
                .shredded()
                .unwrap()
                .array_eq(&inner_core_storage, Precision::Value)
        );
    }

    #[test]
    fn test_derived_shredded_is_reconstructed_through_slice_wrapper() {
        let core_storage =
            make_delegating_core_storage(PrimitiveArray::from_iter(10i32..15).into_array());
        let sliced_core_storage = SliceArray::new(core_storage, 1..4).into_array();
        let array = VariantArray::try_new_derived(sliced_core_storage, "delegated")
            .unwrap()
            .into_array();

        let variant = array.as_opt::<super::Variant>().unwrap();
        assert_arrays_eq!(
            variant.shredded().unwrap(),
            PrimitiveArray::from_iter(11i32..14)
        );
    }

    #[test]
    fn test_derived_shredded_is_reconstructed_through_filter_wrapper() {
        let core_storage =
            make_delegating_core_storage(PrimitiveArray::from_iter(10i32..14).into_array());
        let filtered_core_storage =
            FilterArray::try_new(core_storage, Mask::from_iter([true, false, true, true]))
                .unwrap()
                .into_array();
        let array = VariantArray::try_new_derived(filtered_core_storage, "delegated")
            .unwrap()
            .into_array();

        let variant = array.as_opt::<super::Variant>().unwrap();
        assert_arrays_eq!(
            variant.shredded().unwrap(),
            PrimitiveArray::from_iter([10i32, 12, 13])
        );
    }

    #[test]
    fn test_derived_shredded_is_reconstructed_through_masked_wrapper() {
        let core_storage =
            make_delegating_core_storage(PrimitiveArray::from_iter(10i32..14).into_array());
        let masked_core_storage = MaskedArray::try_new(
            core_storage,
            Validity::from_iter([true, false, true, false]),
        )
        .unwrap()
        .into_array();
        let array = VariantArray::try_new_derived(masked_core_storage, "delegated")
            .unwrap()
            .into_array();

        let variant = array.as_opt::<super::Variant>().unwrap();
        let expected = MaskedArray::try_new(
            PrimitiveArray::from_iter(10i32..14).into_array(),
            Validity::from_iter([true, false, true, false]),
        )
        .unwrap()
        .into_array();
        assert_arrays_eq!(variant.shredded().unwrap(), expected);
    }

    #[test]
    fn test_derived_shredded_is_reconstructed_through_dict_wrapper() {
        let core_storage =
            make_delegating_core_storage(PrimitiveArray::from_iter([10i32, 20, 30]).into_array());
        let taken_core_storage = DictArray::try_new(
            PrimitiveArray::from_iter([2u64, 0, 1, 2]).into_array(),
            core_storage,
        )
        .unwrap()
        .into_array();
        let array = VariantArray::try_new_derived(taken_core_storage, "delegated")
            .unwrap()
            .into_array();

        let variant = array.as_opt::<super::Variant>().unwrap();
        assert_arrays_eq!(
            variant.shredded().unwrap(),
            PrimitiveArray::from_iter([30i32, 10, 20, 30])
        );
    }
}
