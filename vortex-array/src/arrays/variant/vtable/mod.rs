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
use crate::arrays::variant::vtable::variant_metadata_proto::Shredded;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::serde::ArrayChildren;

/// A canonical Vortex array of [`DType::Variant`] values.
///
/// `VariantArray` is Vortex's stable in-memory boundary for variant values. It has two logical
/// children:
/// - slot 0 is mandatory [`core_storage`][crate::arrays::variant::VariantArrayExt::core_storage]
/// - slot 1 is optional [`shredded`][crate::arrays::variant::VariantArrayExt::shredded]
///
/// `core_storage` owns variant encoding semantics. Outer [`DType`], array length, scalar
/// extraction, and validity are all derived from it.
///
/// `shredded` exposes a more concrete same-length child when one is available. It may be stored:
/// - inline, as physical slot 1
/// - derived, by delegation to an encoding-qualified local slot inside `core_storage`
///
/// Physical storage stays simple:
/// - slot 0 always stores `core_storage`
/// - slot 1 is populated only for inline `shredded`
/// - derived `shredded` is accessor-only and is reconstructed from logical `core_storage`
///
/// Delegated `shredded` lookup is defined over logical `core_storage`, not only its top-level
/// slots. The delegated slot name is local to the recorded source encoding ID; this avoids
/// treating names such as `validity`, `values`, or `child` as globally unique across unrelated
/// encodings. Lookup can pass through row-preserving wrappers and nested canonical
/// [`VariantArray`] boundaries produced by execution or normalization, but it stops at the first
/// non-transparent array with the recorded source encoding ID.
///
/// During recursive canonicalization, `core_storage` is preserved as-is. Inline `shredded`
/// children are canonicalized independently; derived `shredded` children remain delegated from
/// `core_storage`.
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
    pub shredded: Option<Shredded>,
}

/// Serialized reference to a derived `shredded` child.
///
/// `slot_name` is local to `source_encoding_id`; it is not a global child name.
#[derive(Clone, prost::Message)]
struct DerivedSlotProto {
    #[prost(string, tag = "1")]
    pub source_encoding_id: String,
    #[prost(string, tag = "2")]
    pub slot_name: String,
}

mod variant_metadata_proto {
    use prost::Oneof;
    use vortex_proto::dtype as pb;

    #[derive(Clone, Oneof)]
    pub enum Shredded {
        #[prost(message, tag = "1")]
        InlineDtype(pb::DType),
        #[prost(message, tag = "2")]
        DerivedSlot(super::DerivedSlotProto),
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
        vortex_ensure!(
            slots[CORE_STORAGE_SLOT].is_some(),
            "VariantArray core_storage slot must be present"
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
            VariantMetadata::Derived {
                source_encoding_id,
                slot_name,
            } => {
                vortex_ensure!(
                    slots[SHREDDED_SLOT].is_none(),
                    "VariantArray derived shredded child must not populate physical slot {SHREDDED_SLOT}"
                );
                let derived_shredded = try_derived_shredded_from_core_storage(
                    core_storage,
                    *source_encoding_id,
                    slot_name,
                )?
                    .ok_or_else(|| vortex_error::vortex_err!(
                        "VariantArray derived shredded slot {source_encoding_id}.{slot_name} is not exposed by core_storage"
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
                shredded: Some(Shredded::InlineDtype(shredded_dtype.try_into()?)),
            }
            .encode_to_vec(),
            VariantMetadata::Derived {
                source_encoding_id,
                slot_name,
            } => VariantMetadataProto {
                shredded: Some(Shredded::DerivedSlot(DerivedSlotProto {
                    source_encoding_id: source_encoding_id.as_ref().to_string(),
                    slot_name: slot_name.clone(),
                })),
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
                Some(Shredded::InlineDtype(dtype)) => VariantMetadata::Inline {
                    shredded_dtype: DType::from_proto(&dtype, session)?,
                },
                Some(Shredded::DerivedSlot(slot)) => VariantMetadata::Derived {
                    source_encoding_id: ArrayId::from(slot.source_encoding_id.as_str()),
                    slot_name: slot.slot_name,
                },
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
    use std::fmt::Display;
    use std::fmt::Formatter;
    use std::hash::Hash;
    use std::hash::Hasher;

    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_mask::Mask;
    use vortex_session::registry::ReadContext;

    use super::*;
    use crate::ArrayContext;
    use crate::ArrayEq;
    use crate::ArrayHash;
    use crate::EmptyArrayData;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::NotSupported;
    use crate::Precision;
    use crate::ValidityChild;
    use crate::ValidityVTableFromChild;
    use crate::arrays::ConstantArray;
    use crate::arrays::Dict;
    use crate::arrays::DictArray;
    use crate::arrays::Filter;
    use crate::arrays::FilterArray;
    use crate::arrays::Masked;
    use crate::arrays::MaskedArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::SliceArray;
    use crate::arrays::variant::VariantArrayExt;
    use crate::arrays::variant::rebuild_variant_array;
    use crate::assert_arrays_eq;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::optimizer::ArrayOptimizer;
    use crate::scalar::Scalar;
    use crate::serde::SerializeOptions;
    use crate::serde::SerializedArray;
    use crate::session::ArraySession;
    use crate::session::ArraySessionExt;
    use crate::validity::Validity;

    fn roundtrip_with_session(array: ArrayRef, session: &VortexSession) -> ArrayRef {
        let dtype = array.dtype().clone();
        let len = array.len();

        let ctx = ArrayContext::empty();
        let serialized = array
            .serialize(&ctx, session, &SerializeOptions::default())
            .unwrap();

        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let parts = SerializedArray::try_from(concat.freeze()).unwrap();
        parts
            .decode(&dtype, len, &ReadContext::new(ctx.to_ids()), session)
            .unwrap()
    }

    fn roundtrip(array: ArrayRef) -> ArrayRef {
        roundtrip_with_session(array, &LEGACY_SESSION)
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
            dtype: &DType,
            len: usize,
            _metadata: &[u8],
            _buffers: &[BufferHandle],
            children: &dyn ArrayChildren,
            _session: &VortexSession,
        ) -> VortexResult<ArrayParts<Self>> {
            let delegated = children.get(
                0,
                &DType::Primitive(PType::I32, Nullability::NonNullable),
                len,
            )?;
            Ok(ArrayParts::new(Self, dtype.clone(), len, EmptyArrayData)
                .with_slots(vec![Some(delegated)]))
        }

        fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
            match idx {
                0 => "delegated".to_string(),
                _ => vortex_panic!("DelegatingCoreStorage slot_name index {idx} out of bounds"),
            }
        }

        fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
            Ok(ExecutionResult::done(array))
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

    fn make_derived_variant(delegated: ArrayRef) -> VortexResult<ArrayRef> {
        let core_storage = make_delegating_core_storage(delegated);
        Ok(
            VariantArray::try_new_derived(core_storage, DelegatingCoreStorage.id(), "delegated")?
                .into_array(),
        )
    }

    #[derive(Clone, Debug)]
    struct NamedSlotCoreStorage;

    #[derive(Clone, Debug)]
    struct NamedSlotCoreStorageData {
        slot_name: &'static str,
    }

    impl ArrayEq for NamedSlotCoreStorageData {
        fn array_eq(&self, other: &Self, _precision: Precision) -> bool {
            self.slot_name == other.slot_name
        }
    }

    impl ArrayHash for NamedSlotCoreStorageData {
        fn array_hash<H: Hasher>(&self, state: &mut H, _precision: Precision) {
            self.slot_name.hash(state);
        }
    }

    impl Display for NamedSlotCoreStorageData {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "slot_name: {}", self.slot_name)
        }
    }

    impl VTable for NamedSlotCoreStorage {
        type ArrayData = NamedSlotCoreStorageData;
        type OperationsVTable = NotSupported;
        type ValidityVTable = ValidityVTableFromChild;

        fn id(&self) -> ArrayId {
            static ID: CachedId = CachedId::new("vortex.variant.test.named_slot_core_storage");
            *ID
        }

        fn validate(
            &self,
            data: &NamedSlotCoreStorageData,
            dtype: &DType,
            len: usize,
            slots: &[Option<ArrayRef>],
        ) -> VortexResult<()> {
            vortex_ensure!(
                matches!(dtype, DType::Variant(_)),
                "expected variant dtype, found {dtype}"
            );
            vortex_ensure!(slots.len() == 1, "expected 1 slot, got {}", slots.len());
            let slot = slots[0]
                .as_ref()
                .ok_or_else(|| vortex_error::vortex_err!("missing {} slot", data.slot_name))?;
            vortex_ensure!(
                slot.len() == len,
                "{} length {} does not match outer length {}",
                data.slot_name,
                slot.len(),
                len
            );
            Ok(())
        }

        fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
            0
        }

        fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
            vortex_panic!("NamedSlotCoreStorage buffer index {idx} out of bounds")
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

        fn slot_name(array: ArrayView<'_, Self>, idx: usize) -> String {
            match idx {
                0 => array.data().slot_name.to_string(),
                _ => vortex_panic!("NamedSlotCoreStorage slot_name index {idx} out of bounds"),
            }
        }

        fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
            Ok(ExecutionResult::done(array))
        }
    }

    impl ValidityChild<NamedSlotCoreStorage> for NamedSlotCoreStorage {
        fn validity_child(array: ArrayView<'_, NamedSlotCoreStorage>) -> ArrayRef {
            array.array().slots()[0].as_ref().unwrap().clone()
        }
    }

    fn make_named_slot_core_storage(slot: ArrayRef, slot_name: &'static str) -> ArrayRef {
        Array::try_from_parts(
            ArrayParts::new(
                NamedSlotCoreStorage,
                DType::Variant(Nullability::NonNullable),
                slot.len(),
                NamedSlotCoreStorageData { slot_name },
            )
            .with_slots(vec![Some(slot)]),
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

        let result = VariantArray::try_from_parts(
            ArrayParts::new(
                Variant,
                core_storage.dtype().clone(),
                core_storage.len(),
                VariantMetadata::None,
            )
            .with_slots(vec![Some(core_storage), Some(shredded)]),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_validate_rejects_missing_core_storage_slot() {
        let result = VariantArray::try_from_parts(
            ArrayParts::new(
                Variant,
                DType::Variant(Nullability::NonNullable),
                3,
                VariantMetadata::None,
            )
            .with_slots(vec![None, None]),
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
        let decoded = decoded.as_opt::<Variant>().unwrap();
        assert!(
            decoded
                .shredded()
                .unwrap()
                .array_eq(&shredded, Precision::Value)
        );
        assert!(matches!(
            ArrayRef::clone(decoded.as_ref())
                .try_downcast::<Variant>()
                .unwrap()
                .into_data(),
            VariantMetadata::Inline { shredded_dtype } if shredded_dtype == shredded.dtype().clone()
        ));
    }

    #[test]
    fn test_try_new_derived_requires_delegated_child() {
        let core_storage = ConstantArray::new(Scalar::variant(Scalar::from(42i32)), 3).into_array();
        let source_encoding_id = core_storage.encoding_id();

        assert!(
            VariantArray::try_new_derived(core_storage, source_encoding_id, "missing").is_err()
        );
    }

    #[test]
    fn test_try_new_derived_rejects_absent_source_encoding_id() {
        let core_storage =
            make_delegating_core_storage(PrimitiveArray::from_iter(10i32..13).into_array());

        let error = VariantArray::try_new_derived(core_storage, NamedSlotCoreStorage.id(), "child")
            .unwrap_err();

        assert!(error.to_string().contains("derived shredded slot"));
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

        let result = VariantArray::try_from_parts(
            ArrayParts::new(
                Variant,
                core_storage.dtype().clone(),
                core_storage.len(),
                VariantMetadata::Derived {
                    source_encoding_id: DelegatingCoreStorage.id(),
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
        let array =
            VariantArray::try_new_derived(delegated_core_storage, Variant.id(), "core_storage")
                .unwrap()
                .into_array();

        let decoded = roundtrip(array.clone());

        assert!(array.array_eq(&decoded, Precision::Value));
        let decoded = decoded.as_opt::<Variant>().unwrap();
        assert!(decoded.shredded_is_derived());
        assert_eq!(
            decoded.derived_shredded_source(),
            Some((Variant.id(), "core_storage"))
        );
        assert!(decoded.as_ref().slots()[1].is_none());
        assert!(
            decoded
                .shredded()
                .unwrap()
                .array_eq(&inner_core_storage, Precision::Value)
        );
    }

    #[test]
    fn test_serde_roundtrip_preserves_derived_shredded_delegation_from_source_encoding()
    -> VortexResult<()> {
        let session = VortexSession::empty().with::<ArraySession>();
        session.arrays().register(DelegatingCoreStorage);

        let delegated = PrimitiveArray::from_iter(10i32..13).into_array();
        let core_storage = make_delegating_core_storage(delegated.clone());
        let array =
            VariantArray::try_new_derived(core_storage, DelegatingCoreStorage.id(), "delegated")?
                .into_array();

        let decoded = roundtrip_with_session(array.clone(), &session);

        assert!(array.array_eq(&decoded, Precision::Value));
        let decoded = decoded.as_opt::<Variant>().unwrap();
        assert!(decoded.shredded_is_derived());
        assert_eq!(
            decoded.derived_shredded_source(),
            Some((DelegatingCoreStorage.id(), "delegated"))
        );
        assert!(decoded.as_ref().slots()[1].is_none());
        assert!(decoded.core_storage().is::<DelegatingCoreStorage>());
        assert_arrays_eq!(decoded.shredded().unwrap(), delegated);

        Ok(())
    }

    #[test]
    fn test_rebuild_adopts_nested_inline_shredded_source() -> VortexResult<()> {
        let original_core_storage =
            make_delegating_core_storage(PrimitiveArray::from_iter(10i32..13).into_array());
        let original = VariantArray::try_new_derived(
            original_core_storage,
            DelegatingCoreStorage.id(),
            "delegated",
        )?;

        let nested_core_storage =
            ConstantArray::new(Scalar::variant(Scalar::from(42i32)), 3).into_array();
        let inline_shredded = PrimitiveArray::from_iter(20i32..23).into_array();
        let transformed_core_storage =
            VariantArray::try_new(nested_core_storage, Some(inline_shredded.clone()))?.into_array();

        let rebuilt = rebuild_variant_array(&original, transformed_core_storage, || {
            unreachable!("derived variants rebuild from core_storage")
        })?;

        assert_eq!(
            rebuilt.derived_shredded_source(),
            Some((Variant.id(), "shredded"))
        );
        assert_arrays_eq!(rebuilt.shredded().unwrap(), inline_shredded);

        Ok(())
    }

    #[test]
    fn test_execute_preserves_derived_metadata_identity() -> VortexResult<()> {
        let shredded = PrimitiveArray::from_iter(10i32..13).into_array();
        let core_storage = make_delegating_core_storage(shredded.clone());
        let array =
            VariantArray::try_new_derived(core_storage, DelegatingCoreStorage.id(), "delegated")?
                .into_array();
        let mut ctx = ExecutionCtx::new(VortexSession::empty());

        let executed = array.clone().execute::<ArrayRef>(&mut ctx)?;
        let executed_variant = executed.as_opt::<Variant>().unwrap();

        assert!(ArrayRef::ptr_eq(&executed, &array));
        assert!(executed_variant.shredded_is_derived());
        assert_eq!(
            executed_variant.derived_shredded_source(),
            Some((DelegatingCoreStorage.id(), "delegated"))
        );
        assert!(executed_variant.as_ref().slots()[1].is_none());
        assert_arrays_eq!(executed_variant.shredded().unwrap(), shredded);

        Ok(())
    }

    #[test]
    fn test_derived_shredded_is_reconstructed_through_slice_wrapper() {
        let core_storage =
            make_delegating_core_storage(PrimitiveArray::from_iter(10i32..15).into_array());
        let sliced_core_storage = SliceArray::new(core_storage, 1..4).into_array();
        let array = VariantArray::try_new_derived(
            sliced_core_storage,
            DelegatingCoreStorage.id(),
            "delegated",
        )
        .unwrap()
        .into_array();

        let variant = array.as_opt::<Variant>().unwrap();
        assert_arrays_eq!(
            variant.shredded().unwrap(),
            PrimitiveArray::from_iter(11i32..14)
        );
    }

    #[test]
    fn test_derived_shredded_is_reconstructed_through_chained_nested_wrapper() -> VortexResult<()> {
        let core_storage =
            make_delegating_core_storage(PrimitiveArray::from_iter(10i32..16).into_array());
        let filtered_core_storage = FilterArray::try_new(
            core_storage,
            Mask::from_iter([true, false, true, true, false, true]),
        )?
        .into_array();
        let nested_variant = VariantArray::try_new_derived(
            filtered_core_storage,
            DelegatingCoreStorage.id(),
            "delegated",
        )?
        .into_array();
        let sliced_nested_variant = SliceArray::new(nested_variant, 1..4).into_array();
        let array = VariantArray::try_new_derived(
            sliced_nested_variant,
            DelegatingCoreStorage.id(),
            "delegated",
        )?
        .into_array();

        let variant = array.as_opt::<Variant>().unwrap();
        let shredded = variant.shredded().unwrap();
        assert_eq!(shredded.len(), variant.as_ref().len());
        assert_arrays_eq!(shredded, PrimitiveArray::from_iter([12i32, 13, 15]));

        Ok(())
    }

    #[test]
    fn test_derived_child_slot_is_reconstructed_through_slice_wrapper() {
        let core_storage = make_named_slot_core_storage(
            PrimitiveArray::from_iter(10i32..15).into_array(),
            "child",
        );
        let sliced_core_storage = SliceArray::new(core_storage, 1..4).into_array();
        let array =
            VariantArray::try_new_derived(sliced_core_storage, NamedSlotCoreStorage.id(), "child")
                .unwrap()
                .into_array();

        let variant = array.as_opt::<Variant>().unwrap();
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
        let array = VariantArray::try_new_derived(
            filtered_core_storage,
            DelegatingCoreStorage.id(),
            "delegated",
        )
        .unwrap()
        .into_array();

        let variant = array.as_opt::<Variant>().unwrap();
        assert_arrays_eq!(
            variant.shredded().unwrap(),
            PrimitiveArray::from_iter([10i32, 12, 13])
        );
    }

    #[test]
    fn test_derived_child_slot_is_reconstructed_through_filter_wrapper() {
        let core_storage = make_named_slot_core_storage(
            PrimitiveArray::from_iter(10i32..14).into_array(),
            "child",
        );
        let filtered_core_storage =
            FilterArray::try_new(core_storage, Mask::from_iter([true, false, true, true]))
                .unwrap()
                .into_array();
        let array = VariantArray::try_new_derived(
            filtered_core_storage,
            NamedSlotCoreStorage.id(),
            "child",
        )
        .unwrap()
        .into_array();

        let variant = array.as_opt::<Variant>().unwrap();
        assert_arrays_eq!(
            variant.shredded().unwrap(),
            PrimitiveArray::from_iter([10i32, 12, 13])
        );
    }

    #[test]
    fn test_derived_shredded_through_masked_wrapper_preserves_child_nulls() {
        let core_storage = make_delegating_core_storage(
            PrimitiveArray::from_option_iter([Some(10i32), None, Some(12), Some(13)]).into_array(),
        );
        let masked_core_storage =
            MaskedArray::try_new(core_storage, Validity::from_iter([true, true, false, true]))
                .unwrap()
                .into_array();
        let array = VariantArray::try_new_derived(
            masked_core_storage,
            DelegatingCoreStorage.id(),
            "delegated",
        )
        .unwrap()
        .into_array();

        let variant = array.as_opt::<Variant>().unwrap();
        assert_arrays_eq!(
            variant.shredded().unwrap(),
            PrimitiveArray::from_option_iter([Some(10i32), None, None, Some(13)])
        );
    }

    #[test]
    fn test_masked_execute_all_valid_derived_variant_reuses_all_valid_core_storage()
    -> VortexResult<()> {
        let delegated = PrimitiveArray::from_iter(10i32..13).into_array();
        let core_storage = MaskedArray::try_new(
            make_delegating_core_storage(delegated.clone()),
            Validity::AllValid,
        )?
        .into_array();
        let variant = VariantArray::try_new_derived(
            core_storage.clone(),
            DelegatingCoreStorage.id(),
            "delegated",
        )?
        .into_array();
        let masked = MaskedArray::try_new(variant, Validity::AllValid)?.into_array();
        let mut ctx = ExecutionCtx::new(VortexSession::empty());

        let result = masked.execute::<ArrayRef>(&mut ctx)?;
        let result = result.as_opt::<Variant>().unwrap();

        assert!(ArrayRef::ptr_eq(result.core_storage(), &core_storage));
        assert_arrays_eq!(
            result.shredded().unwrap(),
            MaskedArray::try_new(delegated, Validity::AllValid)?.into_array()
        );

        Ok(())
    }

    #[test]
    fn test_derived_validity_slot_is_reconstructed_through_masked_wrapper() {
        let core_storage = make_named_slot_core_storage(
            PrimitiveArray::from_option_iter([Some(10i32), None, Some(12), Some(13)]).into_array(),
            "validity",
        );
        let masked_core_storage =
            MaskedArray::try_new(core_storage, Validity::from_iter([true, true, false, true]))
                .unwrap()
                .into_array();
        let array = VariantArray::try_new_derived(
            masked_core_storage,
            NamedSlotCoreStorage.id(),
            "validity",
        )
        .unwrap()
        .into_array();

        let variant = array.as_opt::<Variant>().unwrap();
        assert_arrays_eq!(
            variant.shredded().unwrap(),
            PrimitiveArray::from_option_iter([Some(10i32), None, None, Some(13)])
        );
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
        let array = VariantArray::try_new_derived(
            taken_core_storage,
            DelegatingCoreStorage.id(),
            "delegated",
        )
        .unwrap()
        .into_array();

        let variant = array.as_opt::<Variant>().unwrap();
        assert_arrays_eq!(
            variant.shredded().unwrap(),
            PrimitiveArray::from_iter([30i32, 10, 20, 30])
        );
    }

    #[test]
    fn test_derived_values_slot_is_reconstructed_through_dict_wrapper() {
        let core_storage = make_named_slot_core_storage(
            PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
            "values",
        );
        let taken_core_storage = DictArray::try_new(
            PrimitiveArray::from_iter([2u64, 0, 1, 2]).into_array(),
            core_storage,
        )
        .unwrap()
        .into_array();
        let array =
            VariantArray::try_new_derived(taken_core_storage, NamedSlotCoreStorage.id(), "values")
                .unwrap()
                .into_array();

        let variant = array.as_opt::<Variant>().unwrap();
        assert_arrays_eq!(
            variant.shredded().unwrap(),
            PrimitiveArray::from_iter([30i32, 10, 20, 30])
        );
    }

    #[test]
    fn test_filter_parent_reduce_preserves_derived_shredded_alignment() -> VortexResult<()> {
        let variant = make_derived_variant(PrimitiveArray::from_iter(10i32..14).into_array())?;
        let wrapped =
            FilterArray::try_new(variant, Mask::from_iter([true, false, true, true]))?.into_array();

        let optimized = wrapped.optimize()?;
        let optimized = optimized.as_opt::<Variant>().unwrap();

        assert!(optimized.core_storage().is::<Filter>());
        assert_eq!(optimized.shredded().unwrap().len(), optimized.len());
        assert_arrays_eq!(
            optimized.shredded().unwrap(),
            PrimitiveArray::from_iter([10i32, 12, 13])
        );

        Ok(())
    }

    #[test]
    fn test_take_parent_reduce_preserves_derived_shredded_alignment() -> VortexResult<()> {
        let variant =
            make_derived_variant(PrimitiveArray::from_iter([10i32, 20, 30]).into_array())?;
        let wrapped = DictArray::try_new(
            PrimitiveArray::from_iter([2u64, 0, 1, 2]).into_array(),
            variant,
        )?
        .into_array();

        let optimized = wrapped.optimize()?;
        let optimized = optimized.as_opt::<Variant>().unwrap();

        assert!(optimized.core_storage().is::<Dict>());
        assert_eq!(optimized.shredded().unwrap().len(), optimized.len());
        assert_arrays_eq!(
            optimized.shredded().unwrap(),
            PrimitiveArray::from_iter([30i32, 10, 20, 30])
        );

        Ok(())
    }

    #[test]
    fn test_masked_parent_reduce_preserves_derived_shredded_alignment() -> VortexResult<()> {
        let variant = make_derived_variant(PrimitiveArray::from_iter(10i32..14).into_array())?;
        let validity = Validity::from_iter([true, false, true, false]);
        let wrapped = MaskedArray::try_new(variant, validity.clone())?.into_array();

        let optimized = wrapped.optimize()?;
        let optimized = optimized.as_opt::<Variant>().unwrap();
        let expected =
            MaskedArray::try_new(PrimitiveArray::from_iter(10i32..14).into_array(), validity)?
                .into_array();

        assert!(optimized.core_storage().is::<Masked>());
        assert_eq!(optimized.shredded().unwrap().len(), optimized.len());
        assert_arrays_eq!(optimized.shredded().unwrap(), expected);

        Ok(())
    }
}
