// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
use vortex_utils::aliases::hash_set::HashSet;

use super::Variant;
use super::VariantArray;
use super::VariantArrayExt;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::EmptyArrayData;
use crate::array::NotSupported;
use crate::array::VTable;
use crate::array::ValidityChild;
use crate::array::ValidityVTableFromChild;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::SliceArray;
use crate::assert_arrays_eq;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::normalize::NormalizeOptions;
use crate::normalize::Operation;
use crate::optimizer::ArrayOptimizer;
use crate::serde::ArrayChildren;

#[derive(Clone, Debug)]
struct DerivedCoreStorage;

impl VTable for DerivedCoreStorage {
    type ArrayData = EmptyArrayData;
    type OperationsVTable = NotSupported;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.variant.test.derived_core_storage");
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
        let typed_value = slots[0]
            .as_ref()
            .ok_or_else(|| vortex_error::vortex_err!("missing typed_value slot"))?;
        vortex_ensure!(
            typed_value.len() == len,
            "typed_value length {} does not match outer length {}",
            typed_value.len(),
            len
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_error::vortex_panic!("DerivedCoreStorage buffer index {idx} out of bounds")
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
        vortex_bail!("DerivedCoreStorage::deserialize is only used in tests")
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        match idx {
            0 => "typed_value".to_string(),
            _ => vortex_error::vortex_panic!(
                "DerivedCoreStorage slot_name index {idx} out of bounds"
            ),
        }
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }
}

impl ValidityChild<DerivedCoreStorage> for DerivedCoreStorage {
    fn validity_child(array: ArrayView<'_, DerivedCoreStorage>) -> ArrayRef {
        array.array().slots()[0].as_ref().unwrap().clone()
    }
}

fn make_derived_variant_with_sliced_shredded() -> VortexResult<ArrayRef> {
    let typed_value =
        SliceArray::new(PrimitiveArray::from_iter(10i32..20).into_array(), 2..6).into_array();
    let core_storage = Array::try_from_parts(
        ArrayParts::new(
            DerivedCoreStorage,
            DType::Variant(Nullability::NonNullable),
            typed_value.len(),
            EmptyArrayData,
        )
        .with_slots(vec![Some(typed_value)]),
    )?
    .into_array();

    Ok(VariantArray::try_new_derived(core_storage, "typed_value")?.into_array())
}

#[test]
fn optimize_recursive_rebuilds_derived_variant_from_core_storage() -> VortexResult<()> {
    let array = make_derived_variant_with_sliced_shredded()?;

    let optimized = array.optimize_recursive(&VortexSession::empty())?;
    let optimized = optimized.as_opt::<Variant>().unwrap();

    assert!(optimized.shredded_is_derived());
    assert_arrays_eq!(
        optimized.shredded().unwrap(),
        PrimitiveArray::from_iter(12i32..16)
    );
    assert!(ArrayRef::ptr_eq(
        &optimized.shredded().unwrap(),
        optimized.core_storage().slots()[0].as_ref().unwrap(),
    ));

    Ok(())
}

#[test]
fn normalize_with_execution_rebuilds_derived_variant_from_core_storage() -> VortexResult<()> {
    let array = make_derived_variant_with_sliced_shredded()?;
    let allowed =
        HashSet::from_iter([array.encoding_id(), Primitive.id(), DerivedCoreStorage.id()]);
    let mut ctx = ExecutionCtx::new(VortexSession::empty());

    let normalized = array.normalize(&mut NormalizeOptions {
        allowed: &allowed,
        operation: Operation::Execute(&mut ctx),
    })?;
    let normalized = normalized.as_opt::<Variant>().unwrap();

    assert!(normalized.shredded_is_derived());
    assert_arrays_eq!(
        normalized.shredded().unwrap(),
        PrimitiveArray::from_iter(12i32..16)
    );
    assert!(ArrayRef::ptr_eq(
        &normalized.shredded().unwrap(),
        normalized.core_storage().slots()[0].as_ref().unwrap(),
    ));

    Ok(())
}
