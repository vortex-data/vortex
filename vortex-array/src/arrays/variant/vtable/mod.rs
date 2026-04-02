// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod operations;
mod validity;

use std::hash::Hasher;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::EmptyMetadata;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::Precision;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::variant::NUM_SLOTS;
use crate::arrays::variant::SLOT_NAMES;
use crate::arrays::variant::VariantData;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::serde::ArrayChildren;
use crate::stats::ArrayStats;
use crate::vtable;

vtable!(Variant, Variant, VariantData);

#[derive(Clone, Debug)]
pub struct Variant;

impl Variant {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.variant");
}

impl VTable for Variant {
    type ArrayData = VariantData;

    type Metadata = EmptyMetadata;

    type OperationsVTable = Self;

    type ValidityVTable = Self;

    fn vtable(_array: &Self::ArrayData) -> &Self {
        &Variant
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &Self::ArrayData) -> usize {
        array.child().len()
    }

    fn dtype(array: &Self::ArrayData) -> &DType {
        array.child().dtype()
    }

    fn stats(array: &Self::ArrayData) -> &ArrayStats {
        &array.stats_set
    }

    fn array_hash<H: Hasher>(array: &VariantData, state: &mut H, precision: Precision) {
        array.child().array_hash(state, precision);
    }

    fn array_eq(array: &VariantData, other: &VariantData, precision: Precision) -> bool {
        array.child().array_eq(other.child(), precision)
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

    fn metadata(_array: ArrayView<'_, Self>) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        _bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &vortex_session::VortexSession,
    ) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn build(
        dtype: &DType,
        len: usize,
        _metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<Self::ArrayData> {
        vortex_ensure!(matches!(dtype, DType::Variant(_)), "Expected Variant DType");
        vortex_ensure!(
            children.len() == 1,
            "Expected 1 child, got {}",
            children.len()
        );
        // The child carries the nullability for the whole VariantArray.
        let child = children.get(0, dtype, len)?;
        Ok(VariantData::new(child))
    }

    fn slots(array: ArrayView<'_, Self>) -> &[Option<ArrayRef>] {
        &array.data().slots
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        match SLOT_NAMES.get(idx) {
            Some(name) => (*name).to_string(),
            None => vortex_panic!("VariantArray slot_name index {idx} out of bounds"),
        }
    }

    fn with_slots(array: &mut Self::ArrayData, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "VariantArray expects exactly {} slot, got {}",
            NUM_SLOTS,
            slots.len()
        );
        vortex_ensure!(
            slots[0].is_some(),
            "VariantArray child slot must be present"
        );
        array.slots = slots;
        Ok(())
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;

    #[test]
    fn with_slots_rejects_missing_child() {
        let array = VariantArray::new(PrimitiveArray::from_iter([1u8, 2, 3]).into_array());
        let mut data = array.into_data();

        let err = <Variant as VTable>::with_slots(&mut data, vec![None]).unwrap_err();

        assert!(err.to_string().contains("child slot must be present"));
    }
}
