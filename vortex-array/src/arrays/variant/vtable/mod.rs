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
use crate::vtable;

vtable!(Variant, Variant, VariantData);

#[derive(Clone, Debug)]
pub struct Variant;

impl Variant {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.variant");
}

impl VTable for Variant {
    type ArrayData = VariantData;

    type OperationsVTable = Self;

    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn validate(&self, data: &Self::ArrayData, dtype: &DType, len: usize) -> VortexResult<()> {
        vortex_ensure!(
            matches!(dtype, DType::Variant(_)),
            "Expected Variant DType, got {dtype}"
        );
        vortex_ensure!(
            data.child().dtype() == dtype,
            "VariantArray child dtype {} does not match outer dtype {}",
            data.child().dtype(),
            dtype
        );
        vortex_ensure!(
            data.len() == len,
            "VariantArray length {} does not match outer length {}",
            data.len(),
            len
        );
        Ok(())
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

    fn serialize(_array: ArrayView<'_, Self>) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],

        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &vortex_session::VortexSession,
    ) -> VortexResult<Self::ArrayData> {
        vortex_ensure!(
            metadata.is_empty(),
            "VariantArray expects empty metadata, got {} bytes",
            metadata.len()
        );
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
    use crate::arrays::ConstantArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;

    #[test]
    fn with_slots_rejects_missing_child() {
        let child =
            ConstantArray::new(Scalar::null(DType::Variant(Nullability::Nullable)), 3).into_array();
        let array = VariantArray::new(child);
        let mut data = array.into_data();

        let err = <Variant as VTable>::with_slots(&mut data, vec![None]).unwrap_err();

        assert!(err.to_string().contains("child slot must be present"));
    }
}
