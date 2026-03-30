// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod operations;
mod validity;

use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::EmptyMetadata;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::Precision;
use crate::arrays::VariantArray;
use crate::arrays::variant::NUM_SLOTS;
use crate::arrays::variant::SLOT_NAMES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::serde::ArrayChildren;
use crate::stats::StatsSetRef;
use crate::vtable;
use crate::vtable::Array;
use crate::vtable::ArrayId;
use crate::vtable::VTable;

vtable!(Variant);

#[derive(Clone, Debug)]
pub struct Variant;

impl Variant {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.variant");
}

impl VTable for Variant {
    type Array = VariantArray;

    type Metadata = EmptyMetadata;

    type OperationsVTable = Self;

    type ValidityVTable = Self;

    fn vtable(_array: &Self::Array) -> &Self {
        &Variant
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &Self::Array) -> usize {
        array.child().len()
    }

    fn dtype(array: &Self::Array) -> &DType {
        array.child().dtype()
    }

    fn stats(array: &Self::Array) -> StatsSetRef<'_> {
        array.child().statistics()
    }

    fn array_hash<H: Hasher>(array: &Self::Array, state: &mut H, precision: Precision) {
        array.child().array_hash(state, precision);
    }

    fn array_eq(array: &Self::Array, other: &Self::Array, precision: Precision) -> bool {
        array.child().array_eq(other.child(), precision)
    }

    fn nbuffers(_array: &Self::Array) -> usize {
        0
    }

    fn buffer(_array: &Self::Array, idx: usize) -> BufferHandle {
        vortex_panic!("VariantArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &Self::Array, _idx: usize) -> Option<String> {
        None
    }

    fn slots(array: &Self::Array) -> &[Option<ArrayRef>] {
        &array.slots
    }

    fn slot_name(_array: &Self::Array, idx: usize) -> String {
        match SLOT_NAMES.get(idx) {
            Some(name) => (*name).to_string(),
            None => vortex_panic!("VariantArray slot_name index {idx} out of bounds"),
        }
    }

    fn metadata(_array: &Self::Array) -> VortexResult<Self::Metadata> {
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
    ) -> VortexResult<Self::Array> {
        vortex_ensure!(matches!(dtype, DType::Variant(_)), "Expected Variant DType");
        vortex_ensure!(
            children.len() == 1,
            "Expected 1 child, got {}",
            children.len()
        );
        // The child carries the nullability for the whole VariantArray.
        let child = children.get(0, dtype, len)?;
        Ok(VariantArray::new(child))
    }

    fn with_slots(array: &mut Self::Array, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "VariantArray expects exactly {} slot, got {}",
            NUM_SLOTS,
            slots.len()
        );
        let child = slots
            .into_iter()
            .next()
            .vortex_expect("VariantArray slot vector length was validated")
            .ok_or_else(|| vortex_err!("VariantArray child slot must be present"))?;
        array.slots = [Some(child)];
        Ok(())
    }

    fn execute(array: Arc<Array<Self>>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
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
        let mut array = VariantArray::new(PrimitiveArray::from_iter([1u8, 2, 3]).into_array());

        let err = <Variant as VTable>::with_slots(&mut array, vec![None]).unwrap_err();

        assert!(err.to_string().contains("child slot must be present"));
    }
}
