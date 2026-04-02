// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::EmptyMetadata;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::Precision;
use crate::arrays::FixedSizeListArray;
use crate::arrays::fixed_size_list::array::NUM_SLOTS;
use crate::arrays::fixed_size_list::array::SLOT_NAMES;
use crate::arrays::fixed_size_list::array::VALIDITY_SLOT;
use crate::arrays::fixed_size_list::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::serde::ArrayChildren;
use crate::stats::StatsSetRef;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::Array;
use crate::vtable::ArrayId;
use crate::vtable::VTable;
use crate::vtable::ValidityVTableFromValidityHelper;
mod kernel;
mod operations;
mod validity;

vtable!(FixedSizeList);

#[derive(Clone, Debug)]
pub struct FixedSizeList;

impl FixedSizeList {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.fixed_size_list");
}

impl VTable for FixedSizeList {
    type Array = FixedSizeListArray;

    type Metadata = EmptyMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    fn vtable(_array: &Self::Array) -> &Self {
        &FixedSizeList
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &FixedSizeListArray) -> usize {
        array.len
    }

    fn dtype(array: &FixedSizeListArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &FixedSizeListArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &FixedSizeListArray,
        state: &mut H,
        precision: Precision,
    ) {
        array.dtype.hash(state);
        array.elements().array_hash(state, precision);
        array.list_size().hash(state);
        array.validity.array_hash(state, precision);
        array.len.hash(state);
    }

    fn array_eq(
        array: &FixedSizeListArray,
        other: &FixedSizeListArray,
        precision: Precision,
    ) -> bool {
        array.dtype == other.dtype
            && array.elements().array_eq(other.elements(), precision)
            && array.list_size() == other.list_size()
            && array.validity.array_eq(&other.validity, precision)
            && array.len == other.len
    }

    fn nbuffers(_array: &FixedSizeListArray) -> usize {
        0
    }

    fn buffer(_array: &FixedSizeListArray, idx: usize) -> BufferHandle {
        vortex_panic!("FixedSizeListArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &FixedSizeListArray, idx: usize) -> Option<String> {
        vortex_panic!("FixedSizeListArray buffer_name index {idx} out of bounds")
    }

    fn reduce_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        Self::PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn metadata(_array: &FixedSizeListArray) -> VortexResult<Self::Metadata> {
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
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    /// Builds a [`FixedSizeListArray`].
    ///
    /// This method expects 1 or 2 children (a second child indicates a validity array).
    fn build(
        dtype: &DType,
        len: usize,
        _metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<FixedSizeListArray> {
        vortex_ensure!(
            buffers.is_empty(),
            "`FixedSizeList::build` expects no buffers"
        );

        let DType::FixedSizeList(element_dtype, list_size, _) = &dtype else {
            vortex_bail!("Expected `DType::FixedSizeList`, got {:?}", dtype);
        };

        let validity = {
            if children.len() > 2 {
                vortex_bail!("`FixedSizeList::build` method expected 1 or 2 children")
            }

            if children.len() == 2 {
                let validity = children.get(1, &Validity::DTYPE, len)?;
                Validity::Array(validity)
            } else {
                debug_assert_eq!(children.len(), 1);
                Validity::from(dtype.nullability())
            }
        };

        let num_elements = len * (*list_size as usize);
        let elements = children.get(0, element_dtype.as_ref(), num_elements)?;

        FixedSizeListArray::try_new(elements, *list_size, validity, len)
    }

    fn slots(array: &FixedSizeListArray) -> &[Option<ArrayRef>] {
        &array.slots
    }

    fn slot_name(_array: &FixedSizeListArray, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn with_slots(
        array: &mut FixedSizeListArray,
        slots: Vec<Option<ArrayRef>>,
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "FixedSizeListArray expects exactly {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );
        array.validity = match &slots[VALIDITY_SLOT] {
            Some(arr) => Validity::Array(arr.clone()),
            None => Validity::from(array.dtype.nullability()),
        };
        array.slots = slots;
        Ok(())
    }

    fn execute(array: Arc<Array<Self>>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }
}
