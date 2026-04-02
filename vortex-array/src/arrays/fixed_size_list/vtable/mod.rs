// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

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
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::fixed_size_list::FixedSizeListData;
use crate::arrays::fixed_size_list::array::NUM_SLOTS;
use crate::arrays::fixed_size_list::array::SLOT_NAMES;
use crate::arrays::fixed_size_list::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::serde::ArrayChildren;
use crate::stats::ArrayStats;
use crate::validity::Validity;
use crate::vtable;
mod kernel;
mod operations;
mod validity;

vtable!(FixedSizeList, FixedSizeList, FixedSizeListData);

#[derive(Clone, Debug)]
pub struct FixedSizeList;

impl FixedSizeList {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.fixed_size_list");
}

impl VTable for FixedSizeList {
    type ArrayData = FixedSizeListData;

    type Metadata = EmptyMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    fn vtable(_array: &FixedSizeListData) -> &Self {
        &FixedSizeList
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &FixedSizeListData) -> usize {
        array.len
    }

    fn dtype(array: &FixedSizeListData) -> &DType {
        &array.dtype
    }

    fn stats(array: &FixedSizeListData) -> &ArrayStats {
        &array.stats_set
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &FixedSizeListData,
        state: &mut H,
        precision: Precision,
    ) {
        array.elements().array_hash(state, precision);
        array.list_size().hash(state);
        array.validity().array_hash(state, precision);
    }

    fn array_eq(
        array: &FixedSizeListData,
        other: &FixedSizeListData,
        precision: Precision,
    ) -> bool {
        array.elements().array_eq(other.elements(), precision)
            && array.list_size() == other.list_size()
            && array.validity().array_eq(&other.validity(), precision)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("FixedSizeListArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("FixedSizeListArray buffer_name index {idx} out of bounds")
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        Self::PARENT_KERNELS.execute(array, parent, child_idx, ctx)
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
    ) -> VortexResult<FixedSizeListData> {
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

        FixedSizeListData::try_new(elements, *list_size, validity, len)
    }

    fn slots(array: ArrayView<'_, Self>) -> &[Option<ArrayRef>] {
        &array.data().slots
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn with_slots(array: &mut Self::ArrayData, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "FixedSizeListArray expects exactly {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );
        array.slots = slots;
        Ok(())
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }
}
