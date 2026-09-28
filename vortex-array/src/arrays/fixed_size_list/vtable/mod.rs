// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::array::with_empty_buffers;
use crate::arrays::fixed_size_list::FixedSizeListData;
use crate::arrays::fixed_size_list::FixedSizeListSlots;
use crate::arrays::fixed_size_list::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::FixedSizeListBuilder;
use crate::dtype::DType;
mod kernel;
mod operations;
mod plugin;
mod validity;

/// A [`FixedSizeList`]-encoded Vortex array.
pub type FixedSizeListArray = Array<FixedSizeList>;

pub(crate) fn initialize(session: &VortexSession) {
    kernel::initialize(session);
}

#[derive(Clone, Debug)]
pub struct FixedSizeList;

impl ArrayHash for FixedSizeListData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.degenerate_len.hash(state);
    }
}

impl ArrayEq for FixedSizeListData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.degenerate_len == other.degenerate_len
    }
}

impl VTable for FixedSizeList {
    type TypedArrayData = FixedSizeListData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;
    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.fixed_size_list");
        *ID
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

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn validate(
        &self,
        data: &FixedSizeListData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == FixedSizeListSlots::COUNT,
            "FixedSizeListArray expected {} slots, found {}",
            FixedSizeListSlots::COUNT,
            slots.len()
        );
        let DType::FixedSizeList(_, list_size, nullability) = dtype else {
            vortex_bail!("Expected `DType::FixedSizeList`, got {dtype:?}");
        };
        let elements = slots[FixedSizeListSlots::ELEMENTS]
            .as_ref()
            .vortex_expect("FixedSizeListArray elements slot");
        vortex_ensure!(
            if *list_size == 0 {
                data.degenerate_len == len
            } else {
                elements.len() / *list_size as usize == len
            },
            "FixedSizeListArray length {} does not match outer length {}",
            len,
            len
        );

        let actual_dtype =
            DType::FixedSizeList(Arc::new(elements.dtype().clone()), *list_size, *nullability);
        vortex_ensure!(
            &actual_dtype == dtype,
            "FixedSizeListArray dtype {} does not match outer dtype {}",
            actual_dtype,
            dtype
        );

        Ok(())
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        FixedSizeListSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let Some(builder) = builder.as_any_mut().downcast_mut::<FixedSizeListBuilder>() else {
            vortex_bail!("append_to_builder for FixedSizeList requires a FixedSizeListBuilder");
        };
        builder.append_fixed_size_list_array(&array.into_owned(), ctx)
    }
}
