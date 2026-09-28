// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::registry::CachedId;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::array::with_empty_buffers;
use crate::arrays::list::ListData;
use crate::arrays::list::ListSlots;
use crate::arrays::list::compute::rules::PARENT_RULES;
use crate::arrays::listview::list_view_from_list;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::match_each_list_builder;
mod operations;
mod plugin;
mod validity;
/// A [`List`]-encoded Vortex array.
pub type ListArray = Array<List>;

#[derive(Clone, prost::Message)]
pub struct ListMetadata {
    #[prost(uint64, tag = "1")]
    elements_len: u64,
    #[prost(enumeration = "PType", tag = "2")]
    offset_ptype: i32,
}

impl ArrayHash for ListData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _accuracy: EqMode) {}
}

impl ArrayEq for ListData {
    fn array_eq(&self, _other: &Self, _accuracy: EqMode) -> bool {
        true
    }
}

impl VTable for List {
    type TypedArrayData = ListData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;
    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.list");
        *ID
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("ListArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("ListArray buffer_name index {idx} out of bounds")
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
        _data: &ListData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == ListSlots::COUNT,
            "ListArray expected {} slots, found {}",
            ListSlots::COUNT,
            slots.len()
        );
        let elements = slots[ListSlots::ELEMENTS]
            .as_ref()
            .vortex_expect("ListArray elements slot");
        let offsets = slots[ListSlots::OFFSETS]
            .as_ref()
            .vortex_expect("ListArray offsets slot");
        vortex_ensure!(
            offsets.len().saturating_sub(1) == len,
            "ListArray length {} does not match outer length {}",
            offsets.len().saturating_sub(1),
            len
        );

        let actual_dtype = DType::List(Arc::new(elements.dtype().clone()), dtype.nullability());
        vortex_ensure!(
            &actual_dtype == dtype,
            "ListArray dtype {} does not match outer dtype {}",
            actual_dtype,
            dtype
        );

        Ok(())
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        ListSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            list_view_from_list(array, ctx)?.into_array(),
        ))
    }

    // The complexity comes from the expansion of `match_each_list_builder!`.
    #[expect(clippy::cognitive_complexity)]
    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        match match_each_list_builder!(&mut *builder, |b| b.append_list_array(array, ctx)) {
            Some(result) => result,
            None => vortex_bail!(
                "cannot append a List array of dtype {} to a {} builder",
                array.dtype(),
                builder.dtype()
            ),
        }
    }
}

#[derive(Clone, Debug)]
pub struct List;
