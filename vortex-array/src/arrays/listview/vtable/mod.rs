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
use crate::arrays::listview::ListViewData;
use crate::arrays::listview::ListViewSlots;
use crate::arrays::listview::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::match_each_list_builder;
mod kernel;
mod operations;
mod plugin;
mod validity;
/// A [`ListView`]-encoded Vortex array.
pub type ListViewArray = Array<ListView>;

pub(crate) fn initialize(session: &VortexSession) {
    kernel::initialize(session);
}

#[derive(Clone, Debug)]
pub struct ListView;

#[derive(Clone, prost::Message)]
pub struct ListViewMetadata {
    #[prost(uint64, tag = "1")]
    elements_len: u64,
    #[prost(enumeration = "PType", tag = "2")]
    offset_ptype: i32,
    #[prost(enumeration = "PType", tag = "3")]
    size_ptype: i32,
}

impl ArrayHash for ListViewData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.is_zero_copy_to_list().hash(state);
    }
}

impl ArrayEq for ListViewData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.is_zero_copy_to_list() == other.is_zero_copy_to_list()
    }
}

impl VTable for ListView {
    type TypedArrayData = ListViewData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;
    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.listview");
        *ID
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("ListViewArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("ListViewArray buffer_name index {idx} out of bounds")
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn validate(
        &self,
        _data: &ListViewData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == ListViewSlots::COUNT,
            "ListViewArray expected {} slots, found {}",
            ListViewSlots::COUNT,
            slots.len()
        );
        let elements = slots[ListViewSlots::ELEMENTS]
            .as_ref()
            .vortex_expect("ListViewArray elements slot");
        let offsets = slots[ListViewSlots::OFFSETS]
            .as_ref()
            .vortex_expect("ListViewArray offsets slot");
        let sizes = slots[ListViewSlots::SIZES]
            .as_ref()
            .vortex_expect("ListViewArray sizes slot");
        vortex_ensure!(
            offsets.len() == len && sizes.len() == len,
            "ListViewArray length {} does not match outer length {}",
            offsets.len(),
            len
        );

        let actual_dtype = DType::List(Arc::new(elements.dtype().clone()), dtype.nullability());
        vortex_ensure!(
            &actual_dtype == dtype,
            "ListViewArray dtype {} does not match outer dtype {}",
            actual_dtype,
            dtype
        );

        Ok(())
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        ListViewSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    // The complexity comes from the expansion of `match_each_list_builder!`.
    #[expect(clippy::cognitive_complexity)]
    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        match match_each_list_builder!(&mut *builder, |b| b.append_listview_array(array, ctx)) {
            Some(result) => result,
            None => vortex_bail!(
                "cannot append a ListView array of dtype {} to a {} builder",
                array.dtype(),
                builder.dtype()
            ),
        }
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}
