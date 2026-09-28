// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::EmptyArrayData;
use crate::array::VTable;
use crate::array::with_empty_buffers;
use crate::arrays::union::UnionArrayExt;
use crate::arrays::union::UnionSlots;
use crate::arrays::union::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::dtype::DType;

mod operations;
mod plugin;
mod validate;
mod validity;

/// A canonical [`Union`]-encoded array.
///
/// This is similar to Arrow's "spase" union type.
pub type UnionArray = Array<Union>;

/// The canonical Union encoding.
///
/// This is similar to Arrow's "spase" union type.
#[derive(Clone, Debug)]
pub struct Union;

impl VTable for Union {
    type TypedArrayData = EmptyArrayData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.union");
        *ID
    }

    fn validate(
        &self,
        _data: &EmptyArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let type_ids = slots
            .get(UnionSlots::TYPE_IDS)
            .and_then(Option::as_ref)
            .ok_or_else(|| vortex_err!("UnionArray is missing its type_ids slot"))?;
        let variant_arrays = slots
            .get(UnionSlots::CHILDREN_OFFSET..)
            .unwrap_or_default()
            .iter()
            .enumerate()
            .map(|(index, slot)| {
                slot.as_ref()
                    .ok_or_else(|| vortex_err!("UnionArray is missing child {index}"))
            })
            .collect::<VortexResult<Vec<_>>>()?;

        validate::validate_union_components(type_ids, &variant_arrays, dtype, len)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("UnionArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("UnionArray buffer_name index {idx} out of bounds")
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn slot_name(array: ArrayView<'_, Self>, idx: usize) -> String {
        if idx == UnionSlots::TYPE_IDS {
            "type_ids".to_string()
        } else {
            array.variants().names()[idx - UnionSlots::CHILDREN_OFFSET].to_string()
        }
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    fn append_to_builder(
        _array: ArrayView<'_, Self>,
        _builder: &mut dyn ArrayBuilder,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        todo!("TODO(connor)[Union]: implement append_to_builder for Union arrays")
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}
