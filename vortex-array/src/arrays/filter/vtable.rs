// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hasher;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::IntoArray;
use crate::Precision;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::arrays::filter::array::CHILD_SLOT;
use crate::arrays::filter::array::FilterData;
use crate::arrays::filter::array::SLOT_NAMES;
use crate::arrays::filter::FilterArrayExt;
use crate::arrays::filter::execute::execute_filter;
use crate::arrays::filter::execute::execute_filter_fast_paths;
use crate::arrays::filter::rules::PARENT_RULES;
use crate::arrays::filter::rules::RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable;

vtable!(Filter, Filter, FilterData);

#[derive(Clone, Debug)]
pub struct Filter;

impl Filter {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.filter");
}

impl VTable for Filter {
    type ArrayData = FilterData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn validate(
        &self,
        data: &Self::ArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(slots[CHILD_SLOT].is_some(), "FilterArray child slot must be present");
        let child = slots[CHILD_SLOT].as_ref().vortex_expect("validated child slot");
        vortex_ensure!(
            child.dtype() == dtype,
            "FilterArray dtype {} does not match outer dtype {}",
            child.dtype(),
            dtype
        );
        vortex_ensure!(
            data.len() == len,
            "FilterArray length {} does not match outer length {}",
            data.len(),
            len
        );
        vortex_ensure!(
            child.len() == data.mask.len(),
            "FilterArray child length {} does not match mask length {}",
            child.len(),
            data.mask.len()
        );
        Ok(())
    }

    fn array_hash<H: Hasher>(array: ArrayView<'_, Self>, state: &mut H, precision: Precision) {
        array.child().array_hash(state, precision);
        array.mask.array_hash(state, precision);
    }

    fn array_eq(array: ArrayView<'_, Self>, other: ArrayView<'_, Self>, precision: Precision) -> bool {
        array.child().array_eq(other.child(), precision)
            && array.mask.array_eq(&other.mask, precision)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, _idx: usize) -> BufferHandle {
        vortex_panic!("FilterArray has no buffers")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn serialize(_array: ArrayView<'_, Self>) -> VortexResult<Option<Vec<u8>>> {
        // TODO(joe): make this configurable
        vortex_bail!("Filter array is not serializable")
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],

        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<crate::array::ArrayParts<Self>> {
        vortex_bail!("Filter array is not serializable")
    }


    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        if let Some(canonical) = execute_filter_fast_paths(array.as_view(), ctx)? {
            return Ok(ExecutionResult::done(canonical));
        }
        let Mask::Values(mask_values) = &array.mask else {
            unreachable!("`execute_filter_fast_paths` handles AllTrue and AllFalse")
        };

        // We rely on the optimization pass that runs prior to this execution for filter pushdown,
        // so now we can just execute the filter without worrying.
        Ok(ExecutionResult::done(
            execute_filter(array.child().clone().execute(ctx)?, mask_values).into_array(),
        ))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn reduce(array: ArrayView<'_, Self>) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array)
    }
}
impl OperationsVTable<Filter> for Filter {
    fn scalar_at(
        array: ArrayView<'_, Filter>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let rank_idx = array.mask.rank(index);
        array.child().scalar_at(rank_idx)
    }
}

impl ValidityVTable<Filter> for Filter {
    fn validity(array: ArrayView<'_, Filter>) -> VortexResult<Validity> {
        array.child().validity()?.filter(&array.mask)
    }
}
