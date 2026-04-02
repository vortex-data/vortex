// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::DynArray;
use crate::IntoArray;
use crate::Precision;
use crate::arrays::filter::array::FilterArray;
use crate::arrays::filter::array::NUM_SLOTS;
use crate::arrays::filter::array::SLOT_NAMES;
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
use crate::stats::StatsSetRef;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::Array;
use crate::vtable::ArrayId;
use crate::vtable::OperationsVTable;
use crate::vtable::VTable;
use crate::vtable::ValidityVTable;

vtable!(Filter);

#[derive(Clone, Debug)]
pub struct Filter;

impl Filter {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.filter");
}

impl VTable for Filter {
    type Array = FilterArray;
    type Metadata = FilterMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    fn vtable(_array: &Self::Array) -> &Self {
        &Filter
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &FilterArray) -> usize {
        array.mask.true_count()
    }

    fn dtype(array: &FilterArray) -> &DType {
        array.child().dtype()
    }

    fn stats(array: &FilterArray) -> StatsSetRef<'_> {
        array.stats.to_ref(array.as_ref())
    }

    fn array_hash<H: Hasher>(array: &FilterArray, state: &mut H, precision: Precision) {
        array.child().array_hash(state, precision);
        array.mask.array_hash(state, precision);
    }

    fn array_eq(array: &FilterArray, other: &FilterArray, precision: Precision) -> bool {
        array.child().array_eq(other.child(), precision)
            && array.mask.array_eq(&other.mask, precision)
    }

    fn nbuffers(_array: &Self::Array) -> usize {
        0
    }

    fn buffer(_array: &Self::Array, _idx: usize) -> BufferHandle {
        vortex_panic!("FilterArray has no buffers")
    }

    fn buffer_name(_array: &Self::Array, _idx: usize) -> Option<String> {
        None
    }

    fn slots(array: &Self::Array) -> &[Option<ArrayRef>] {
        &array.slots
    }

    fn slot_name(_array: &Self::Array, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn metadata(array: &Self::Array) -> VortexResult<Self::Metadata> {
        Ok(FilterMetadata(array.mask.clone()))
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        // TODO(joe): make this configurable
        vortex_bail!("Filter array is not serializable")
    }

    fn deserialize(
        _bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        vortex_bail!("Filter array is not serializable")
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &FilterMetadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<Self::Array> {
        assert_eq!(len, metadata.0.true_count());
        let child = children.get(0, dtype, metadata.0.len())?;
        FilterArray::try_new(child, metadata.0.clone())
    }

    fn with_slots(array: &mut Self::Array, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "FilterArray expects exactly {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );
        array.slots = slots;
        Ok(())
    }

    fn execute(array: Arc<Array<Self>>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        if let Some(canonical) = execute_filter_fast_paths(&array, ctx)? {
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
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn reduce(array: &Array<Self>) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array)
    }
}
impl OperationsVTable<Filter> for Filter {
    fn scalar_at(
        array: &FilterArray,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let rank_idx = array.mask.rank(index);
        array.child().scalar_at(rank_idx)
    }
}

impl ValidityVTable<Filter> for Filter {
    fn validity(array: &FilterArray) -> VortexResult<Validity> {
        array.child().validity()?.filter(&array.mask)
    }
}

pub struct FilterMetadata(pub(super) Mask);

impl Debug for FilterMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} / {} => {}",
            self.0.true_count(),
            self.0.len(),
            self.0.density()
        )
    }
}
