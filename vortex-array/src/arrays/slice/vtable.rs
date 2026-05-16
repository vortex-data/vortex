// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::AnyCanonical;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::Precision;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::arrays::slice::SliceArrayExt;
use crate::arrays::slice::array::CHILD_SLOT;
use crate::arrays::slice::array::SLOT_NAMES;
use crate::arrays::slice::array::SliceData;
use crate::arrays::slice::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::require_child;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

/// A [`Slice`]-encoded Vortex array.
pub type SliceArray = Array<Slice>;

#[derive(Clone, Debug)]
pub struct Slice;

impl ArrayHash for SliceData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _precision: Precision) {
        self.range.start.hash(state);
        self.range.end.hash(state);
    }
}

impl ArrayEq for SliceData {
    fn array_eq(&self, other: &Self, _precision: Precision) -> bool {
        self.range == other.range
    }
}

impl VTable for Slice {
    type TypedArrayData = SliceData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.slice");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots[CHILD_SLOT].is_some(),
            "SliceArray child slot must be present"
        );
        let child = slots[CHILD_SLOT]
            .as_ref()
            .vortex_expect("validated child slot");
        vortex_ensure!(
            child.dtype() == dtype,
            "SliceArray dtype {} does not match outer dtype {}",
            child.dtype(),
            dtype
        );
        vortex_ensure!(
            data.len() == len,
            "SliceArray length {} does not match outer length {}",
            data.len(),
            len
        );
        vortex_ensure!(
            data.range.end <= child.len(),
            "SliceArray range {:?} exceeds child length {}",
            data.range,
            child.len()
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, _idx: usize) -> BufferHandle {
        vortex_panic!("SliceArray has no buffers")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        // TODO(joe): make this configurable
        vortex_bail!("Slice array is not serializable")
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
        vortex_bail!("Slice array is not serializable")
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let array = require_child!(array, array.child(), CHILD_SLOT => AnyCanonical);

        debug_assert!(array.child().is_canonical());
        // TODO(ngates): we should inline canonical slice logic here.
        array
            .child()
            .slice(array.range.clone())
            .map(ExecutionResult::done)
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}
impl OperationsVTable<Slice> for Slice {
    fn scalar_at(
        array: ArrayView<'_, Slice>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        array.child().execute_scalar(array.range.start + index, ctx)
    }

    fn point_scalar_at(
        array: ArrayView<'_, Slice>,
        index: usize,
        d: &mut dyn crate::point_fn::PointDispatch,
    ) -> VortexResult<Scalar> {
        // Push through to the child so the session's caches apply at both
        // levels (slice-level and child-level).
        d.scalar_at(array.child(), array.range.start + index)
    }

    /// Search the child once, then clamp the result into the slice's local
    /// coordinate space. The child must be sorted for this to be meaningful;
    /// that precondition is inherited from the slice array.
    fn point_search_sorted(
        array: ArrayView<'_, Slice>,
        value: &Scalar,
        side: crate::search_sorted::SearchSortedSide,
        d: &mut dyn crate::point_fn::PointDispatch,
    ) -> VortexResult<crate::search_sorted::SearchResult> {
        use crate::search_sorted::SearchResult;
        let offset = array.range.start;
        let stop = array.range.end;
        let len = stop - offset;

        let child_result = d.search_sorted(array.child(), value, side)?;
        let raw = child_result.to_index();

        // Map child's position into the slice's [0, len] range. A child position
        // inside [offset, stop) maps to (raw - offset); outside that range, the
        // value lies before/after the slice entirely.
        Ok(match child_result {
            SearchResult::Found(i) if i >= offset && i < stop => SearchResult::Found(i - offset),
            SearchResult::Found(_) | SearchResult::NotFound(_) => {
                let local = if raw <= offset {
                    0
                } else if raw >= stop {
                    len
                } else {
                    raw - offset
                };
                SearchResult::NotFound(local)
            }
        })
    }
}

impl ValidityVTable<Slice> for Slice {
    fn validity(array: ArrayView<'_, Slice>) -> VortexResult<Validity> {
        array.child().validity()?.slice(array.range.clone())
    }
}

pub struct SliceMetadata(pub(super) Range<usize>);

impl Debug for SliceMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}..{}", self.0.start, self.0.end)
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::SliceArray;
    use crate::assert_arrays_eq;

    #[test]
    fn test_slice_slice() -> VortexResult<()> {
        // Slice(1..4, Slice(2..8, base)) combines to Slice(3..6, base)
        let arr = PrimitiveArray::from_iter(0i32..10).into_array();
        let inner_slice = SliceArray::new(arr, 2..8).into_array();
        let slice = inner_slice.slice(1..4)?;

        assert_arrays_eq!(slice, PrimitiveArray::from_iter([3i32, 4, 5]));

        Ok(())
    }
}
