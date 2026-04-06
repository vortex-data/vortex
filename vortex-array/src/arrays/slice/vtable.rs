// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::AnyCanonical;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::Precision;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::arrays::slice::array::NUM_SLOTS;
use crate::arrays::slice::array::SLOT_NAMES;
use crate::arrays::slice::array::SliceData;
use crate::arrays::slice::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::stats::ArrayStats;
use crate::validity::Validity;
use crate::vtable;

vtable!(Slice, Slice, SliceData);

#[derive(Clone, Debug)]
pub struct Slice;

impl Slice {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.slice");
}

impl VTable for Slice {
    type ArrayData = SliceData;
    type Metadata = SliceMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    fn vtable(_array: &SliceData) -> &Self {
        &Slice
    }

    fn id(&self) -> ArrayId {
        Slice::ID
    }

    fn len(array: &SliceData) -> usize {
        array.range.len()
    }

    fn dtype(array: &SliceData) -> &DType {
        array.child().dtype()
    }

    fn stats(array: &SliceData) -> &ArrayStats {
        &array.stats
    }

    fn array_hash<H: Hasher>(array: &SliceData, state: &mut H, precision: Precision) {
        array.child().array_hash(state, precision);
        array.range.start.hash(state);
        array.range.end.hash(state);
    }

    fn array_eq(array: &SliceData, other: &SliceData, precision: Precision) -> bool {
        array.child().array_eq(other.child(), precision) && array.range == other.range
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

    fn slots(array: ArrayView<'_, Self>) -> &[Option<ArrayRef>] {
        &array.data().slots
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn metadata(array: ArrayView<'_, Self>) -> VortexResult<Self::Metadata> {
        Ok(SliceMetadata(array.range.clone()))
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        // TODO(joe): make this configurable
        vortex_bail!("Slice array is not serializable")
    }

    fn deserialize(
        _bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        vortex_bail!("Slice array is not serializable")
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &SliceMetadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<Self::ArrayData> {
        assert_eq!(len, metadata.0.len());
        let child = children.get(0, dtype, metadata.0.end)?;
        SliceData::try_new(child, metadata.0.clone())
    }

    fn with_slots(array: &mut Self::ArrayData, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "SliceArray expects exactly {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );
        array.slots = slots;
        Ok(())
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        // Execute the child to get canonical form, then slice it
        let Some(canonical) = array.child().as_opt::<AnyCanonical>() else {
            // If the child is not canonical, recurse.
            return array
                .child()
                .clone()
                .execute::<ArrayRef>(ctx)?
                .slice(array.slice_range().clone())
                .map(ExecutionResult::done);
        };

        // TODO(ngates): we should inline canonical slice logic here.
        Canonical::from(canonical)
            .into_array()
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
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        array.child().scalar_at(array.range.start + index)
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
