// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;
use std::sync::Arc;

use vortex_error::VortexExpect;
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
use crate::DynArray;
use crate::Precision;
use crate::arrays::slice::array::SliceArray;
use crate::arrays::slice::rules::PARENT_RULES;
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

vtable!(Slice);

#[derive(Clone, Debug)]
pub struct Slice;

impl Slice {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.slice");
}

impl VTable for Slice {
    type Array = SliceArray;
    type Metadata = SliceMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    fn vtable(_array: &Self::Array) -> &Self {
        &Slice
    }

    fn id(&self) -> ArrayId {
        Slice::ID
    }

    fn len(array: &SliceArray) -> usize {
        array.range.len()
    }

    fn dtype(array: &SliceArray) -> &DType {
        array.child.dtype()
    }

    fn stats(array: &SliceArray) -> StatsSetRef<'_> {
        array.stats.to_ref(array.as_ref())
    }

    fn array_hash<H: Hasher>(array: &SliceArray, state: &mut H, precision: Precision) {
        array.child.array_hash(state, precision);
        array.range.start.hash(state);
        array.range.end.hash(state);
    }

    fn array_eq(array: &SliceArray, other: &SliceArray, precision: Precision) -> bool {
        array.child.array_eq(&other.child, precision) && array.range == other.range
    }

    fn nbuffers(_array: &Self::Array) -> usize {
        0
    }

    fn buffer(_array: &Self::Array, _idx: usize) -> BufferHandle {
        vortex_panic!("SliceArray has no buffers")
    }

    fn buffer_name(_array: &Self::Array, _idx: usize) -> Option<String> {
        None
    }

    fn nchildren(_array: &Self::Array) -> usize {
        1
    }

    fn child(array: &Self::Array, idx: usize) -> ArrayRef {
        match idx {
            0 => array.child.clone(),
            _ => vortex_panic!("SliceArray child index {idx} out of bounds"),
        }
    }

    fn child_name(_array: &Self::Array, idx: usize) -> String {
        match idx {
            0 => "child".to_string(),
            _ => vortex_panic!("SliceArray child_name index {idx} out of bounds"),
        }
    }

    fn metadata(array: &Self::Array) -> VortexResult<Self::Metadata> {
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
    ) -> VortexResult<Self::Array> {
        assert_eq!(len, metadata.0.len());
        let child = children.get(0, dtype, metadata.0.end)?;
        Ok(SliceArray {
            child,
            range: metadata.0.clone(),
            stats: Default::default(),
        })
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 1,
            "SliceArray expects exactly 1 child, got {}",
            children.len()
        );
        array.child = children
            .into_iter()
            .next()
            .vortex_expect("children length already validated");
        Ok(())
    }

    fn execute(array: Arc<Array<Self>>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        // Execute the child to get canonical form, then slice it
        let Some(canonical) = array.child.as_opt::<AnyCanonical>() else {
            // If the child is not canonical, recurse.
            return array
                .child
                .clone()
                .execute::<ArrayRef>(ctx)?
                .slice(array.slice_range().clone())
                .map(ExecutionResult::done);
        };

        // TODO(ngates): we should inline canonical slice logic here.
        Canonical::from(canonical)
            .as_ref()
            .slice(array.range.clone())
            .map(ExecutionResult::done)
    }

    fn reduce_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}
impl OperationsVTable<Slice> for Slice {
    fn scalar_at(
        array: &SliceArray,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        array.child.scalar_at(array.range.start + index)
    }
}

impl ValidityVTable<Slice> for Slice {
    fn validity(array: &SliceArray) -> VortexResult<Validity> {
        array.child.validity()?.slice(array.range.clone())
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

    use crate::DynArray;
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
