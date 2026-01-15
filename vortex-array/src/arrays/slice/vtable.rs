// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;

use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::Array;
use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::Precision;
use crate::VortexSessionExecute;
use crate::arrays::slice::array::SliceArray;
use crate::arrays::slice::rules::RULES;
use crate::buffer::BufferHandle;
use crate::executor::ExecutionCtx;
use crate::serde::ArrayChildren;
use crate::stats::StatsSetRef;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::ArrayVTable;
use crate::vtable::ArrayVTableExt;
use crate::vtable::BaseArrayVTable;
use crate::vtable::CanonicalVTable;
use crate::vtable::NotSupported;
use crate::vtable::OperationsVTable;
use crate::vtable::VTable;
use crate::vtable::ValidityVTable;
use crate::vtable::VisitorVTable;

vtable!(Slice);

#[derive(Debug)]
pub struct SliceVTable;

impl VTable for SliceVTable {
    type Array = SliceArray;
    type Metadata = SliceMetadata;
    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;

    fn id(&self) -> ArrayId {
        ArrayId::from("vortex.slice")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        SliceVTable.as_vtable()
    }

    fn metadata(array: &Self::Array) -> VortexResult<Self::Metadata> {
        Ok(SliceMetadata(array.range.clone()))
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    fn deserialize(_bytes: &[u8]) -> VortexResult<Self::Metadata> {
        vortex_bail!("Slice array is not serializable")
    }

    fn build(
        &self,
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

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
        // Execute the child to get canonical form, then slice it
        let canonical = array.child.clone().execute::<Canonical>(ctx)?;
        let result = canonical.as_ref().slice(array.range.clone());
        assert!(
            result.is_canonical(),
            "this must be canonical fix the slice impl for the dtype {} showing this error",
            array.dtype()
        );
        // TODO(joe): this is a downcast not a execute.
        Ok(result.to_canonical())
    }

    fn reduce(array: &Self::Array) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array)
    }

    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let inner_range = array.slice_range();

        let combined_start = inner_range.start + range.start;
        let combined_end = inner_range.start + range.end;

        Ok(Some(
            SliceArray::new(array.child().clone(), combined_start..combined_end).into_array(),
        ))
    }
}

impl BaseArrayVTable<SliceVTable> for SliceVTable {
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
}

impl CanonicalVTable<SliceVTable> for SliceVTable {
    fn canonicalize(array: &SliceArray) -> Canonical {
        SliceVTable::execute(array, &mut LEGACY_SESSION.create_execution_ctx())
            .vortex_expect("Canonicalize should be fallible")
    }
}

impl OperationsVTable<SliceVTable> for SliceVTable {
    fn scalar_at(array: &SliceArray, index: usize) -> Scalar {
        array.child.scalar_at(array.range.start + index)
    }
}

impl ValidityVTable<SliceVTable> for SliceVTable {
    fn is_valid(array: &SliceArray, index: usize) -> bool {
        array.child.is_valid(array.range.start + index)
    }

    fn all_valid(array: &SliceArray) -> bool {
        // This is an over-approximation: if the entire child is all valid,
        // then the slice is all valid too.
        array.child.all_valid()
    }

    fn all_invalid(array: &SliceArray) -> bool {
        // This is an over-approximation: if the entire child is all invalid,
        // then the slice is all invalid too.
        array.child.all_invalid()
    }

    fn validity(array: &SliceArray) -> VortexResult<Validity> {
        Ok(array.child.validity()?.slice(array.range.clone()))
    }

    fn validity_mask(array: &SliceArray) -> Mask {
        array.child.validity_mask().slice(array.range.clone())
    }
}

impl VisitorVTable<SliceVTable> for SliceVTable {
    fn visit_buffers(_array: &SliceArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &SliceArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("child", &array.child);
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

    use crate::Array;
    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::SliceArray;
    use crate::assert_arrays_eq;

    #[test]
    fn test_slice_slice() -> VortexResult<()> {
        // Slice(1..4, Slice(2..8, base)) combines to Slice(3..6, base)
        let arr = PrimitiveArray::from_iter(0i32..10).into_array();
        let inner_slice = SliceArray::new(arr, 2..8).into_array();
        let slice = inner_slice.slice(1..4);

        assert_arrays_eq!(slice, PrimitiveArray::from_iter([3i32, 4, 5]));

        Ok(())
    }
}
