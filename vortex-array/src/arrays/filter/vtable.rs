// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hasher;
use std::ops::Range;

use vortex_compute::filter::Filter;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use super::execute::filter_canonical;
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
use crate::arrays::ConstantArray;
use crate::arrays::filter::array::FilterArray;
use crate::arrays::filter::rules::PARENT_RULES;
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

vtable!(Filter);

#[derive(Debug)]
pub struct FilterVTable;

impl VTable for FilterVTable {
    type Array = FilterArray;
    type Metadata = FilterMetadata;
    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;

    fn id(&self) -> ArrayId {
        ArrayId::from("vortex.filter")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        FilterVTable.as_vtable()
    }

    fn metadata(array: &Self::Array) -> VortexResult<Self::Metadata> {
        Ok(FilterMetadata(array.mask.clone()))
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    fn deserialize(_bytes: &[u8]) -> VortexResult<Self::Metadata> {
        vortex_bail!("Filter array is not serializable")
    }

    fn build(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &FilterMetadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<Self::Array> {
        assert_eq!(len, metadata.0.true_count());
        let child = children.get(0, dtype, metadata.0.len())?;
        Ok(FilterArray {
            child,
            mask: metadata.0.clone(),
            offset: 0,
            len,
            stats: Default::default(),
        })
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 1,
            "FilterArray expects exactly 1 child, got {}",
            children.len()
        );
        array.child = children
            .into_iter()
            .next()
            .vortex_expect("children length already validated");
        Ok(())
    }

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
        if let Some(canonical) = execute_fast_path(array, ctx)? {
            return Ok(canonical);
        }

        let canonical = filter_canonical(array.child.clone().execute(ctx)?, &array.mask);

        let full_len = array.mask.true_count();
        vortex_ensure!(
            canonical.as_ref().dtype() == array.dtype(),
            "Filter result dtype mismatch: expected {:?}, got {:?}",
            array.dtype(),
            canonical.as_ref().dtype()
        );
        vortex_ensure!(
            canonical.as_ref().len() == full_len,
            "Filter result length mismatch: expected {}, got {}",
            full_len,
            canonical.as_ref().len()
        );

        // If this is a sliced view, slice the result
        if array.offset > 0 || array.len < full_len {
            let sliced = canonical
                .as_ref()
                .slice(array.offset..array.offset + array.len);
            return sliced.execute(ctx);
        }

        Ok(canonical)
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}

/// Check for fast-path execution conditions.
pub(super) fn execute_fast_path(
    array: &FilterArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Canonical>> {
    // Empty result - array has no elements
    if array.len == 0 {
        return Ok(Some(Canonical::empty(array.dtype())));
    }

    // Full pass-through - mask selects everything and no offset/slice
    let true_count = array.mask.true_count();
    if true_count == array.mask.len() && array.offset == 0 && array.len == true_count {
        return Ok(Some(array.child.clone().execute(ctx)?));
    }

    // All null - child has no valid values
    if array.validity_mask().true_count() == 0 {
        return Ok(Some(
            ConstantArray::new(Scalar::null(array.dtype().clone()), array.len)
                .into_array()
                .execute(ctx)?,
        ));
    }

    Ok(None)
}

impl BaseArrayVTable<FilterVTable> for FilterVTable {
    fn len(array: &FilterArray) -> usize {
        array.len
    }

    fn dtype(array: &FilterArray) -> &DType {
        array.child.dtype()
    }

    fn stats(array: &FilterArray) -> StatsSetRef<'_> {
        array.stats.to_ref(array.as_ref())
    }

    fn array_hash<H: Hasher>(array: &FilterArray, state: &mut H, precision: Precision) {
        array.child.array_hash(state, precision);
        array.mask.array_hash(state, precision);
    }

    fn array_eq(array: &FilterArray, other: &FilterArray, precision: Precision) -> bool {
        array.child.array_eq(&other.child, precision) && array.mask.array_eq(&other.mask, precision)
    }
}

impl CanonicalVTable<FilterVTable> for FilterVTable {
    fn canonicalize(array: &FilterArray) -> Canonical {
        FilterVTable::execute(array, &mut LEGACY_SESSION.create_execution_ctx())
            .vortex_expect("Canonicalize should be fallible")
    }
}

impl OperationsVTable<FilterVTable> for FilterVTable {
    fn slice(array: &FilterArray, range: Range<usize>) -> ArrayRef {
        array.slice(range.start, range.len()).into_array()
    }

    fn scalar_at(array: &FilterArray, index: usize) -> Scalar {
        let logical_idx = array.offset + index;
        let physical_idx = array.mask.rank(logical_idx);
        array.child.scalar_at(physical_idx)
    }
}

impl ValidityVTable<FilterVTable> for FilterVTable {
    fn is_valid(array: &FilterArray, index: usize) -> bool {
        let logical_idx = array.offset + index;
        let physical_idx = array.mask.rank(logical_idx);
        array.child.is_valid(physical_idx)
    }

    fn all_valid(array: &FilterArray) -> bool {
        // An over-approximation: if the child is all valid, then the filtered array is all valid.
        array.child.all_valid()
    }

    fn all_invalid(array: &FilterArray) -> bool {
        // An over-approximation: if the child is all invalid, then the filtered array is all invalid.
        array.child.all_invalid()
    }

    fn validity(array: &FilterArray) -> VortexResult<Validity> {
        let full_validity = array.child.validity()?.filter(&array.mask)?;
        // If this is a sliced view, slice the validity
        let full_len = array.mask.true_count();
        if array.offset > 0 || array.len < full_len {
            return Ok(full_validity.slice(array.offset..array.offset + array.len));
        }
        Ok(full_validity)
    }

    fn validity_mask(array: &FilterArray) -> Mask {
        let full_mask = Filter::filter(&array.child.validity_mask(), &array.mask);
        // If this is a sliced view, slice the mask
        let full_len = array.mask.true_count();
        if array.offset > 0 || array.len < full_len {
            return full_mask.slice(array.offset..array.offset + array.len);
        }
        full_mask
    }
}

impl VisitorVTable<FilterVTable> for FilterVTable {
    fn visit_buffers(_array: &FilterArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &FilterArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("child", &array.child);
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

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use crate::Array;
    use crate::IntoArray;
    use crate::arrays::FilterArray;
    use crate::arrays::PrimitiveArray;

    #[test]
    fn test_slice_basic() -> VortexResult<()> {
        // child = [0, 1, 2, 3, 4]
        // mask = [T, F, T, F, T] (selects indices 0, 2, 4)
        // FilterArray logical elements: [0, 2, 4] with logical indices 0, 1, 2
        let child = PrimitiveArray::from_iter([0i32, 1, 2, 3, 4]).into_array();
        let mask = Mask::from_iter([true, false, true, false, true]);
        let filter_array = FilterArray::new(child, mask).into_array();

        assert_eq!(filter_array.len(), 3);

        // slice(0..2) should give [0, 2] (logical indices 0 and 1)
        let sliced = filter_array.slice(0..2);
        assert_eq!(sliced.len(), 2);
        assert_eq!(sliced.scalar_at(0).as_primitive().as_::<i32>(), Some(0));
        assert_eq!(sliced.scalar_at(1).as_primitive().as_::<i32>(), Some(2));

        Ok(())
    }

    #[test]
    fn test_slice_middle() -> VortexResult<()> {
        // child = [0, 1, 2, 3, 4]
        // mask = [T, F, T, F, T] (selects indices 0, 2, 4)
        // FilterArray logical elements: [0, 2, 4]
        let child = PrimitiveArray::from_iter([0i32, 1, 2, 3, 4]).into_array();
        let mask = Mask::from_iter([true, false, true, false, true]);
        let filter_array = FilterArray::new(child, mask).into_array();

        // slice(1..3) should give [2, 4] (logical indices 1 and 2)
        let sliced = filter_array.slice(1..3);
        assert_eq!(sliced.len(), 2);
        assert_eq!(sliced.scalar_at(0).as_primitive().as_::<i32>(), Some(2));
        assert_eq!(sliced.scalar_at(1).as_primitive().as_::<i32>(), Some(4));

        Ok(())
    }

    #[test]
    fn test_slice_single_element() -> VortexResult<()> {
        let child = PrimitiveArray::from_iter([0i32, 1, 2, 3, 4]).into_array();
        let mask = Mask::from_iter([true, false, true, false, true]);
        let filter_array = FilterArray::new(child, mask).into_array();

        // slice(1..2) should give [2] (logical index 1 only)
        let sliced = filter_array.slice(1..2);
        assert_eq!(sliced.len(), 1);
        assert_eq!(sliced.scalar_at(0).as_primitive().as_::<i32>(), Some(2));

        Ok(())
    }

    #[test]
    fn test_slice_empty() -> VortexResult<()> {
        let child = PrimitiveArray::from_iter([0i32, 1, 2, 3, 4]).into_array();
        let mask = Mask::from_iter([true, false, true, false, true]);
        let filter_array = FilterArray::new(child, mask).into_array();

        // slice(1..1) should give empty array
        let sliced = filter_array.slice(1..1);
        assert_eq!(sliced.len(), 0);

        Ok(())
    }

    #[test]
    fn test_scalar_at() -> VortexResult<()> {
        // child = [0, 1, 2, 3, 4]
        // mask = [T, F, T, F, T] (selects indices 0, 2, 4)
        // FilterArray logical elements: [0, 2, 4]
        let child = PrimitiveArray::from_iter([0i32, 1, 2, 3, 4]).into_array();
        let mask = Mask::from_iter([true, false, true, false, true]);
        let filter_array = FilterArray::new(child, mask).into_array();

        assert_eq!(
            filter_array.scalar_at(0).as_primitive().as_::<i32>(),
            Some(0)
        );
        assert_eq!(
            filter_array.scalar_at(1).as_primitive().as_::<i32>(),
            Some(2)
        );
        assert_eq!(
            filter_array.scalar_at(2).as_primitive().as_::<i32>(),
            Some(4)
        );

        Ok(())
    }

    #[test]
    fn test_slice_chained() -> VortexResult<()> {
        // Test that chained slices work correctly (verifies O(1) offset accumulation)
        // child = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        // mask = [T, F, T, F, T, F, T, F, T, F] (selects indices 0, 2, 4, 6, 8)
        // FilterArray logical elements: [0, 2, 4, 6, 8]
        let child = PrimitiveArray::from_iter([0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9]).into_array();
        let mask = Mask::from_iter([
            true, false, true, false, true, false, true, false, true, false,
        ]);
        let filter_array = FilterArray::new(child, mask).into_array();

        assert_eq!(filter_array.len(), 5);

        // First slice: [1..4] -> [2, 4, 6]
        let sliced1 = filter_array.slice(1..4);
        assert_eq!(sliced1.len(), 3);
        assert_eq!(sliced1.scalar_at(0).as_primitive().as_::<i32>(), Some(2));
        assert_eq!(sliced1.scalar_at(1).as_primitive().as_::<i32>(), Some(4));
        assert_eq!(sliced1.scalar_at(2).as_primitive().as_::<i32>(), Some(6));

        // Second slice of the first slice: [1..2] -> [4]
        let sliced2 = sliced1.slice(1..2);
        assert_eq!(sliced2.len(), 1);
        assert_eq!(sliced2.scalar_at(0).as_primitive().as_::<i32>(), Some(4));

        Ok(())
    }
}
