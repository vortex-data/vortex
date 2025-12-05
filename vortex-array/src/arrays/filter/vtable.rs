// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hasher;
use std::ops::Range;

use vortex_buffer::BufferHandle;
use vortex_compute::filter::Filter;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
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
use crate::Precision;
use crate::arrays::LEGACY_SESSION;
use crate::arrays::filter::array::FilterArray;
use crate::execution::ExecutionCtx;
use crate::kernel::KernelRef;
use crate::kernel::kernel;
use crate::serde::ArrayChildren;
use crate::stats::StatsSetRef;
use crate::vectors::VectorIntoArray;
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
    type Metadata = Mask;
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
        Ok(array.mask.clone())
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
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<Self::Array> {
        let child = children.get(0, dtype, len)?;
        Ok(FilterArray {
            child,
            mask: metadata.clone(),
            stats: Default::default(),
        })
    }

    fn bind_kernel(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<KernelRef> {
        let child = array.child.bind_kernel(ctx)?;
        let mask = array.mask.clone();
        Ok(kernel(move || Ok(Filter::filter(&child.execute()?, &mask))))
    }
}

impl BaseArrayVTable<FilterVTable> for FilterVTable {
    fn len(array: &FilterArray) -> usize {
        array.mask.true_count()
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
        let vector =
            FilterVTable::bind_kernel(array, &mut ExecutionCtx::new(LEGACY_SESSION.clone()))
                .vortex_expect("Canonicalize should be fallible")
                .execute()
                .vortex_expect("Canonicalize should be fallible");
        vector.into_array(array.dtype()).to_canonical()
    }
}

impl OperationsVTable<FilterVTable> for FilterVTable {
    fn slice(array: &FilterArray, range: Range<usize>) -> ArrayRef {
        FilterArray::new(array.child.slice(range.clone()), array.mask.slice(range)).into_array()
    }

    fn scalar_at(array: &FilterArray, index: usize) -> Scalar {
        let rank_idx = array.mask.rank(index);
        array.child.scalar_at(rank_idx)
    }
}

impl ValidityVTable<FilterVTable> for FilterVTable {
    fn is_valid(array: &FilterArray, index: usize) -> bool {
        let rank_idx = array.mask.rank(index);
        array.child.is_valid(rank_idx)
    }

    fn all_valid(array: &FilterArray) -> bool {
        // An over-approximation: if the child is all valid, then the filtered array is all valid.
        array.child.all_valid()
    }

    fn all_invalid(array: &FilterArray) -> bool {
        // An over-approximation: if the child is all invalid, then the filtered array is all invalid.
        array.child.all_invalid()
    }

    fn validity_mask(array: &FilterArray) -> Mask {
        Filter::filter(&array.child.validity_mask(), &array.mask)
    }
}

impl VisitorVTable<FilterVTable> for FilterVTable {
    fn visit_buffers(_array: &FilterArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &FilterArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("child", &array.child);
    }
}
