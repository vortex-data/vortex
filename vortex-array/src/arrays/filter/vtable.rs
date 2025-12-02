// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hasher;

use vortex_buffer::BufferHandle;
use vortex_dtype::DType;
use vortex_error::vortex_bail;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_vector::Vector;

use crate::arrays::filter::array::FilterArray;
use crate::execution::ExecutionCtx;
use crate::serde::ArrayChildren;
use crate::stats::StatsSetRef;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::ArrayVTable;
use crate::vtable::ArrayVTableExt;
use crate::vtable::BaseArrayVTable;
use crate::vtable::NotSupported;
use crate::vtable::VTable;
use crate::vtable::VisitorVTable;
use crate::Array;
use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::Precision;

vtable!(Filter);

#[derive(Debug)]
pub struct FilterVTable;

impl VTable for FilterVTable {
    type Array = FilterArray;
    type Metadata = Mask;
    type ArrayVTable = Self;
    type CanonicalVTable = NotSupported;
    type OperationsVTable = NotSupported;
    type ValidityVTable = NotSupported;
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

    fn batch_execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<Vector> {
        let child = array.child.batch_execute(ctx)?;
        Ok(vortex_compute::filter::Filter::filter(&child, &array.mask))
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

impl VisitorVTable<FilterVTable> for FilterVTable {
    fn visit_buffers(_array: &FilterArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &FilterArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("child", &array.child);
    }
}
