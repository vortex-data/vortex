// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Range;

use arrow_array::ArrayRef as ArrowArrayRef;
use vortex_buffer::ByteBuffer;
use vortex_dtype::arrow::FromArrowType;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexResult, vortex_bail, vortex_panic};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::arrow::FromArrowArray;
use crate::serde::ArrayChildren;
use crate::stats::{ArrayStats, StatsSetRef};
use crate::vtable::{
    ArrayId, ArrayVTable, ArrayVTableExt, BaseArrayVTable, CanonicalVTable, NotSupported,
    OperationsVTable, VTable, ValidityVTable, VisitorVTable,
};
use crate::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayRef, Canonical, EmptyMetadata, IntoArray,
    Precision, vtable,
};

vtable!(Arrow);

impl VTable for ArrowVTable {
    type Array = ArrowArray;

    type Metadata = EmptyMetadata;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type OperatorVTable = NotSupported;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("vortex.arrow")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        ArrowVTable.as_vtable()
    }

    fn metadata(_array: &Self::Array) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    fn deserialize(_buffer: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn build(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &Self::Metadata,
        _buffers: &[ByteBuffer],
        _children: &dyn ArrayChildren,
    ) -> VortexResult<Self::Array> {
        vortex_bail!("ArrowArray cannot be deserialized")
    }
}

/// A Vortex array that wraps an in-memory Arrow array.
// TODO(ngates): consider having each Arrow encoding be a separate encoding ID.
#[derive(Clone, Debug)]
pub struct ArrowVTable;

#[derive(Clone, Debug)]
pub struct ArrowArray {
    inner: ArrowArrayRef,
    dtype: DType,
    stats_set: ArrayStats,
}

impl ArrowArray {
    pub fn new(arrow_array: ArrowArrayRef, nullability: Nullability) -> Self {
        let dtype = DType::from_arrow((arrow_array.data_type(), nullability));
        Self {
            inner: arrow_array,
            dtype,
            stats_set: Default::default(),
        }
    }

    pub fn inner(&self) -> &ArrowArrayRef {
        &self.inner
    }
}

impl BaseArrayVTable<ArrowVTable> for ArrowVTable {
    fn len(array: &ArrowArray) -> usize {
        array.inner.len()
    }

    fn dtype(array: &ArrowArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ArrowArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &ArrowArray, state: &mut H, _precision: Precision) {
        array.dtype.hash(state);
        // Hash based on pointer to the inner Arrow array since Arrow doesn't support hashing.
        std::sync::Arc::as_ptr(&array.inner).hash(state);
    }

    fn array_eq(array: &ArrowArray, other: &ArrowArray, _precision: Precision) -> bool {
        array.dtype == other.dtype && std::sync::Arc::ptr_eq(&array.inner, &other.inner)
    }
}

impl CanonicalVTable<ArrowVTable> for ArrowVTable {
    fn canonicalize(array: &ArrowArray) -> Canonical {
        ArrayRef::from_arrow(array.inner.as_ref(), array.dtype.is_nullable()).to_canonical()
    }
}

impl OperationsVTable<ArrowVTable> for ArrowVTable {
    fn slice(array: &ArrowArray, range: Range<usize>) -> ArrayRef {
        let inner = array.inner.slice(range.start, range.len());
        let new_array = ArrowArray {
            inner,
            dtype: array.dtype.clone(),
            stats_set: Default::default(),
        };
        new_array.into_array()
    }

    fn scalar_at(_array: &ArrowArray, _index: usize) -> Scalar {
        vortex_panic!("Not supported")
    }
}

impl ValidityVTable<ArrowVTable> for ArrowVTable {
    fn is_valid(array: &ArrowArray, index: usize) -> bool {
        array.inner.is_valid(index)
    }

    fn all_valid(array: &ArrowArray) -> bool {
        array.inner.logical_null_count() == 0
    }

    fn all_invalid(array: &ArrowArray) -> bool {
        array.inner.logical_null_count() == array.inner.len()
    }

    fn validity_mask(array: &ArrowArray) -> Mask {
        array
            .inner
            .logical_nulls()
            .map(|null_buffer| Mask::from_buffer(null_buffer.inner().clone().into()))
            .unwrap_or_else(|| Mask::new_true(array.inner.len()))
    }
}

impl VisitorVTable<ArrowVTable> for ArrowVTable {
    fn visit_buffers(_array: &ArrowArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(_array: &ArrowArray, _visitor: &mut dyn ArrayChildVisitor) {}
}
