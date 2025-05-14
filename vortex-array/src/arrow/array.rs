use std::fmt::Debug;

use arrow_array::ArrayRef as ArrowArrayRef;
use vortex_dtype::arrow::FromArrowType;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::arrow::FromArrowArray;
use crate::stats::{ArrayStats, StatsSetRef};
use crate::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, OperationsVTable, VTable, ValidityVTable,
    VisitorVTable,
};
use crate::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayRef, Canonical, EncodingId, EncodingRef,
    IntoArray, vtable,
};

vtable!(Arrow);

impl VTable for ArrowVTable {
    type Array = ArrowArray;
    type Encoding = ArrowEncoding;
    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type SerdeVTable = NotSupported;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.arrow")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(ArrowEncoding.as_ref())
    }
}

/// A Vortex array that wraps an in-memory Arrow array.
// TODO(ngates): consider having each Arrow encoding be a separate encoding ID.
#[derive(Clone, Debug)]
pub struct ArrowEncoding;

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

impl ArrayVTable<ArrowVTable> for ArrowVTable {
    fn len(array: &ArrowArray) -> usize {
        array.inner.len()
    }

    fn dtype(array: &ArrowArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ArrowArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

impl CanonicalVTable<ArrowVTable> for ArrowVTable {
    fn canonicalize(array: &ArrowArray) -> VortexResult<Canonical> {
        ArrayRef::from_arrow(array.inner.clone(), array.dtype.is_nullable()).to_canonical()
    }
}

impl OperationsVTable<ArrowVTable> for ArrowVTable {
    fn slice(array: &ArrowArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        let inner = array.inner.slice(start, stop - start);
        let new_array = ArrowArray {
            inner,
            dtype: array.dtype.clone(),
            stats_set: Default::default(),
        };
        Ok(new_array.into_array())
    }

    fn scalar_at(_array: &ArrowArray, _index: usize) -> VortexResult<Scalar> {
        vortex_bail!("Not supported")
    }
}

impl ValidityVTable<ArrowVTable> for ArrowVTable {
    fn is_valid(array: &ArrowArray, index: usize) -> VortexResult<bool> {
        Ok(array.inner.is_valid(index))
    }

    fn all_valid(array: &ArrowArray) -> VortexResult<bool> {
        Ok(array.inner.logical_null_count() == 0)
    }

    fn all_invalid(array: &ArrowArray) -> VortexResult<bool> {
        Ok(array.inner.logical_null_count() == array.inner.len())
    }

    fn validity_mask(array: &ArrowArray) -> VortexResult<Mask> {
        Ok(array
            .inner
            .logical_nulls()
            .map(|null_buffer| Mask::from_buffer(null_buffer.inner().clone()))
            .unwrap_or_else(|| Mask::new_true(array.inner.len())))
    }
}

impl VisitorVTable<ArrowVTable> for ArrowVTable {
    fn visit_buffers(_array: &ArrowArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(_array: &ArrowArray, _visitor: &mut dyn ArrayChildVisitor) {}
}
