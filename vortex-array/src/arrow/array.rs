use std::fmt::Debug;

use arrow_array::{Array, ArrayRef as ArrowArrayRef};
use vortex_dtype::arrow::FromArrowType;
use vortex_dtype::{DType, Nullability};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arcref::ArcRef;
use crate::stats::StatsSetRef;
use crate::vtable::{ComputeVTable, EncodingVTable, VTableRef};
use crate::{
    ArrayCanonicalImpl, ArrayImpl, ArrayRef, ArrayStatisticsImpl, ArrayValidityImpl,
    ArrayVariantsImpl, ArrayVisitorImpl, Canonical, EmptyMetadata, Encoding, EncodingId,
};

/// A Vortex array that wraps an in-memory Arrow array.
#[derive(Debug)]
pub struct ArrowArrayEncoding;

impl Encoding for ArrowArrayEncoding {
    type Array = ArrowArray;
    type Metadata = EmptyMetadata;
}

impl EncodingVTable for ArrowArrayEncoding {
    fn id(&self) -> EncodingId {
        todo!()
    }
}

impl ComputeVTable for ArrowArrayEncoding {}

#[derive(Clone, Debug)]
pub struct ArrowArray {
    inner: ArrowArrayRef,
    dtype: DType,
}

impl ArrowArray {
    pub fn new(arrow_array: ArrowArrayRef, nullability: Nullability) -> Self {
        let dtype = DType::from_arrow((arrow_array.data_type(), nullability));
        Self {
            inner: arrow_array,
            dtype,
        }
    }

    pub fn inner(&self) -> &ArrowArrayRef {
        &self.inner
    }
}

impl ArrayCanonicalImpl for ArrowArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        todo!()
    }
}

impl ArrayStatisticsImpl for ArrowArray {
    fn _stats_ref(&self) -> StatsSetRef<'_> {
        todo!()
    }
}

impl ArrayValidityImpl for ArrowArray {
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        Ok(self.inner.is_valid(index))
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        Ok(self.inner.logical_null_count() == 0)
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        Ok(self.inner.logical_null_count() == self.inner.len())
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        Ok(self
            .inner
            .logical_nulls()
            .map(|null_buffer| Mask::from_buffer(null_buffer.inner().clone()))
            .unwrap_or_else(|| Mask::new_true(self.inner.len())))
    }
}

impl ArrayVariantsImpl for ArrowArray {}

impl ArrayVisitorImpl<EmptyMetadata> for ArrowArray {
    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}

impl ArrayImpl for ArrowArray {
    type Encoding = ArrowArrayEncoding;

    fn _len(&self) -> usize {
        self.inner.len()
    }

    fn _dtype(&self) -> &DType {
        &self.dtype
    }

    fn _vtable(&self) -> VTableRef {
        ArcRef::new_ref(&ArrowArrayEncoding)
    }

    fn _with_children(&self, _children: &[ArrayRef]) -> VortexResult<Self> {
        Ok(self.clone())
    }
}
