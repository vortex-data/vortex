// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;

use arrow_array::ArrayRef as ArrowArrayRef;
use vortex_buffer::BitBuffer;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::arrow::FromArrowType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::Array;
use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::ArrayRef;
use crate::Canonical;
use crate::EmptyMetadata;
use crate::IntoArray;
use crate::Precision;
use crate::arrays::BoolArray;
use crate::arrow::FromArrowArray;
use crate::buffer::BufferHandle;
use crate::serde::ArrayChildren;
use crate::stats::ArrayStats;
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
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
    ) -> VortexResult<Self::Array> {
        vortex_bail!("ArrowArray cannot be deserialized")
    }

    fn with_children(_array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.is_empty(),
            "ArrowArray has no children, got {}",
            children.len()
        );
        Ok(())
    }
}

/// A Vortex array that wraps an in-memory Arrow array.
// TODO(ngates): consider having each Arrow encoding be a separate encoding ID.
#[derive(Debug)]
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

    fn validity(array: &ArrowArray) -> VortexResult<Validity> {
        Ok(match array.inner.logical_nulls() {
            None => Validity::AllValid,
            Some(null_buffer) => match null_buffer.null_count() {
                0 => Validity::AllValid,
                n if n == array.inner.len() => Validity::AllInvalid,
                _ => Validity::Array(
                    BoolArray::new(
                        BitBuffer::from(null_buffer.inner().clone()),
                        Validity::NonNullable,
                    )
                    .into_array(),
                ),
            },
        })
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
