use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::serde::ArrayChildren;
use crate::stats::{ArrayStats, StatsSetRef};
use crate::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, OperationsVTable, SerdeVTable, VTable,
    ValidityVTable, VisitorVTable,
};
use crate::{
    ArrayBufferVisitor, ArrayChildVisitor, ArrayRef, Canonical, EmptyMetadata, EncodingId,
    EncodingRef, IntoArray, vtable,
};

mod compute;

vtable!(Null);

impl VTable for NullVTable {
    type Array = NullArray;
    type Encoding = NullEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.null")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(NullEncoding.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct NullArray {
    len: usize,
    stats_set: ArrayStats,
}

#[derive(Clone, Debug)]
pub struct NullEncoding;

impl NullArray {
    pub fn new(len: usize) -> Self {
        Self {
            len,
            stats_set: Default::default(),
        }
    }
}

impl ArrayVTable<NullVTable> for NullVTable {
    fn len(array: &NullArray) -> usize {
        array.len
    }

    fn dtype(_array: &NullArray) -> &DType {
        &DType::Null
    }

    fn stats(array: &NullArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

impl SerdeVTable<NullVTable> for NullVTable {
    type Metadata = EmptyMetadata;

    fn metadata(_array: &NullArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(EmptyMetadata))
    }

    fn build(
        _encoding: &NullEncoding,
        _dtype: &DType,
        len: usize,
        _metadata: &Self::Metadata,
        _buffers: &[ByteBuffer],
        _children: &dyn ArrayChildren,
    ) -> VortexResult<NullArray> {
        Ok(NullArray::new(len))
    }
}

impl VisitorVTable<NullVTable> for NullVTable {
    fn visit_buffers(_array: &NullArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(_array: &NullArray, _visitor: &mut dyn ArrayChildVisitor) {}
}

impl CanonicalVTable<NullVTable> for NullVTable {
    fn canonicalize(array: &NullArray) -> VortexResult<Canonical> {
        Ok(Canonical::Null(array.clone()))
    }
}

impl OperationsVTable<NullVTable> for NullVTable {
    fn slice(_array: &NullArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        Ok(NullArray::new(stop - start).into_array())
    }

    fn scalar_at(_array: &NullArray, _index: usize) -> VortexResult<Scalar> {
        Ok(Scalar::null(DType::Null))
    }
}

impl ValidityVTable<NullVTable> for NullVTable {
    fn is_valid(_array: &NullArray, _index: usize) -> VortexResult<bool> {
        Ok(false)
    }

    fn all_valid(array: &NullArray) -> VortexResult<bool> {
        Ok(array.is_empty())
    }

    fn all_invalid(array: &NullArray) -> VortexResult<bool> {
        Ok(!array.is_empty())
    }

    fn validity_mask(array: &NullArray) -> VortexResult<Mask> {
        Ok(Mask::AllFalse(array.len))
    }
}
