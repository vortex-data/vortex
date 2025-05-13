use vortex::buffer::ByteBuffer;
use vortex::compute::{ComputeFn, InvocationArgs, Output};
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::mask::Mask;
use vortex::scalar::Scalar;
use vortex::serde::ArrayParts;
use vortex::stats::StatsSetRef;
use vortex::vtable::{
    ArrayVTable, CanonicalVTable, ComputeVTable, EncodeVTable, OperationsVTable, SerdeVTable,
    VTable, ValidityVTable, VisitorVTable,
};
use vortex::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayContext, ArrayRef, Canonical,
    DeserializeMetadata, EmptyMetadata, EncodingId, EncodingRef, vtable,
};

use crate::arrays::py::{PythonArray, PythonEncoding};

vtable!(Python);

impl VTable for PythonVTable {
    type Array = PythonArray;
    type Encoding = PythonEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = Self;
    type EncodeVTable = Self;
    type SerdeVTable = Self;

    fn id(encoding: &Self::Encoding) -> EncodingId {
        encoding.id.clone()
    }

    fn encoding(array: &Self::Array) -> EncodingRef {
        array.encoding.clone()
    }
}

impl ArrayVTable<PythonVTable> for PythonVTable {
    fn len(array: &PythonArray) -> usize {
        array.len
    }

    fn dtype(array: &PythonArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &PythonArray) -> StatsSetRef<'_> {
        array.stats.to_ref(array.as_ref())
    }
}

impl CanonicalVTable<PythonVTable> for PythonVTable {
    fn canonicalize(_array: &PythonArray) -> VortexResult<Canonical> {
        todo!()
    }
}

impl OperationsVTable<PythonVTable> for PythonVTable {
    fn slice(_array: &PythonArray, _start: usize, _stop: usize) -> VortexResult<ArrayRef> {
        todo!()
    }

    fn scalar_at(_array: &PythonArray, _index: usize) -> VortexResult<Scalar> {
        todo!()
    }
}

impl ValidityVTable<PythonVTable> for PythonVTable {
    fn is_valid(_array: &PythonArray, _index: usize) -> VortexResult<bool> {
        todo!()
    }

    fn all_valid(_array: &PythonArray) -> VortexResult<bool> {
        todo!()
    }

    fn all_invalid(_array: &PythonArray) -> VortexResult<bool> {
        todo!()
    }

    fn validity_mask(_array: &PythonArray) -> VortexResult<Mask> {
        todo!()
    }
}

impl VisitorVTable<PythonVTable> for PythonVTable {
    fn visit_buffers(_array: &PythonArray, _visitor: &mut dyn ArrayBufferVisitor) {
        todo!()
    }

    fn visit_children(_array: &PythonArray, _visitor: &mut dyn ArrayChildVisitor) {
        todo!()
    }

    fn with_children(_array: &PythonArray, _children: &[ArrayRef]) -> VortexResult<PythonArray> {
        todo!()
    }
}

impl ComputeVTable<PythonVTable> for PythonVTable {
    fn invoke(
        _array: &PythonArray,
        _compute_fn: &ComputeFn,
        _args: &InvocationArgs,
    ) -> VortexResult<Option<Output>> {
        todo!()
    }
}

impl EncodeVTable<PythonVTable> for PythonVTable {
    fn encode(
        _encoding: &PythonEncoding,
        _canonical: &Canonical,
        _like: Option<&dyn Array>,
    ) -> VortexResult<Option<PythonArray>> {
        todo!()
    }
}

impl SerdeVTable<PythonVTable> for PythonVTable {
    type Metadata = EmptyMetadata;

    fn metadata(_array: &PythonArray) -> Option<Self::Metadata> {
        todo!()
    }

    fn decode(
        _encoding: &PythonEncoding,
        _dtype: DType,
        _len: usize,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _buffers: &[ByteBuffer],
        _children: &[ArrayParts],
        _ctx: &ArrayContext,
    ) -> VortexResult<PythonArray> {
        todo!()
    }
}
