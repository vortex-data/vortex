// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::Python;
use vortex::buffer::ByteBuffer;
use vortex::compute::{ComputeFn, InvocationArgs, Output};
use vortex::dtype::DType;
use vortex::error::{VortexExpect, VortexResult};
use vortex::mask::Mask;
use vortex::scalar::Scalar;
use vortex::serde::ArrayChildren;
use vortex::stats::StatsSetRef;
use vortex::vtable::{
    ArrayVTable, CanonicalVTable, ComputeVTable, EncodeVTable, NotSupported, OperationsVTable,
    SerdeVTable, VTable, ValidityVTable, VisitorVTable,
};
use vortex::{
    vtable, ArrayBufferVisitor, ArrayChildVisitor, ArrayRef, Canonical, DeserializeMetadata,
    EncodingId, EncodingRef, RawMetadata,
};

use crate::arrays::py::{PythonArray, PythonEncoding};

vtable!(Python);

impl VTable for PythonVTable {
    type Array = PythonArray;
    type Encoding = PythonEncoding;
    type Metadata = RawMetadata;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = Self;
    type EncodeVTable = Self;
    type SerdeVTable = Self;
    type PipelineVTable = NotSupported;

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
    fn canonicalize(_array: &PythonArray) -> Canonical {
        todo!()
    }
}

impl OperationsVTable<PythonVTable> for PythonVTable {
    fn slice(_array: &PythonArray, _range: Range<usize>) -> ArrayRef {
        todo!()
    }

    fn scalar_at(_array: &PythonArray, _index: usize) -> Scalar {
        todo!()
    }
}

impl ValidityVTable<PythonVTable> for PythonVTable {
    fn is_valid(_array: &PythonArray, _index: usize) -> bool {
        todo!()
    }

    fn all_valid(_array: &PythonArray) -> bool {
        todo!()
    }

    fn all_invalid(_array: &PythonArray) -> bool {
        todo!()
    }

    fn validity_mask(_array: &PythonArray) -> Mask {
        todo!()
    }
}

impl VisitorVTable<PythonVTable> for PythonVTable {
    fn metadata(array: &PythonArray) -> <PythonVTable as VTable>::Metadata {
        // TODO(ngates): metadata should be extracted in the constructor and stored in the array
        // so it can be returned here without error. For now we use expect which will panic
        // if the Python call fails.
        Python::attach(|py| {
            let obj = array.object.bind(py);

            let bytes = obj
                .call_method("__vx_metadata__", (), None)
                .vortex_expect("Failed to call __vx_metadata__")
                .downcast::<PyBytes>()
                .vortex_expect("Expected array metadata to be Python bytes")
                .as_bytes()
                .to_vec();

            RawMetadata(bytes)
        })
    }

    fn visit_buffers(_array: &PythonArray, _visitor: &mut dyn ArrayBufferVisitor) {
        todo!()
    }

    fn visit_children(_array: &PythonArray, _visitor: &mut dyn ArrayChildVisitor) {
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
        _like: Option<&PythonArray>,
    ) -> VortexResult<Option<PythonArray>> {
        todo!()
    }
}

impl SerdeVTable<PythonVTable> for PythonVTable {
    fn build(
        _encoding: &PythonEncoding,
        _dtype: &DType,
        _len: usize,
        _metadata: &<<PythonVTable as VTable>::Metadata as DeserializeMetadata>::Output,
        _buffers: &[ByteBuffer],
        _children: &dyn ArrayChildren,
    ) -> VortexResult<PythonArray> {
        todo!()
    }
}
