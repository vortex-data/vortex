// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::ops::Range;

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::{Python, intern};
use vortex::buffer::ByteBuffer;
use vortex::compute::{ComputeFn, InvocationArgs, Output};
use vortex::dtype::DType;
use vortex::error::{VortexResult, vortex_err};
use vortex::mask::Mask;
use vortex::scalar::Scalar;
use vortex::serde::ArrayChildren;
use vortex::stats::StatsSetRef;
use vortex::vtable::{
    ArrayVTable, CanonicalVTable, ComputeVTable, EncodeVTable, NotSupported, OperationsVTable,
    SerdeVTable, VTable, ValidityVTable, VisitorVTable,
};
use vortex::{
    ArrayBufferVisitor, ArrayChildVisitor, ArrayRef, Canonical, DeserializeMetadata, EncodingId,
    EncodingRef, Precision, RawMetadata, vtable,
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

    fn array_hash<H: std::hash::Hasher>(array: &PythonArray, state: &mut H, _precision: Precision) {
        std::sync::Arc::as_ptr(&array.object).hash(state);
        array.encoding.id().hash(state);
        array.len.hash(state);
        array.dtype.hash(state);
    }

    fn array_eq(array: &PythonArray, other: &PythonArray, _precision: Precision) -> bool {
        std::sync::Arc::ptr_eq(&array.object, &other.object)
            && array.encoding == other.encoding
            && array.len == other.len
            && array.dtype == other.dtype
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
    type Metadata = RawMetadata;

    fn metadata(array: &PythonArray) -> VortexResult<Option<Self::Metadata>> {
        Python::attach(|py| {
            let obj = array.object.bind(py);
            if !obj.hasattr(intern!(py, "metadata"))? {
                // The class does not have a metadata attribute so does not support serialization.
                return Ok(None);
            }

            let bytes = obj
                .call_method("__vx_metadata__", (), None)?
                .downcast::<PyBytes>()
                .map_err(|_| vortex_err!("Expected array metadata to be Python bytes"))?
                .as_bytes()
                .to_vec();

            Ok(Some(RawMetadata(bytes)))
        })
    }

    fn build(
        _encoding: &PythonEncoding,
        _dtype: &DType,
        _len: usize,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _buffers: &[ByteBuffer],
        _children: &dyn ArrayChildren,
    ) -> VortexResult<PythonArray> {
        todo!()
    }
}
