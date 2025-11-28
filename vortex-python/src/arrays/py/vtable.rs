// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::ops::Range;
use std::sync::Arc;

use pyo3::Python;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyType;
use vortex::ArrayBufferVisitor;
use vortex::ArrayChildVisitor;
use vortex::ArrayRef;
use vortex::Canonical;
use vortex::Precision;
use vortex::RawMetadata;
use vortex::SerializeMetadata;
use vortex::buffer::BufferHandle;
use vortex::compute::ComputeFn;
use vortex::compute::InvocationArgs;
use vortex::compute::Output;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::mask::Mask;
use vortex::scalar::Scalar;
use vortex::serde::ArrayChildren;
use vortex::stats::StatsSetRef;
use vortex::vtable;
use vortex::vtable::ArrayId;
use vortex::vtable::ArrayVTable;
use vortex::vtable::BaseArrayVTable;
use vortex::vtable::CanonicalVTable;
use vortex::vtable::ComputeVTable;
use vortex::vtable::EncodeVTable;
use vortex::vtable::NotSupported;
use vortex::vtable::OperationsVTable;
use vortex::vtable::VTable;
use vortex::vtable::ValidityVTable;
use vortex::vtable::VisitorVTable;

use crate::arrays::py::PythonArray;

vtable!(Python);

/// Wrapper struct encapsulating a Python encoding.
#[allow(dead_code)]
#[derive(Debug)]
pub struct PythonVTable {
    pub(super) id: ArrayId,
    pub(super) cls: Py<PyType>,
}

/// Convert a Python class into a [`PythonVTable`].
impl<'py> FromPyObject<'_, 'py> for PythonVTable {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        let cls = ob.cast::<PyType>()?;

        let id = ArrayId::new_arc(
            cls.getattr("id")
                .map_err(|_| {
                    PyValueError::new_err(format!(
                        "PyEncoding subclass {cls:?} must have an 'id' attribute"
                    ))
                })?
                .extract::<String>()
                .map_err(|_| PyValueError::new_err("'id' attribute must be a string"))?
                .into(),
        );

        Ok(PythonVTable {
            id,
            cls: cls.to_owned().unbind(),
        })
    }
}

impl VTable for PythonVTable {
    type Array = PythonArray;

    type Metadata = RawMetadata;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = Self;
    type EncodeVTable = Self;
    type OperatorVTable = NotSupported;

    fn id(&self) -> ArrayId {
        self.id.clone()
    }

    fn encoding(array: &Self::Array) -> ArrayVTable {
        array.vtable.clone()
    }

    fn metadata(array: &PythonArray) -> VortexResult<Self::Metadata> {
        Python::attach(|py| {
            let obj = array.object.bind(py);
            if !obj.hasattr(intern!(py, "metadata"))? {
                // The class does not have a metadata attribute so does not support serialization.
                return Ok(RawMetadata(vec![]));
            }

            let bytes = obj
                .call_method("__vx_metadata__", (), None)?
                .cast::<PyBytes>()
                .map_err(|_| vortex_err!("Expected array metadata to be Python bytes"))?
                .as_bytes()
                .to_vec();

            Ok(RawMetadata(bytes))
        })
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
    }

    fn deserialize(bytes: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(RawMetadata(bytes.to_vec()))
    }

    fn build(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
    ) -> VortexResult<PythonArray> {
        todo!()
    }
}

impl BaseArrayVTable<PythonVTable> for PythonVTable {
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
        Arc::as_ptr(&array.object).hash(state);
        array.vtable.id().hash(state);
        array.len.hash(state);
        array.dtype.hash(state);
    }

    fn array_eq(array: &PythonArray, other: &PythonArray, _precision: Precision) -> bool {
        Arc::ptr_eq(&array.object, &other.object)
            && array.vtable == other.vtable
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
        _vtable: &PythonVTable,
        _canonical: &Canonical,
        _like: Option<&PythonArray>,
    ) -> VortexResult<Option<PythonArray>> {
        todo!()
    }
}
