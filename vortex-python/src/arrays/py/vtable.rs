// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::sync::Arc;

use pyo3::Python;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyType;
use vortex::array::ArrayBufferVisitor;
use vortex::array::ArrayChildVisitor;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::Precision;
use vortex::array::RawMetadata;
use vortex::array::SerializeMetadata;
use vortex::array::buffer::BufferHandle;
use vortex::array::serde::ArrayChildren;
use vortex::array::stats::StatsSetRef;
use vortex::array::validity::Validity;
use vortex::array::vtable;
use vortex::array::vtable::ArrayId;
use vortex::array::vtable::ArrayVTable;
use vortex::array::vtable::BaseArrayVTable;
use vortex::array::vtable::ComputeVTable;
use vortex::array::vtable::OperationsVTable;
use vortex::array::vtable::VTable;
use vortex::array::vtable::ValidityVTable;
use vortex::array::vtable::VisitorVTable;
use vortex::compute::ComputeFn;
use vortex::compute::InvocationArgs;
use vortex::compute::Output;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::mask::Mask;
use vortex::scalar::Scalar;

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
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = Self;

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

    fn with_children(_array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.is_empty(),
            "PythonArray has no children, got {}",
            children.len()
        );
        Ok(())
    }

    fn execute(_array: &Self::Array, _ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
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

impl OperationsVTable<PythonVTable> for PythonVTable {
    fn scalar_at(_array: &PythonArray, _index: usize) -> Scalar {
        todo!()
    }
}

impl ValidityVTable<PythonVTable> for PythonVTable {
    fn validity(_array: &PythonArray) -> VortexResult<Validity> {
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
