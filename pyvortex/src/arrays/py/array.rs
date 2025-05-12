use std::sync::Arc;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::{Bound, FromPyObject, Py, PyAny, PyResult};
use vortex::dtype::DType;
use vortex::error::{VortexError, VortexResult};
use vortex::mask::Mask;
use vortex::scalar::Scalar;
use vortex::stats::{ArrayStats, StatsSetRef};
use vortex::vtable::{
    ArrayVTable, CanonicalVTable, OperationsVTable, ValidityVTable, VisitorVTable,
};
use vortex::{ArrayBufferVisitor, ArrayChildVisitor, ArrayRef, Canonical, EncodingRef};

use crate::arrays::py::{PythonEncoding, PythonVTable};
use crate::dtype::PyDType;

/// Wrapper struct encapsulating a Vortex array implemented using a Python object.
///
/// The user-code object is expected to subclass the abstract base class `vx.PyArray` which
/// will ensure the object implements the necessary methods.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PythonArray {
    obj: Arc<Py<PyAny>>,
    encoding: EncodingRef,
    len: usize,
    dtype: DType,
    stats: ArrayStats,
}

impl PythonArray {
    pub(super) fn encoding(&self) -> EncodingRef {
        self.encoding.clone()
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

impl<'py> FromPyObject<'py> for PythonArray {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = ob.py();

        // Check if the object is a subclass of `vx.PyArray`.
        let pyarray_cls = py.import("vortex").and_then(|m| m.getattr("PyArray"))?;
        if !ob.is_instance(&pyarray_cls)? {
            return Err(PyTypeError::new_err("Expected a subclass of `vx.PyArray`"));
        }

        // Extract the length and dtype from the object.
        let len = ob.len()?;
        let dtype = ob
            .getattr("dtype")
            .map_err(|_| PyValueError::new_err("Missing `dtype` property"))?
            .extract::<PyDType>()?
            .into_inner();

        // Use the Python class as the encoding VTable.
        let cls = PythonEncoding::extract_bound(&ob.get_type())?;

        Ok(Self {
            obj: Arc::new(ob.clone().unbind()),
            encoding: cls.to_encoding(),
            len,
            dtype,
            // TODO(ngates): do stats need to be held in the Python object?
            stats: Default::default(),
        })
    }
}

impl<'py> IntoPyObject<'py> for PythonArray {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = VortexError;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.obj.as_ref().bind(py).to_owned())
    }
}
