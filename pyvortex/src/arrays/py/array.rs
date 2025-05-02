use std::sync::Arc;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::{Bound, FromPyObject, Py, PyAny, PyResult};
use vortex::arcref::ArcRef;
use vortex::dtype::DType;
use vortex::error::{VortexError, VortexResult};
use vortex::mask::Mask;
use vortex::stats::StatsSetRef;
use vortex::vtable::VTableRef;
use vortex::{
    ArrayCanonicalImpl, ArrayImpl, ArrayOperationsImpl, ArrayRef, ArrayStatisticsImpl,
    ArrayValidityImpl, ArrayVariantsImpl, ArrayVisitorImpl, Canonical, EmptyMetadata,
};

use crate::arrays::py::PyEncodingClass;
use crate::dtype::PyDType;

/// Wrapper struct encapsulating a Vortex array implemented using a Python object.
///
/// The user-code object is expected to subclass the abstract base class `vx.PyArray` which
/// will ensure the object implements the necessary methods.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PyArrayInstance {
    obj: Arc<Py<PyAny>>,
    cls: VTableRef,
    len: usize,
    dtype: DType,
}

impl ArrayImpl for PyArrayInstance {
    type Encoding = PyEncodingClass;

    fn _len(&self) -> usize {
        self.len
    }

    fn _dtype(&self) -> &DType {
        &self.dtype
    }

    fn _vtable(&self) -> VTableRef {
        self.cls.clone()
    }

    fn _with_children(&self, _children: &[ArrayRef]) -> VortexResult<Self> {
        todo!()
    }
}

impl ArrayCanonicalImpl for PyArrayInstance {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        todo!()
    }
}

impl ArrayOperationsImpl for PyArrayInstance {
    fn _slice(&self, _start: usize, _stop: usize) -> VortexResult<ArrayRef> {
        todo!()
    }
}

impl ArrayStatisticsImpl for PyArrayInstance {
    fn _stats_ref(&self) -> StatsSetRef<'_> {
        todo!()
    }
}

impl ArrayValidityImpl for PyArrayInstance {
    fn _is_valid(&self, _index: usize) -> VortexResult<bool> {
        todo!()
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        todo!()
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        todo!()
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        todo!()
    }
}

impl ArrayVariantsImpl for PyArrayInstance {}

impl ArrayVisitorImpl for PyArrayInstance {
    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}

impl<'py> FromPyObject<'py> for PyArrayInstance {
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
        let cls = PyEncodingClass::extract_bound(&ob.get_type())?;

        Ok(Self {
            obj: Arc::new(ob.clone().unbind()),
            cls: ArcRef::new_arc(Arc::new(cls) as _),
            len,
            dtype,
        })
    }
}

impl<'py> IntoPyObject<'py> for PyArrayInstance {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = VortexError;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.obj.as_ref().bind(py).to_owned())
    }
}
