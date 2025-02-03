use pyo3::prelude::PyDictMethods;
use pyo3::types::{PyDict, PyList};
use pyo3::{IntoPy, PyObject, Python};
use vortex::buffer::{BufferString, ByteBuffer};
use vortex::dtype::half::f16;
use vortex::dtype::{DType, PType};
use vortex::error::{vortex_err, VortexExpect};
use vortex::scalar::{ListScalar, Scalar, StructScalar};

use crate::PyVortex;

impl IntoPy<PyObject> for PyVortex<&'_ Scalar> {
    fn into_py(self, py: Python) -> PyObject {
        match self.0.dtype() {
            DType::Null => py.None(),
            DType::Bool(_) => self.0.as_bool().value().into_py(py),
            DType::Primitive(ptype, ..) => {
                let p = self.0.as_primitive();
                match ptype {
                    PType::U8 => p.typed_value::<u8>().into_py(py),
                    PType::U16 => p.typed_value::<u16>().into_py(py),
                    PType::U32 => p.typed_value::<u32>().into_py(py),
                    PType::U64 => p.typed_value::<u64>().into_py(py),
                    PType::I8 => p.typed_value::<i8>().into_py(py),
                    PType::I16 => p.typed_value::<i16>().into_py(py),
                    PType::I32 => p.typed_value::<i32>().into_py(py),
                    PType::I64 => p.typed_value::<i64>().into_py(py),
                    PType::F16 => p.typed_value::<f16>().map(f16::to_f32).into_py(py),
                    PType::F32 => p.typed_value::<f32>().into_py(py),
                    PType::F64 => p.typed_value::<f64>().into_py(py),
                }
            }
            DType::Utf8(_) => self.0.as_utf8().value().map(PyVortex).into_py(py),
            DType::Binary(_) => self.0.as_binary().value().map(PyVortex).into_py(py),
            DType::Struct(..) => PyVortex(self.0.as_struct()).into_py(py),
            DType::List(..) => PyVortex(self.0.as_list()).into_py(py),
            DType::Extension(_) => PyVortex(&self.0.as_extension().storage()).into_py(py),
        }
    }
}

impl IntoPy<PyObject> for PyVortex<BufferString> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.0.as_str().into_py(py)
    }
}

impl IntoPy<PyObject> for PyVortex<ByteBuffer> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.0.as_slice().into_py(py)
    }
}

impl IntoPy<PyObject> for PyVortex<StructScalar<'_>> {
    fn into_py(self, py: Python) -> PyObject {
        let Some(fields) = self.0.fields() else {
            return py.None();
        };

        let dict = PyDict::new_bound(py);
        for (child, name) in fields.iter().zip(self.0.struct_dtype().names().iter()) {
            dict.set_item(name.to_string(), PyVortex(child).into_py(py))
                .map_err(|e| vortex_err!("Failed to set item in dictionary {}", e))
                .vortex_expect("Failed to set item in dictionary");
        }
        dict.into_py(py)
    }
}

impl IntoPy<PyObject> for PyVortex<ListScalar<'_>> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let Some(elements) = self.0.elements() else {
            return py.None();
        };

        PyList::new_bound(
            py,
            elements
                .into_iter()
                .map(|child| PyVortex(&child).into_py(py)),
        )
        .into_py(py)
    }
}
