use pyo3::prelude::PyDictMethods;
use pyo3::types::{PyBytes, PyDict, PyList, PyString};
use pyo3::{Bound, IntoPyObject, PyAny, PyErr, Python};
use vortex::buffer::{BufferString, ByteBuffer};
use vortex::dtype::half::f16;
use vortex::dtype::{DType, PType};
use vortex::error::{VortexExpect, vortex_err};
use vortex::scalar::{ListScalar, Scalar, StructScalar};

use crate::PyVortex;

impl<'py> IntoPyObject<'py> for PyVortex<&'_ Scalar> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self.0.dtype() {
            DType::Null => Ok(py.None().into_pyobject(py)?),
            DType::Bool(_) => Ok(self.0.as_bool().value().as_ref().into_pyobject(py)?),
            DType::Primitive(ptype, ..) => {
                let p = self.0.as_primitive();
                let primitive_py = match ptype {
                    PType::U8 => p.typed_value::<u8>().into_pyobject(py),
                    PType::U16 => p.typed_value::<u16>().into_pyobject(py),
                    PType::U32 => p.typed_value::<u32>().into_pyobject(py),
                    PType::U64 => p.typed_value::<u64>().into_pyobject(py),
                    PType::I8 => p.typed_value::<i8>().into_pyobject(py),
                    PType::I16 => p.typed_value::<i16>().into_pyobject(py),
                    PType::I32 => p.typed_value::<i32>().into_pyobject(py),
                    PType::I64 => p.typed_value::<i64>().into_pyobject(py),
                    PType::F16 => p.typed_value::<f16>().map(f16::to_f32).into_pyobject(py),
                    PType::F32 => p.typed_value::<f32>().into_pyobject(py),
                    PType::F64 => p.typed_value::<f64>().into_pyobject(py),
                };

                primitive_py.map_err(PyErr::from)
            }
            DType::Utf8(_) => self.0.as_utf8().value().map(PyVortex).into_pyobject(py),
            DType::Binary(_) => self.0.as_binary().value().map(PyVortex).into_pyobject(py),
            DType::Struct(..) => PyVortex(self.0.as_struct()).into_pyobject(py),
            DType::List(..) => PyVortex(self.0.as_list()).into_pyobject(py),
            DType::Extension(_) => PyVortex(&self.0.as_extension().storage()).into_pyobject(py),
        }
    }
}

impl<'py> IntoPyObject<'py> for PyVortex<BufferString> {
    type Target = PyString;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0.as_str().into_pyobject(py)?)
    }
}

impl<'py> IntoPyObject<'py> for PyVortex<ByteBuffer> {
    type Target = PyBytes;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(PyBytes::new(py, self.0.as_slice()))
    }
}

impl<'py> IntoPyObject<'py> for PyVortex<StructScalar<'_>> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let Some(fields) = self.0.fields() else {
            return Ok(py.None().into_pyobject(py)?);
        };

        let dict = PyDict::new(py);
        for (child, name) in fields.iter().zip(self.0.struct_dtype().names().iter()) {
            dict.set_item(name.to_string(), PyVortex(child).into_pyobject(py)?)
                .map_err(|e| vortex_err!("Failed to set item in dictionary {}", e))
                .vortex_expect("Failed to set item in dictionary");
        }
        Ok(dict.into_pyobject(py)?.into_any())
    }
}

impl<'py> IntoPyObject<'py> for PyVortex<ListScalar<'_>> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let Some(elements) = self.0.elements() else {
            return Ok(py.None().into_pyobject(py)?);
        };

        PyList::new(py, elements.iter().map(PyVortex)).map(|l| l.into_any())
    }
}
