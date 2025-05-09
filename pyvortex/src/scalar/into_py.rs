use std::cmp::Ordering;

use pyo3::prelude::{PyAnyMethods, PyDictMethods};
use pyo3::types::{PyBytes, PyDict, PyList, PyString};
use pyo3::{Bound, IntoPyObject, PyAny, PyErr, PyResult, Python};
use vortex::buffer::{BufferString, ByteBuffer};
use vortex::dtype::half::f16;
use vortex::dtype::{DType, PType};
use vortex::error::{VortexExpect, vortex_err};
use vortex::match_each_decimal_value;
use vortex::scalar::{DecimalValue, ListScalar, Scalar, StructScalar, i256};

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
            DType::Decimal(decimal_type, ..) => {
                let decimal = self.0.as_decimal();
                match decimal.decimal_value() {
                    None => Ok(py.None().into_pyobject(py)?),
                    Some(value) => decimal_value_to_py(py, decimal_type.scale(), *value),
                }
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
        for (child, name) in fields.iter().zip(self.0.names().iter()) {
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

trait DecimalIntoParts: Sized {
    /// Split an integer encoding a decimal with the given `scale` into a
    /// (whole number, decimal) parts.
    ///
    /// For example, for the number 123i128 and scale 2, this will return returns (1, 23).
    fn decimal_parts(self, scale: i8) -> (Self, Self);
}

macro_rules! impl_decimal_into_parts {
    ($ty:ident, $ten:expr) => {
        impl DecimalIntoParts for $ty {
            fn decimal_parts(self, scale: i8) -> (Self, Self) {
                let scale_factor = $ten.pow(scale.unsigned_abs() as u32);
                match scale.cmp(&0) {
                    Ordering::Equal => (self, 0),
                    Ordering::Less => {
                        // Negative scale -> apply the given number of trailing zeros
                        (self * scale_factor, 0)
                    }
                    Ordering::Greater => {
                        // Positive scale -> extract the leading/trailing digits separately.
                        (self / scale_factor, self % scale_factor)
                    }
                }
            }
        }
    };
}

impl_decimal_into_parts!(i8, 10i8);
impl_decimal_into_parts!(i16, 10i16);
impl_decimal_into_parts!(i32, 10i32);
impl_decimal_into_parts!(i64, 10i64);
impl_decimal_into_parts!(i128, 10i128);

impl DecimalIntoParts for i256 {
    fn decimal_parts(self, scale: i8) -> (Self, Self) {
        match scale.cmp(&0) {
            Ordering::Equal => (self, i256::ZERO),
            Ordering::Less => {
                // Negative scale -> apply the given number of trailing zeros
                let scale_factor = i256::from_i128(10).wrapping_pow(-scale as u32);
                (self * scale_factor, i256::ZERO)
            }
            Ordering::Greater => {
                // Positive scale -> extract the leading/trailing digits separately.
                let scale_factor = i256::from_i128(10).wrapping_pow(scale as u32);
                (self / scale_factor, self % scale_factor)
            }
        }
    }
}

fn decimal_value_to_py(
    py: Python,
    scale: i8,
    decimal_value: DecimalValue,
) -> PyResult<Bound<PyAny>> {
    let m = py.import("decimal")?;
    let decimal_class = m.getattr("Decimal")?;

    match_each_decimal_value!(decimal_value, |$V| {
       {
            let (whole, decimal) = $V.decimal_parts(scale);
            let repr = format!("{}.{:0>width$}", whole, decimal, width = scale as usize)
                .into_pyobject(py)?;
            decimal_class.call1((repr,))
        }
    })
}
