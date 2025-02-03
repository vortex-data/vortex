mod chunked;
mod constant;
mod struct_;

pub(crate) use chunked::*;
pub(crate) use constant::*;
use pyo3::prelude::*;
pub(crate) use struct_::*;
use vortex::array::{
    BoolEncoding, ExtensionEncoding, ListEncoding, NullEncoding, PrimitiveEncoding, VarBinEncoding,
    VarBinViewEncoding,
};
use vortex::encodings::bytebool::ByteBoolEncoding;

use crate::arrays::{EncodingSubclass, PyArray};

/// Concrete class for arrays with `vortex.null` encoding.
#[pyclass(name = "NullEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyNullEncoding;

impl EncodingSubclass for PyNullEncoding {
    type Encoding = NullEncoding;
}

#[pymethods]
impl PyNullEncoding {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, PyNullEncoding)
    }
}

/// Concrete class for arrays with `vortex.bool` encoding.
#[pyclass(name = "BoolEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyBoolEncoding;

impl EncodingSubclass for PyBoolEncoding {
    type Encoding = BoolEncoding;
}

#[pymethods]
impl PyBoolEncoding {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, PyBoolEncoding)
    }
}

/// Concrete class for arrays with `vortex.bytebool` encoding.
#[pyclass(name = "ByteBoolEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyByteBoolEncoding;

impl EncodingSubclass for PyByteBoolEncoding {
    type Encoding = ByteBoolEncoding;
}

#[pymethods]
impl PyByteBoolEncoding {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, PyByteBoolEncoding)
    }
}

/// Concrete class for arrays with `vortex.primitive` encoding.
#[pyclass(name = "PrimitiveEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyPrimitiveEncoding;

impl EncodingSubclass for PyPrimitiveEncoding {
    type Encoding = PrimitiveEncoding;
}

#[pymethods]
impl PyPrimitiveEncoding {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, PyPrimitiveEncoding)
    }
}

/// Concrete class for arrays with `vortex.varbin` encoding.
#[pyclass(name = "VarBinEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyVarBinEncoding;

impl EncodingSubclass for PyVarBinEncoding {
    type Encoding = VarBinEncoding;
}

#[pymethods]
impl PyVarBinEncoding {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, PyVarBinEncoding)
    }
}

/// Concrete class for arrays with `vortex.varbinview` encoding.
#[pyclass(name = "VarBinViewEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyVarBinViewEncoding;

impl EncodingSubclass for PyVarBinViewEncoding {
    type Encoding = VarBinViewEncoding;
}

#[pymethods]
impl PyVarBinViewEncoding {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, PyVarBinViewEncoding)
    }
}

/// Concrete class for arrays with `vortex.list` encoding.
#[pyclass(name = "ListEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyListEncoding;

impl EncodingSubclass for PyListEncoding {
    type Encoding = ListEncoding;
}

#[pymethods]
impl PyListEncoding {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, PyListEncoding)
    }
}

/// Concrete class for arrays with `vortex.ext` encoding.
#[pyclass(name = "ExtensionEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyExtensionEncoding;

impl EncodingSubclass for PyExtensionEncoding {
    type Encoding = ExtensionEncoding;
}

#[pymethods]
impl PyExtensionEncoding {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, PyExtensionEncoding)
    }
}
