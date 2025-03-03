mod chunked;
mod constant;
mod struct_;

pub(crate) use chunked::*;
pub(crate) use constant::*;
use pyo3::prelude::*;
pub(crate) use struct_::*;
use vortex::arrays::{
    BoolEncoding, ExtensionEncoding, ListEncoding, NullEncoding, PrimitiveEncoding, VarBinEncoding,
    VarBinViewEncoding,
};
use vortex::encodings::bytebool::ByteBoolEncoding;

use crate::arrays::{EncodingSubclass, PyArray};

/// Concrete class for arrays with `vortex.null` encoding.
#[pyclass(name = "NullArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyNullArray;

impl EncodingSubclass for PyNullArray {
    type Encoding = NullEncoding;
}

#[pymethods]
impl PyNullArray {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, &NullEncoding, PyNullArray)
    }
}

/// Concrete class for arrays with `vortex.bool` encoding.
#[pyclass(name = "BoolArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyBoolArray;

impl EncodingSubclass for PyBoolArray {
    type Encoding = BoolEncoding;
}

#[pymethods]
impl PyBoolArray {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, &BoolEncoding, PyBoolArray)
    }
}

/// Concrete class for arrays with `vortex.bytebool` encoding.
#[pyclass(name = "ByteBoolArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyByteBoolArray;

impl EncodingSubclass for PyByteBoolArray {
    type Encoding = ByteBoolEncoding;
}

#[pymethods]
impl PyByteBoolArray {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, &ByteBoolEncoding, PyByteBoolArray)
    }
}

/// Concrete class for arrays with `vortex.primitive` encoding.
#[pyclass(name = "PrimitiveArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyPrimitiveArray;

impl EncodingSubclass for PyPrimitiveArray {
    type Encoding = PrimitiveEncoding;
}

#[pymethods]
impl PyPrimitiveArray {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, &PrimitiveEncoding, PyPrimitiveArray)
    }
}

/// Concrete class for arrays with `vortex.varbin` encoding.
#[pyclass(name = "VarBinArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyVarBinArray;

impl EncodingSubclass for PyVarBinArray {
    type Encoding = VarBinEncoding;
}

#[pymethods]
impl PyVarBinArray {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, &VarBinEncoding, PyVarBinArray)
    }
}

/// Concrete class for arrays with `vortex.varbinview` encoding.
#[pyclass(name = "VarBinViewArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyVarBinViewArray;

impl EncodingSubclass for PyVarBinViewArray {
    type Encoding = VarBinViewEncoding;
}

#[pymethods]
impl PyVarBinViewArray {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, &VarBinViewEncoding, PyVarBinViewArray)
    }
}

/// Concrete class for arrays with `vortex.list` encoding.
#[pyclass(name = "ListArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyListArray;

impl EncodingSubclass for PyListArray {
    type Encoding = ListEncoding;
}

#[pymethods]
impl PyListArray {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, &ListEncoding, PyListArray)
    }
}

/// Concrete class for arrays with `vortex.ext` encoding.
#[pyclass(name = "ExtensionArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyExtensionArray;

impl EncodingSubclass for PyExtensionArray {
    type Encoding = ExtensionEncoding;
}

#[pymethods]
impl PyExtensionArray {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, &ExtensionEncoding, PyExtensionArray)
    }
}
