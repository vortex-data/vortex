use std::ops::Deref;

use pyo3::PyClass;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use vortex::arrays::{
    BoolArray, ChunkedArray, ConstantArray, DecimalArray, ExtensionArray, ListArray, NullArray,
    PrimitiveArray, StructArray, VarBinArray, VarBinViewArray,
};
use vortex::encodings::alp::{ALPArray, ALPRDArray};
use vortex::encodings::bytebool::ByteBoolArray;
use vortex::encodings::datetime_parts::DateTimePartsArray;
use vortex::encodings::dict::DictArray;
use vortex::encodings::fastlanes::{BitPackedArray, DeltaArray, FoRArray};
use vortex::encodings::fsst::FSSTArray;
use vortex::encodings::runend::RunEndArray;
use vortex::encodings::sparse::SparseArray;
use vortex::encodings::zigzag::ZigZagArray;
use vortex::error::VortexExpect;
use vortex::nbytes::NBytes;
use vortex::{Array, ArrayRef, Encoding};

use crate::arrays::PyArray;
use crate::arrays::builtins::{
    PyBoolArray, PyByteBoolArray, PyChunkedArray, PyConstantArray, PyDecimalArray,
    PyExtensionArray, PyListArray, PyNullArray, PyPrimitiveArray, PyStructArray, PyVarBinArray,
    PyVarBinViewArray,
};
use crate::arrays::compressed::{
    PyAlpArray, PyAlpRdArray, PyDateTimePartsArray, PyDictArray, PyFsstArray, PyRunEndArray,
    PySparseArray, PyZigZagArray,
};
use crate::arrays::fastlanes::{
    PyFastLanesBitPackedArray, PyFastLanesDeltaArray, PyFastLanesFoRArray,
};
use crate::dtype::PyDType;

#[pyclass(name = "NativeArray", module = "vortex", extends=PyArray, sequence, subclass, frozen)]
pub struct PyNativeArray(ArrayRef);

impl Deref for PyNativeArray {
    type Target = ArrayRef;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PyNativeArray {
    /// Initialize a [`PyArray`] from a Vortex [`ArrayRef`], ensuring we return the correct typed
    /// subclass array.
    pub fn init(py: Python, array: ArrayRef) -> PyResult<Bound<PyNativeArray>> {
        let any = array.as_any();

        if any.is::<NullArray>() {
            return Self::with_subclass(py, array, PyNullArray);
        }

        if any.is::<BoolArray>() {
            return Self::with_subclass(py, array, PyBoolArray);
        }

        if any.is::<PrimitiveArray>() {
            return Self::with_subclass(py, array, PyPrimitiveArray);
        }

        if any.is::<VarBinArray>() {
            return Self::with_subclass(py, array, PyVarBinArray);
        }

        if any.is::<VarBinViewArray>() {
            return Self::with_subclass(py, array, PyVarBinViewArray);
        }

        if any.is::<StructArray>() {
            return Self::with_subclass(py, array, PyStructArray);
        }

        if any.is::<ListArray>() {
            return Self::with_subclass(py, array, PyListArray);
        }

        if any.is::<ExtensionArray>() {
            return Self::with_subclass(py, array, PyExtensionArray);
        }

        if any.is::<ChunkedArray>() {
            return Self::with_subclass(py, array, PyChunkedArray);
        }

        if any.is::<ConstantArray>() {
            return Self::with_subclass(py, array, PyConstantArray);
        }

        if any.is::<ByteBoolArray>() {
            return Self::with_subclass(py, array, PyByteBoolArray);
        }

        if any.is::<SparseArray>() {
            return Self::with_subclass(py, array, PySparseArray);
        }

        if any.is::<ALPArray>() {
            return Self::with_subclass(py, array, PyAlpArray);
        }

        if any.is::<ALPRDArray>() {
            return Self::with_subclass(py, array, PyAlpRdArray);
        }

        if any.is::<DateTimePartsArray>() {
            return Self::with_subclass(py, array, PyDateTimePartsArray);
        }

        if any.is::<DictArray>() {
            return Self::with_subclass(py, array, PyDictArray);
        }

        if any.is::<FSSTArray>() {
            return Self::with_subclass(py, array, PyFsstArray);
        }

        if any.is::<RunEndArray>() {
            return Self::with_subclass(py, array, PyRunEndArray);
        }

        if any.is::<ZigZagArray>() {
            return Self::with_subclass(py, array, PyZigZagArray);
        }

        if any.is::<BitPackedArray>() {
            return Self::with_subclass(py, array, PyFastLanesBitPackedArray);
        }

        if any.is::<DeltaArray>() {
            return Self::with_subclass(py, array, PyFastLanesDeltaArray);
        }

        if any.is::<FoRArray>() {
            return Self::with_subclass(py, array, PyFastLanesFoRArray);
        }

        if any.is::<DecimalArray>() {
            return Self::with_subclass(py, array, PyDecimalArray);
        }

        Err(PyTypeError::new_err(format!(
            "Unrecognized native array {}",
            array.encoding()
        )))
    }

    fn with_subclass<S: PyClass<BaseType = PyNativeArray>>(
        py: Python,
        array: ArrayRef,
        subclass: S,
    ) -> PyResult<Bound<PyNativeArray>> {
        Ok(Bound::new(
            py,
            PyClassInitializer::from(PyArray)
                .add_subclass(PyNativeArray(array))
                .add_subclass(subclass),
        )?
        .into_any()
        .downcast_into::<PyNativeArray>()?)
    }

    pub fn inner(&self) -> &ArrayRef {
        &self.0
    }

    pub fn into_inner(self) -> ArrayRef {
        self.0
    }
}

#[pymethods]
impl PyNativeArray {
    fn __len__(&self) -> usize {
        self.len()
    }

    fn __str__(&self) -> String {
        format!("{}", self.0)
    }

    /// Returns the encoding ID of this array.
    #[getter]
    fn id(&self) -> String {
        self.0.encoding().to_string()
    }

    /// Returns the number of bytes used by this array.
    #[getter]
    fn nbytes(&self) -> usize {
        self.0.nbytes()
    }

    #[getter]
    fn dtype(self_: PyRef<Self>) -> PyResult<Bound<PyDType>> {
        PyDType::init(self_.py(), self_.0.dtype().clone())
    }
}

/// A marker trait indicating a PyO3 class is a subclass of Vortex `Array`.
pub trait EncodingSubclass: PyClass<BaseType = PyNativeArray> {
    type Encoding: Encoding;
}

/// Unwrap a downcasted Vortex array from a `PyRef<ArraySubclass>`.
pub trait AsArrayRef<T> {
    fn as_array_ref(&self) -> &T;
}

impl<A: EncodingSubclass> AsArrayRef<<A::Encoding as Encoding>::Array> for PyRef<'_, A> {
    fn as_array_ref(&self) -> &<A::Encoding as Encoding>::Array {
        self.as_super()
            .inner()
            .as_any()
            .downcast_ref::<<A::Encoding as Encoding>::Array>()
            .vortex_expect("Failed to downcast array")
    }
}
