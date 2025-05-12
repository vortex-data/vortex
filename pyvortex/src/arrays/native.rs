use std::ops::Deref;

use pyo3::PyClass;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use vortex::arrays::{
    BoolVTable, ChunkedVTable, ConstantVTable, DecimalVTable, ExtensionVTable, ListVTable,
    NullVTable, PrimitiveVTable, StructVTable, VarBinVTable, VarBinViewVTable,
};
use vortex::encodings::alp::{ALPRDVTable, ALPVTable};
use vortex::encodings::bytebool::ByteBoolVTable;
use vortex::encodings::datetime_parts::DateTimePartsVTable;
use vortex::encodings::dict::DictVTable;
use vortex::encodings::fastlanes::{BitPackedVTable, DeltaVTable, FoRVTable};
use vortex::encodings::fsst::FSSTVTable;
use vortex::encodings::runend::RunEndVTable;
use vortex::encodings::sparse::SparseVTable;
use vortex::encodings::zigzag::ZigZagVTable;
use vortex::error::VortexExpect;
use vortex::nbytes::NBytes;
use vortex::vtable::VTable;
use vortex::{Array, ArrayRef};

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

        if any.is::<NullVTable>() {
            return Self::with_subclass(py, array, PyNullArray);
        }

        if any.is::<BoolVTable>() {
            return Self::with_subclass(py, array, PyBoolArray);
        }

        if any.is::<PrimitiveVTable>() {
            return Self::with_subclass(py, array, PyPrimitiveArray);
        }

        if any.is::<VarBinVTable>() {
            return Self::with_subclass(py, array, PyVarBinArray);
        }

        if any.is::<VarBinViewVTable>() {
            return Self::with_subclass(py, array, PyVarBinViewArray);
        }

        if any.is::<StructVTable>() {
            return Self::with_subclass(py, array, PyStructArray);
        }

        if any.is::<ListVTable>() {
            return Self::with_subclass(py, array, PyListArray);
        }

        if any.is::<ExtensionVTable>() {
            return Self::with_subclass(py, array, PyExtensionArray);
        }

        if any.is::<ChunkedVTable>() {
            return Self::with_subclass(py, array, PyChunkedArray);
        }

        if any.is::<ConstantVTable>() {
            return Self::with_subclass(py, array, PyConstantArray);
        }

        if any.is::<ByteBoolVTable>() {
            return Self::with_subclass(py, array, PyByteBoolArray);
        }

        if any.is::<SparseVTable>() {
            return Self::with_subclass(py, array, PySparseArray);
        }

        if any.is::<ALPVTable>() {
            return Self::with_subclass(py, array, PyAlpArray);
        }

        if any.is::<ALPRDVTable>() {
            return Self::with_subclass(py, array, PyAlpRdArray);
        }

        if any.is::<DateTimePartsVTable>() {
            return Self::with_subclass(py, array, PyDateTimePartsArray);
        }

        if any.is::<DictVTable>() {
            return Self::with_subclass(py, array, PyDictArray);
        }

        if any.is::<FSSTVTable>() {
            return Self::with_subclass(py, array, PyFsstArray);
        }

        if any.is::<RunEndVTable>() {
            return Self::with_subclass(py, array, PyRunEndArray);
        }

        if any.is::<ZigZagVTable>() {
            return Self::with_subclass(py, array, PyZigZagArray);
        }

        if any.is::<BitPackedVTable>() {
            return Self::with_subclass(py, array, PyFastLanesBitPackedArray);
        }

        if any.is::<DeltaVTable>() {
            return Self::with_subclass(py, array, PyFastLanesDeltaArray);
        }

        if any.is::<FoRVTable>() {
            return Self::with_subclass(py, array, PyFastLanesFoRArray);
        }

        if any.is::<DecimalVTable>() {
            return Self::with_subclass(py, array, PyDecimalArray);
        }

        Err(PyTypeError::new_err(format!(
            "Unrecognized native array {}",
            array.encoding_id()
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
        self.0.encoding_id().to_string()
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
    type VTable: VTable;
}

/// Unwrap a downcasted Vortex array from a `PyRef<ArraySubclass>`.
pub trait AsArrayRef<T> {
    fn as_array_ref(&self) -> &T;
}

impl<V: EncodingSubclass> AsArrayRef<<V::VTable as VTable>::Array> for PyRef<'_, V> {
    fn as_array_ref(&self) -> &<V::VTable as VTable>::Array {
        self.as_super()
            .inner()
            .as_any()
            .downcast_ref::<<V::VTable as VTable>::Array>()
            .vortex_expect("Failed to downcast array")
    }
}
