// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;

use pyo3::PyClass;
use pyo3::prelude::*;
use vortex::array::ArrayAdapter;
use vortex::array::ArrayRef;
use vortex::array::DynArray;
use vortex::array::arrays::BoolVTable;
use vortex::array::arrays::ChunkedVTable;
use vortex::array::arrays::ConstantVTable;
use vortex::array::arrays::DecimalVTable;
use vortex::array::arrays::DictVTable;
use vortex::array::arrays::ExtensionVTable;
use vortex::array::arrays::FixedSizeListVTable;
use vortex::array::arrays::ListVTable;
use vortex::array::arrays::NullVTable;
use vortex::array::arrays::PrimitiveVTable;
use vortex::array::arrays::StructVTable;
use vortex::array::arrays::VarBinVTable;
use vortex::array::arrays::VarBinViewVTable;
use vortex::array::vtable::VTable;
use vortex::encodings::alp::ALPRDVTable;
use vortex::encodings::alp::ALPVTable;
use vortex::encodings::bytebool::ByteBoolVTable;
use vortex::encodings::datetime_parts::DateTimePartsVTable;
use vortex::encodings::fastlanes::BitPackedVTable;
use vortex::encodings::fastlanes::DeltaVTable;
use vortex::encodings::fastlanes::FoRVTable;
use vortex::encodings::fsst::FSSTVTable;
use vortex::encodings::runend::RunEndVTable;
use vortex::encodings::sequence::SequenceVTable;
use vortex::encodings::sparse::SparseVTable;
use vortex::encodings::zigzag::ZigZagVTable;
use vortex::error::VortexExpect;

use crate::arrays::PyArray;
use crate::arrays::builtins::PyBoolArray;
use crate::arrays::builtins::PyByteBoolArray;
use crate::arrays::builtins::PyChunkedArray;
use crate::arrays::builtins::PyConstantArray;
use crate::arrays::builtins::PyDecimalArray;
use crate::arrays::builtins::PyExtensionArray;
use crate::arrays::builtins::PyFixedSizeListArray;
use crate::arrays::builtins::PyListArray;
use crate::arrays::builtins::PyNullArray;
use crate::arrays::builtins::PyPrimitiveArray;
use crate::arrays::builtins::PyStructArray;
use crate::arrays::builtins::PyVarBinArray;
use crate::arrays::builtins::PyVarBinViewArray;
use crate::arrays::compressed::PyAlpArray;
use crate::arrays::compressed::PyAlpRdArray;
use crate::arrays::compressed::PyDateTimePartsArray;
use crate::arrays::compressed::PyDictArray;
use crate::arrays::compressed::PyFsstArray;
use crate::arrays::compressed::PyRunEndArray;
use crate::arrays::compressed::PySequenceArray;
use crate::arrays::compressed::PySparseArray;
use crate::arrays::compressed::PyZigZagArray;
use crate::arrays::fastlanes::PyFastLanesBitPackedArray;
use crate::arrays::fastlanes::PyFastLanesDeltaArray;
use crate::arrays::fastlanes::PyFastLanesFoRArray;
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
        if array.is::<NullVTable>() {
            return Self::with_subclass(py, array, PyNullArray);
        }

        if array.is::<BoolVTable>() {
            return Self::with_subclass(py, array, PyBoolArray);
        }

        if array.is::<PrimitiveVTable>() {
            return Self::with_subclass(py, array, PyPrimitiveArray);
        }

        if array.is::<VarBinVTable>() {
            return Self::with_subclass(py, array, PyVarBinArray);
        }

        if array.is::<VarBinViewVTable>() {
            return Self::with_subclass(py, array, PyVarBinViewArray);
        }

        if array.is::<StructVTable>() {
            return Self::with_subclass(py, array, PyStructArray);
        }

        if array.is::<ListVTable>() {
            return Self::with_subclass(py, array, PyListArray);
        }

        if array.is::<FixedSizeListVTable>() {
            return Self::with_subclass(py, array, PyFixedSizeListArray);
        }

        if array.is::<ExtensionVTable>() {
            return Self::with_subclass(py, array, PyExtensionArray);
        }

        if array.is::<ChunkedVTable>() {
            return Self::with_subclass(py, array, PyChunkedArray);
        }

        if array.is::<ConstantVTable>() {
            return Self::with_subclass(py, array, PyConstantArray);
        }

        if array.is::<ByteBoolVTable>() {
            return Self::with_subclass(py, array, PyByteBoolArray);
        }

        if array.is::<SparseVTable>() {
            return Self::with_subclass(py, array, PySparseArray);
        }

        if array.is::<ALPVTable>() {
            return Self::with_subclass(py, array, PyAlpArray);
        }

        if array.is::<ALPRDVTable>() {
            return Self::with_subclass(py, array, PyAlpRdArray);
        }

        if array.is::<DateTimePartsVTable>() {
            return Self::with_subclass(py, array, PyDateTimePartsArray);
        }

        if array.is::<DictVTable>() {
            return Self::with_subclass(py, array, PyDictArray);
        }

        if array.is::<FSSTVTable>() {
            return Self::with_subclass(py, array, PyFsstArray);
        }

        if array.is::<RunEndVTable>() {
            return Self::with_subclass(py, array, PyRunEndArray);
        }

        if array.is::<ZigZagVTable>() {
            return Self::with_subclass(py, array, PyZigZagArray);
        }

        if array.is::<BitPackedVTable>() {
            return Self::with_subclass(py, array, PyFastLanesBitPackedArray);
        }

        if array.is::<DeltaVTable>() {
            return Self::with_subclass(py, array, PyFastLanesDeltaArray);
        }

        if array.is::<FoRVTable>() {
            return Self::with_subclass(py, array, PyFastLanesFoRArray);
        }

        if array.is::<DecimalVTable>() {
            return Self::with_subclass(py, array, PyDecimalArray);
        }

        if array.is::<SequenceVTable>() {
            return Self::with_subclass(py, array, PySequenceArray);
        }

        Ok(Bound::new(
            py,
            PyClassInitializer::from(PyArray).add_subclass(PyNativeArray(array)),
        )?
        .into_any()
        .cast_into::<PyNativeArray>()?)
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
        .cast_into::<PyNativeArray>()?)
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
    fn nbytes(&self) -> u64 {
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
            .downcast_ref::<ArrayAdapter<V::VTable>>()
            .vortex_expect("Failed to downcast array")
            .as_inner()
    }
}
