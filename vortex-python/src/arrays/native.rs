// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;

use pyo3::PyClass;
use pyo3::prelude::*;
use vortex::array::ArrayRef;
use vortex::array::VTable;
use vortex::array::arrays::Bool;
use vortex::array::arrays::Chunked;
use vortex::array::arrays::Constant;
use vortex::array::arrays::Decimal;
use vortex::array::arrays::Dict;
use vortex::array::arrays::Extension;
use vortex::array::arrays::FixedSizeList;
use vortex::array::arrays::List;
use vortex::array::arrays::Null;
use vortex::array::arrays::Primitive;
use vortex::array::arrays::Struct;
use vortex::array::arrays::VarBin;
use vortex::array::arrays::VarBinView;
use vortex::encodings::alp::ALP;
use vortex::encodings::alp::ALPRD;
use vortex::encodings::bytebool::ByteBool;
use vortex::encodings::datetime_parts::DateTimeParts;
use vortex::encodings::fastlanes::BitPacked;
use vortex::encodings::fastlanes::Delta;
use vortex::encodings::fastlanes::FoR;
use vortex::encodings::fsst::FSST;
use vortex::encodings::runend::RunEnd;
use vortex::encodings::sequence::Sequence;
use vortex::encodings::sparse::Sparse;
use vortex::encodings::zigzag::ZigZag;
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
pub struct PyNativeArray {
    array: ArrayRef,
}

impl Deref for PyNativeArray {
    type Target = ArrayRef;

    fn deref(&self) -> &Self::Target {
        &self.array
    }
}

impl PyNativeArray {
    /// Initialize a [`PyArray`] from a Vortex [`ArrayRef`], ensuring we return the correct typed
    /// subclass array.
    pub fn init(py: Python, array: ArrayRef) -> PyResult<Bound<PyNativeArray>> {
        if array.is::<Null>() {
            return Self::with_subclass(py, array, PyNullArray);
        }

        if array.is::<Bool>() {
            return Self::with_subclass(py, array, PyBoolArray);
        }

        if array.is::<Primitive>() {
            return Self::with_subclass(py, array, PyPrimitiveArray);
        }

        if array.is::<VarBin>() {
            return Self::with_subclass(py, array, PyVarBinArray);
        }

        if array.is::<VarBinView>() {
            return Self::with_subclass(py, array, PyVarBinViewArray);
        }

        if array.is::<Struct>() {
            return Self::with_subclass(py, array, PyStructArray);
        }

        if array.is::<List>() {
            return Self::with_subclass(py, array, PyListArray);
        }

        if array.is::<FixedSizeList>() {
            return Self::with_subclass(py, array, PyFixedSizeListArray);
        }

        if array.is::<Extension>() {
            return Self::with_subclass(py, array, PyExtensionArray);
        }

        if array.is::<Chunked>() {
            return Self::with_subclass(py, array, PyChunkedArray);
        }

        if array.is::<Constant>() {
            return Self::with_subclass(py, array, PyConstantArray);
        }

        if array.is::<ByteBool>() {
            return Self::with_subclass(py, array, PyByteBoolArray);
        }

        if array.is::<Sparse>() {
            return Self::with_subclass(py, array, PySparseArray);
        }

        if array.is::<ALP>() {
            return Self::with_subclass(py, array, PyAlpArray);
        }

        if array.is::<ALPRD>() {
            return Self::with_subclass(py, array, PyAlpRdArray);
        }

        if array.is::<DateTimeParts>() {
            return Self::with_subclass(py, array, PyDateTimePartsArray);
        }

        if array.is::<Dict>() {
            return Self::with_subclass(py, array, PyDictArray);
        }

        if array.is::<FSST>() {
            return Self::with_subclass(py, array, PyFsstArray);
        }

        if array.is::<RunEnd>() {
            return Self::with_subclass(py, array, PyRunEndArray);
        }

        if array.is::<ZigZag>() {
            return Self::with_subclass(py, array, PyZigZagArray);
        }

        if array.is::<BitPacked>() {
            return Self::with_subclass(py, array, PyFastLanesBitPackedArray);
        }

        if array.is::<Delta>() {
            return Self::with_subclass(py, array, PyFastLanesDeltaArray);
        }

        if array.is::<FoR>() {
            return Self::with_subclass(py, array, PyFastLanesFoRArray);
        }

        if array.is::<Decimal>() {
            return Self::with_subclass(py, array, PyDecimalArray);
        }

        if array.is::<Sequence>() {
            return Self::with_subclass(py, array, PySequenceArray);
        }

        Ok(Bound::new(
            py,
            PyClassInitializer::from(PyArray).add_subclass(PyNativeArray { array }),
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
                .add_subclass(PyNativeArray { array })
                .add_subclass(subclass),
        )?
        .into_any()
        .cast_into::<PyNativeArray>()?)
    }

    pub fn inner(&self) -> &ArrayRef {
        &self.array
    }

    pub fn into_inner(self) -> ArrayRef {
        self.array
    }
}

#[pymethods]
impl PyNativeArray {
    fn __len__(&self) -> usize {
        self.len()
    }

    fn __str__(&self) -> String {
        format!("{}", self.array)
    }

    /// Returns the encoding ID of this array.
    #[getter]
    fn id(&self) -> String {
        self.array.encoding_id().to_string()
    }

    /// Returns the number of bytes used by this array.
    #[getter]
    fn nbytes(&self) -> u64 {
        self.array.nbytes()
    }

    #[getter]
    fn dtype(self_: PyRef<Self>) -> PyResult<Bound<PyDType>> {
        PyDType::init(self_.py(), self_.array.dtype().clone())
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

impl<V: EncodingSubclass> AsArrayRef<<V::VTable as VTable>::TypedArrayData> for PyRef<'_, V> {
    fn as_array_ref(&self) -> &<V::VTable as VTable>::TypedArrayData {
        self.as_super()
            .inner()
            .as_opt::<V::VTable>()
            .vortex_expect("Failed to downcast array")
            .data()
    }
}
