// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::prelude::*;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::Dict;
use vortex::array::arrays::PrimitiveArray;
use vortex::encodings::alp::ALP;
use vortex::encodings::alp::ALPRD;
use vortex::encodings::datetime_parts::DateTimeParts;
use vortex::encodings::fsst::FSST;
use vortex::encodings::runend::RunEnd;
use vortex::encodings::sequence::Sequence;
use vortex::encodings::sparse::Sparse;
use vortex::encodings::zigzag::ZigZag;
use vortex::encodings::zigzag::zigzag_encode;
use vortex::error::VortexResult;

use crate::arrays::PyArrayRef;
use crate::arrays::native::EncodingSubclass;
use crate::arrays::native::PyNativeArray;
use crate::error::PyVortexResult;
use crate::session::session;

/// Concrete class for arrays with `vortex.alp` encoding.
#[pyclass(name = "AlpArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyAlpArray;

impl EncodingSubclass for PyAlpArray {
    type VTable = ALP;
}

/// Concrete class for arrays with `vortex.alprd` encoding.
#[pyclass(name = "AlpRdArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyAlpRdArray;

impl EncodingSubclass for PyAlpRdArray {
    type VTable = ALPRD;
}

/// Concrete class for arrays with `vortex.datetimeparts` encoding.
#[pyclass(name = "DateTimePartsArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyDateTimePartsArray;

impl EncodingSubclass for PyDateTimePartsArray {
    type VTable = DateTimeParts;
}

/// Concrete class for arrays with `vortex.dict` encoding.
#[pyclass(name = "DictArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyDictArray;

impl EncodingSubclass for PyDictArray {
    type VTable = Dict;
}

/// Concrete class for arrays with `vortex.fsst` encoding.
#[pyclass(name = "FsstArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyFsstArray;

impl EncodingSubclass for PyFsstArray {
    type VTable = FSST;
}

/// Concrete class for arrays with `vortex.runend` encoding.
#[pyclass(name = "RunEndArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyRunEndArray;

impl EncodingSubclass for PyRunEndArray {
    type VTable = RunEnd;
}

/// Concrete class for arrays with `vortex.sparse` encoding.
#[pyclass(name = "SparseArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PySparseArray;

impl EncodingSubclass for PySparseArray {
    type VTable = Sparse;
}

/// Concrete class for arrays with `vortex.zigzag` encoding.
#[pyclass(name = "ZigZagArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyZigZagArray;

impl EncodingSubclass for PyZigZagArray {
    type VTable = ZigZag;
}

#[pymethods]
impl PyZigZagArray {
    #[staticmethod]
    pub fn encode(py: Python, array: PyArrayRef) -> PyVortexResult<PyArrayRef> {
        let session = session();
        let array = array.into_inner();
        let encoded = py.detach(move || -> VortexResult<ArrayRef> {
            let primitive = array.execute::<PrimitiveArray>(&mut session.create_execution_ctx())?;
            Ok(zigzag_encode(primitive.as_view())?.into_array())
        })?;
        Ok(PyArrayRef::from(encoded))
    }
}

/// Concrete class for arrays with `vortex.sequence` encoding.
#[pyclass(name = "SequenceArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PySequenceArray;

impl EncodingSubclass for PySequenceArray {
    type VTable = Sequence;
}
