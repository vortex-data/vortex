// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;

use pyo3::PyRef;
use pyo3::PyResult;
use pyo3::pyclass;
use pyo3::pymethods;
use vortex::dtype::DType;
use vortex::error::vortex_panic;

use crate::dtype::PyDType;

/// Concrete class for fixed-size binary dtypes.
#[pyclass(name = "FixedSizeBinaryDType", module = "vortex", extends=PyDType, frozen)]
pub(crate) struct PyFixedSizeBinaryDType;

#[pymethods]
impl PyFixedSizeBinaryDType {
    /// The number of bytes in each value.
    #[getter]
    fn byte_width(slf: PyRef<Self>) -> PyResult<u32> {
        let DType::FixedSizeBinary(byte_width, _) = slf.as_ref().deref() else {
            vortex_panic!("Not a fixed-size binary DType");
        };
        Ok(*byte_width)
    }
}
