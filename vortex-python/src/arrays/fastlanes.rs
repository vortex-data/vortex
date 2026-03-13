// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::prelude::*;
use vortex::encodings::fastlanes::BitPacked;
use vortex::encodings::fastlanes::Delta;
use vortex::encodings::fastlanes::FoR;

use crate::arrays::native::AsArrayRef;
use crate::arrays::native::EncodingSubclass;
use crate::arrays::native::PyNativeArray;

/// Concrete class for arrays with `fastlanes.bitpacked` encoding.
#[pyclass(name = "FastLanesBitPackedArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyFastLanesBitPackedArray;

impl EncodingSubclass for PyFastLanesBitPackedArray {
    type VTable = BitPacked;
}

#[pymethods]
impl PyFastLanesBitPackedArray {
    /// Returns the bit width of the packed values.
    #[getter]
    fn bit_width(self_: PyRef<'_, Self>) -> u8 {
        self_.as_array_ref().bit_width()
    }
}

/// Concrete class for arrays with `fastlanes.delta` encoding.
#[pyclass(name = "FastLanesDeltaArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyFastLanesDeltaArray;

impl EncodingSubclass for PyFastLanesDeltaArray {
    type VTable = Delta;
}

/// Concrete class for arrays with `fastlanes.for` encoding.
#[pyclass(name = "FastLanesFoRArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyFastLanesFoRArray;

impl EncodingSubclass for PyFastLanesFoRArray {
    type VTable = FoR;
}
