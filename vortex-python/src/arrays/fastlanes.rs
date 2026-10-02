// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::prelude::*;
use vortex::encodings::fastlanes::BitPacked;
use vortex::encodings::fastlanes::BitPackedArrayExt;
use vortex::encodings::fastlanes::BitWidths;
use vortex::encodings::fastlanes::Delta;
use vortex::encodings::fastlanes::FoR;

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
    /// Returns the bit width shared by every block, or `None` if blocks have different widths.
    #[getter]
    fn bit_width(self_: PyRef<'_, Self>) -> Option<u8> {
        match self_.as_super().inner().as_::<BitPacked>().bit_widths() {
            BitWidths::Global(bit_width) => Some(bit_width),
            BitWidths::Blocked(_) => None,
        }
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
