use pyo3::prelude::*;

use crate::arrays::PyArray;

/// Concrete class for arrays with `fastlanes.bitpacked` encoding.
#[pyclass(name = "FastLanesBitPackedArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyFastLanesBitPackedArray;

/// Concrete class for arrays with `fastlanes.delta` encoding.
#[pyclass(name = "FastLanesDeltaArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyFastLanesDeltaArray;

/// Concrete class for arrays with `fastlanes.for` encoding.
#[pyclass(name = "FastLanesForArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyFastLanesForArray;
