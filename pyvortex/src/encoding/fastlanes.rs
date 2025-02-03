use pyo3::prelude::*;

use crate::arrays::PyArray;

/// Concrete class for arrays with `fastlanes.bitpacked` encoding.
#[pyclass(name = "FastLanesBitPackedEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyFastLanesBitPackedEncoding;

/// Concrete class for arrays with `fastlanes.delta` encoding.
#[pyclass(name = "FastLanesDeltaEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyFastLanesDeltaEncoding;

/// Concrete class for arrays with `fastlanes.for` encoding.
#[pyclass(name = "FastLanesForEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyFastLanesForEncoding;
