use pyo3::prelude::*;
use vortex::sampling_compressor::SamplingCompressor;

use crate::encoding::PyArray;

#[pyfunction]
/// Attempt to compress a vortex array.
pub fn compress(array: &Bound<PyArray>) -> PyResult<PyArray> {
    let compressor = SamplingCompressor::default();
    let inner = compressor
        .compress(array.borrow().unwrap(), None)?
        .into_array();
    Ok(PyArray::new(inner))
}
