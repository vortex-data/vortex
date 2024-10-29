use pyo3::prelude::*;
use vortex::sampling_compressor::SamplingCompressor;

use crate::array::PyArray;

#[pyfunction]
/// Attempt to compress a vortex array.
///
/// Parameters
/// ----------
/// array : :class:`~vortex.encoding.Array`
///     The array.
///
/// Examples
/// --------
///
/// Compress a very sparse array of integers:
///
/// >>> a = vortex.array([42 for _ in range(1000)])
/// >>> str(vortex.compress(a))
/// 'vortex.constant(0x09)(i64, len=1000)'
///
/// Compress an array of increasing integers:
///
/// >>> a = vortex.array(list(range(1000)))
/// >>> str(vortex.compress(a))
/// 'fastlanes.for(0x17)(i64, len=1000)'
///
/// Compress an array of increasing floating-point numbers and a few nulls:
///
/// >>> a = vortex.array([
/// ...     float(x) if x % 20 != 0 else None
/// ...     for x in range(1000)
/// ... ])
/// >>> str(vortex.compress(a))
/// 'vortex.alp(0x11)(f64?, len=1000)'
pub fn compress(array: &Bound<PyArray>) -> PyResult<PyArray> {
    let compressor = SamplingCompressor::default();
    let inner = compressor
        .compress(array.borrow().unwrap(), None)?
        .into_array();
    Ok(PyArray::new(inner))
}
