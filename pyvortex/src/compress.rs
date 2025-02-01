use pyo3::prelude::*;
use vortex::sampling_compressor::SamplingCompressor;

use crate::arrays::PyArray;
use crate::install_module;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(py, "compress")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.compress", &m)?;

    m.add_function(wrap_pyfunction!(compress, &m)?)?;

    Ok(())
}

/// Attempt to compress a vortex array.
///
/// Parameters
/// ----------
/// array : :class:`~vortex.Array`
///     The array.
///
/// Examples
/// --------
///
/// Compress a very sparse array of integers:
///
///    >>> import vortex as vx
///    >>> a = vx.array([42 for _ in range(1000)])
///    >>> str(vx.compress(a))
///    'vortex.constant(0x09)(i64, len=1000)'
///
/// Compress an array of increasing integers:
///
///    >>> a = vx.array(list(range(1000)))
///    >>> str(vx.compress(a))
///    'fastlanes.bitpacked(0x16)(i64, len=1000)'
///
/// Compress an array of increasing floating-point numbers and a few nulls:
///
///    >>> a = vx.array([
///    ...     float(x) if x % 20 != 0 else None
///    ...     for x in range(1000)
///    ... ])
///    >>> str(vx.compress(a))
///    'vortex.alp(0x11)(f64?, len=1000)'
#[pyfunction]
pub fn compress(array: &Bound<PyArray>) -> PyResult<PyArray> {
    let compressor = SamplingCompressor::default();
    let inner = compressor.compress(&array.borrow(), None)?.into_array();
    Ok(PyArray(inner))
}
