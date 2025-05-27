use pyo3::prelude::*;
use vortex::compressor::BtrBlocksCompressor;

use crate::arrays::PyArrayRef;
use crate::install_module;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "compress")?;
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
///    'vortex.constant(i64, len=1000)'
///
/// Compress an array of increasing integers:
///
///    >>> a = vx.array(list(range(1000)))
///    >>> str(vx.compress(a))
///    'fastlanes.bitpacked(i64, len=1000)'
///
/// Compress an array of increasing floating-point numbers and a few nulls:
///
///    >>> a = vx.array([
///    ...     float(x) if x % 20 != 0 else None
///    ...     for x in range(1000)
///    ... ])
///    >>> str(vx.compress(a))
///    'vortex.alp(f64?, len=1000)'
#[pyfunction]
pub fn compress(array: PyArrayRef) -> PyResult<PyArrayRef> {
    let compressed = BtrBlocksCompressor.compress(array.inner())?;
    Ok(PyArrayRef::from(compressed))
}
