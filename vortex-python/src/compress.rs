// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
///    >>> vx.compress(a).display_tree()
///    'root: vortex.constant(i64, len=1000) nbytes=2 B (100.00%)\n  metadata: EmptyMetadata\n  buffer (align=1): 2 B (100.00%)\n'
///
/// Compress an array of increasing integers:
///
///    >>> a = vx.array(list(range(1000)))
///    >>> vx.compress(a).display_tree()
///    'root: vortex.sequence(i64, len=1000) nbytes=0 B (NaN%)\n  metadata: SequenceMetadata { base: Some(ScalarValue { kind: Some(Int64Value(0)) }), multiplier: Some(ScalarValue { kind: Some(Int64Value(1)) }) }\n'
///
/// Compress an array of increasing floating-point numbers and a few nulls:
///
///    >>> a = vx.array([
///    ...     float(x) if x % 20 != 0 else None
///    ...     for x in range(1000)
///    ... ])
///    >>> vx.compress(a).display_tree()
///    'root: vortex.alp(f64?, len=1000) nbytes=1.92 kB (100.00%)\n  metadata: ALPMetadata { exp_e: 16, exp_f: 15, patches: None }\n  encoded: fastlanes.for(i64?, len=1000) nbytes=1.92 kB (100.00%)\n    metadata: 10i64\n    encoded: fastlanes.bitpacked(u64?, len=1000) nbytes=1.92 kB (100.00%)\n      metadata: BitPackedMetadata { bit_width: 14, offset: 0, patches: None }\n      buffer (align=8): 1.79 kB (93.48%)\n      validity: vortex.bool(bool, len=1000) nbytes=125 B (6.52%)\n        metadata: BoolMetadata { offset: 0 }\n        buffer (align=1): 125 B (100.00%)\n'
#[pyfunction]
pub fn compress(array: PyArrayRef) -> PyResult<PyArrayRef> {
    let compressed = BtrBlocksCompressor.compress(array.inner())?;
    Ok(PyArrayRef::from(compressed))
}
