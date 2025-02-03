use pyo3::prelude::*;
use pyo3::{Bound, PyResult, Python};

mod builtins;
mod compressed;
mod fastlanes;

use builtins::{
    PyBoolEncoding, PyChunkedEncoding, PyConstantEncoding, PyExtensionEncoding, PyListEncoding,
    PyNullEncoding, PyPrimitiveEncoding, PyStructEncoding, PyVarBinEncoding, PyVarBinViewEncoding,
};
use compressed::{
    PyAlpEncoding, PyAlpRdEncoding, PyDateTimePartsEncoding, PyDictEncoding, PyFsstEncoding,
    PyRunEndEncoding, PySparseEncoding, PyZigZagEncoding,
};
use fastlanes::{PyFastLanesBitPackedEncoding, PyFastLanesDeltaEncoding, PyFastLanesForEncoding};

use crate::install_module;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(py, "encoding")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.encoding", &m)?;

    // Canonical encodings
    m.add_class::<PyConstantEncoding>()?;
    m.add_class::<PyChunkedEncoding>()?;
    m.add_class::<PyNullEncoding>()?;
    m.add_class::<PyBoolEncoding>()?;
    m.add_class::<PyPrimitiveEncoding>()?;
    m.add_class::<PyVarBinEncoding>()?;
    m.add_class::<PyVarBinViewEncoding>()?;
    m.add_class::<PyStructEncoding>()?;
    m.add_class::<PyListEncoding>()?;
    m.add_class::<PyExtensionEncoding>()?;

    // Compressed encodings
    m.add_class::<PyAlpEncoding>()?;
    m.add_class::<PyAlpRdEncoding>()?;
    m.add_class::<PyDateTimePartsEncoding>()?;
    m.add_class::<PyDictEncoding>()?;
    m.add_class::<PyFsstEncoding>()?;
    m.add_class::<PyRunEndEncoding>()?;
    m.add_class::<PySparseEncoding>()?;
    m.add_class::<PyZigZagEncoding>()?;

    // Fastlanes encodings
    m.add_class::<PyFastLanesBitPackedEncoding>()?;
    m.add_class::<PyFastLanesDeltaEncoding>()?;
    m.add_class::<PyFastLanesForEncoding>()?;

    Ok(())
}
