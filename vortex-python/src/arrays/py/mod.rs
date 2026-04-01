// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod python;
mod vtable;

pub(crate) use array::*;
use pyo3::Bound;
use pyo3::PyAny;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::PyAnyMethods;
pub(crate) use python::*;
use vortex::array::vtable::ArrayId;
pub(crate) use vtable::*;

use crate::error::PyVortexResult;

/// Extract the array id from a Python class `id` attribute.
pub fn id_from_obj(cls: &Bound<PyAny>) -> PyVortexResult<ArrayId> {
    Ok(ArrayId::new_arc(
        cls.getattr(intern!(cls.py(), "id"))
            .map_err(|_| {
                PyValueError::new_err(format!(
                    "PyEncoding subclass {cls:?} must have an 'id' attribute"
                ))
            })?
            .extract::<String>()
            .map_err(|_| PyValueError::new_err("'id' attribute must be a string"))?
            .into(),
    ))
}
