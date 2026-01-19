// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;

use itertools::Itertools;
use pyo3::pyclass;
use pyo3::pymethods;
use vortex::array::ArrayContext;
use vortex::array::session::ArraySessionExt;

use crate::SESSION;

/// An ArrayContext captures an ordered set of encodings.
///
/// In a serialized array, encodings are identified by a positional index into such an
/// :class:`~vortex.ArrayContext`.
#[pyclass(name = "ArrayContext", module = "vortex", frozen)]
pub(crate) struct PyArrayContext(ArrayContext);

impl From<ArrayContext> for PyArrayContext {
    fn from(context: ArrayContext) -> Self {
        Self(context)
    }
}

impl Deref for PyArrayContext {
    type Target = ArrayContext;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[pymethods]
impl PyArrayContext {
    #[new]
    fn new() -> Self {
        Self(ArrayContext::default())
    }

    fn __str__(&self) -> String {
        self.0.to_ids().iter().join(", ")
    }

    fn __len__(&self) -> usize {
        self.to_ids().len()
    }
}
