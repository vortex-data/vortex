// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;
use std::sync::Arc;

use itertools::Itertools;
use pyo3::pyclass;
use pyo3::pymethods;
use vortex::array::ArrayContext;
use vortex::session::registry::Id;
use vortex::session::registry::ReadContext;

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
        Self(ArrayContext::empty())
    }

    fn __str__(&self) -> String {
        self.0.to_ids().iter().join(", ")
    }

    fn __len__(&self) -> usize {
        self.to_ids().len()
    }
}

/// A ReadContext captures an ordered set of encodings.
///
/// In a serialized array, encodings are identified by a positional index into such an
/// :class:`~vortex.ReadContext`.
#[pyclass(name = "ReadContext", module = "vortex", frozen)]
pub(crate) struct PyReadContext(ReadContext);

impl From<ReadContext> for PyReadContext {
    fn from(context: ReadContext) -> Self {
        Self(context)
    }
}

impl Deref for PyReadContext {
    type Target = ReadContext;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[pymethods]
impl PyReadContext {
    #[new]
    fn new(ids: Vec<String>) -> Self {
        Self(ReadContext::new(
            ids.into_iter().map(|i| Id::new(&i)).collect::<Arc<_>>(),
        ))
    }

    fn __str__(&self) -> String {
        self.0.ids().iter().join(", ")
    }

    fn __len__(&self) -> usize {
        self.ids().len()
    }
}
