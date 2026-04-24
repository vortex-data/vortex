// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::prelude::*;

use crate::dtype::PyDType;

/// Concrete class for utf8 dtypes.
#[pyclass(name = "BinaryDType", module = "vortex", extends=PyDType, frozen)]
pub(crate) struct PyBinaryDType;
