// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::pyclass;
use vortex::scalar::DecimalScalar;

use crate::scalar::PyScalar;
use crate::scalar::ScalarSubclass;

/// Concrete class for primitive scalars.
#[pyclass(name = "DecimalScalar", module = "vortex", extends=PyScalar, frozen)]
pub(crate) struct PyDecimalScalar;

impl ScalarSubclass for PyDecimalScalar {
    type Scalar<'a> = DecimalScalar<'a>;
}
