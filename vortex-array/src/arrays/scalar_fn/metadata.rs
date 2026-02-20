// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;

use crate::dtype::DType;
use crate::expr::ScalarFn;

#[derive(Clone)]
pub struct ScalarFnMetadata {
    pub(super) scalar_fn: ScalarFn,
    pub(super) child_dtypes: Vec<DType>,
}

// Array tree display wrongly uses debug...
impl Debug for ScalarFnMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.scalar_fn.options())
    }
}
