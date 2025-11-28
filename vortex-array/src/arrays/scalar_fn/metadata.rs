// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::functions::scalar::ScalarFn;
use vortex_dtype::DType;

#[derive(Clone, Debug)]
pub struct ScalarFnMetadata {
    pub(super) scalar_fn: ScalarFn,
    pub(super) child_dtypes: Vec<DType>,
}
