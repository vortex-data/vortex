// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;

use crate::expr::functions::scalar::ScalarFn;

#[derive(Clone, Debug)]
pub struct ScalarFnMetadata {
    pub(super) scalar_fn: ScalarFn,
    pub(super) child_dtypes: Vec<DType>,
}
