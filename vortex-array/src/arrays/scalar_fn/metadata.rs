// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;

use crate::expr::BoundExpression;

#[derive(Clone, Debug)]
pub struct ScalarFnMetadata {
    pub(super) bound: BoundExpression,
    pub(super) child_dtypes: Vec<DType>,
}
