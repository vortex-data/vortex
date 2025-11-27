// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::functions::v2::ScalarFnRef;
use vortex_dtype::DType;

pub struct ScalarFnMetadata {
    pub scalar_fn: ScalarFnRef,
    pub child_dtypes: Vec<DType>,
}
