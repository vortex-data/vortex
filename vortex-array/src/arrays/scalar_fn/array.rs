// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::functions::v2::ScalarFnRef;
use crate::stats::ArrayStats;
use crate::ArrayRef;

pub struct ScalarFnArray {
    scalar_fn: ScalarFnRef,
    children: Vec<ArrayRef>,
    stats: ArrayStats,
}
