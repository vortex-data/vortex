// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::vector::vector::Vector;
use vortex_array::stats::StatsSet;
use vortex_dtype::DType;

/// An array wraps up raw vectors into a more logically sound structure (but heavier-weight)
/// object.
/// Maybe???
pub struct Array {
    dtype: DType,
    stats: StatsSet,
    vectors: Vec<Vector>,
}
