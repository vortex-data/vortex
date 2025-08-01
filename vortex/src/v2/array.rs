// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::v2::node::Node;
use vortex_array::stats::StatsSet;

/// Represents a logical in-memory array of data.
pub struct Array {
    node: Box<dyn Node>,
    stats: StatsSet,
}
