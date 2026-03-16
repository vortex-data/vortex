// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod clickbench;
#[allow(clippy::cast_possible_truncation)]
mod tpch;

use crate::fixtures::ArrayFixture;

/// All dataset-derived fixtures.
pub fn fixtures() -> Vec<Box<dyn ArrayFixture>> {
    let mut fixtures = Vec::new();
    fixtures.extend(tpch::fixtures());
    fixtures.extend(clickbench::fixtures());
    fixtures
}
