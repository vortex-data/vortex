// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod datasets;
mod synthetic;

use super::ArrayFixture;

/// All array fixtures.
pub fn fixtures() -> Vec<Box<dyn ArrayFixture>> {
    let mut fixtures = Vec::new();
    fixtures.extend(synthetic::fixtures());
    fixtures.extend(datasets::fixtures());
    fixtures
}
