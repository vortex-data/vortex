// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod arrays;
#[allow(clippy::cast_possible_truncation)]
mod encodings;

use crate::fixtures::ArrayFixture;

/// All synthetic fixtures (arrays + encodings).
pub fn fixtures() -> Vec<Box<dyn ArrayFixture>> {
    let mut fixtures = Vec::new();
    fixtures.extend(arrays::fixtures());
    fixtures.extend(encodings::fixtures());
    fixtures
}
