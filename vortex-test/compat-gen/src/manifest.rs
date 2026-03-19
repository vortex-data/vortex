// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use serde::Deserialize;
use serde::Serialize;

/// One entry in the fixture manifest.
#[derive(Debug, Serialize, Deserialize)]
pub struct FixtureEntry {
    /// Filename, e.g. "primitives.vortex".
    pub name: String,
    /// Short description of what this fixture tests.
    pub description: String,
}
