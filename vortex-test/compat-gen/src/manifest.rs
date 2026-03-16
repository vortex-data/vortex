// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use chrono::DateTime;
use chrono::Utc;
use serde::Deserialize;
use serde::Serialize;

/// Manifest listing all fixtures generated for a given version.
#[derive(Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub version: String,
    pub generated_at: DateTime<Utc>,
    pub fixtures: Vec<FixtureEntry>,
}

/// One entry in the manifest's fixture list.
#[derive(Debug, Serialize, Deserialize)]
pub struct FixtureEntry {
    /// Filename, e.g. "primitives.vortex".
    pub name: String,
    /// Short description of what this fixture tests.
    pub description: String,
}
