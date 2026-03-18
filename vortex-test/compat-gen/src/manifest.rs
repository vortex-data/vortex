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
    /// SHA-256 hex digest of the file contents (populated after writing).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}
