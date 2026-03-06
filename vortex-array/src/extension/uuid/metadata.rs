// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

/// Metadata for the UUID extension type, which is empty.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct UuidMetadata;

impl fmt::Display for UuidMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UUID")
    }
}
