// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Extension types.

use std::fmt;

pub mod datetime;

/// An empty metadata struct for extension dtypes that do not require any metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EmptyMetadata;
impl fmt::Display for EmptyMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "")
    }
}
