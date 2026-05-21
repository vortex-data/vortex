// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;

use uuid::Version;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

/// Converts a `u8` discriminant back to a [`uuid::Version`].
pub(crate) fn u8_to_version(b: u8) -> VortexResult<Version> {
    match b {
        0 => Ok(Version::Nil),
        1 => Ok(Version::Mac),
        2 => Ok(Version::Dce),
        3 => Ok(Version::Md5),
        4 => Ok(Version::Random),
        5 => Ok(Version::Sha1),
        6 => Ok(Version::SortMac),
        7 => Ok(Version::SortRand),
        8 => Ok(Version::Custom),
        // UUID crate changed from 0xff to 0x0f for maximum uuid version in 1.23.0
        0x0f => Ok(Version::Max),
        0xff => Ok(Version::Max),
        _ => vortex_bail!("unknown UUID version discriminant: {b}"),
    }
}

/// Metadata for the UUID extension type.
///
/// Optionally records which UUID version the column contains (e.g. v4 random, v7
/// sort-random). When `None`, the column may contain any mix of versions.
#[derive(Clone, Debug, Default)]
pub struct UuidMetadata {
    /// The UUID version, if known.
    pub version: Option<Version>,
}

impl fmt::Display for UuidMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.version {
            None => write!(f, ""),
            Some(v) => write!(f, "v{}", v as u8),
        }
    }
}

// `uuid::Version` derives `PartialEq` but not `Eq` or `Hash`, so we implement these
// manually using the `#[repr(u8)]` discriminant.

impl PartialEq for UuidMetadata {
    fn eq(&self, other: &Self) -> bool {
        self.version.map(|v| v as u8) == other.version.map(|v| v as u8)
    }
}

impl Eq for UuidMetadata {}

impl Hash for UuidMetadata {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.version.map(|v| v as u8).hash(state);
    }
}
