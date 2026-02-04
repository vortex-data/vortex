// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression context types for recursive compression.

use crate::FloatCode;
use crate::IntCode;
use crate::MAX_CASCADE;
use crate::StringCode;

/// Holds references to exclude lists for each compression code type.
///
/// This struct is passed through recursive compression calls to specify
/// which schemes should be excluded at each level.
#[derive(Debug, Clone, Copy, Default)]
pub struct Excludes<'a> {
    /// Integer schemes to exclude.
    pub int: &'a [IntCode],
    /// Float schemes to exclude.
    pub float: &'a [FloatCode],
    /// String schemes to exclude.
    pub string: &'a [StringCode],
}

impl<'a> Excludes<'a> {
    /// Creates an empty excludes (no exclusions).
    pub const fn none() -> Self {
        Self {
            int: &[],
            float: &[],
            string: &[],
        }
    }

    /// Creates excludes with only integer exclusions.
    pub const fn int_only(int: &'a [IntCode]) -> Self {
        Self {
            int,
            float: &[],
            string: &[],
        }
    }

    /// Creates excludes with only float exclusions.
    pub const fn float_only(float: &'a [FloatCode]) -> Self {
        Self {
            int: &[],
            float,
            string: &[],
        }
    }

    /// Creates excludes with only string exclusions.
    pub const fn string_only(string: &'a [StringCode]) -> Self {
        Self {
            int: &[],
            float: &[],
            string,
        }
    }
}

impl<'a> From<&'a [IntCode]> for Excludes<'a> {
    fn from(int: &'a [IntCode]) -> Self {
        Self::int_only(int)
    }
}

impl<'a, const N: usize> From<&'a [IntCode; N]> for Excludes<'a> {
    fn from(int: &'a [IntCode; N]) -> Self {
        Self::int_only(int)
    }
}

impl<'a> From<&'a [FloatCode]> for Excludes<'a> {
    fn from(float: &'a [FloatCode]) -> Self {
        Self::float_only(float)
    }
}

impl<'a, const N: usize> From<&'a [FloatCode; N]> for Excludes<'a> {
    fn from(float: &'a [FloatCode; N]) -> Self {
        Self::float_only(float)
    }
}

impl<'a> From<&'a [StringCode]> for Excludes<'a> {
    fn from(string: &'a [StringCode]) -> Self {
        Self::string_only(string)
    }
}

impl<'a, const N: usize> From<&'a [StringCode; N]> for Excludes<'a> {
    fn from(string: &'a [StringCode; N]) -> Self {
        Self::string_only(string)
    }
}

/// Context passed through recursive compression calls.
///
/// Bundles `is_sample` and `allowed_cascading` which always travel together.
/// Excludes are passed separately since they're type-specific.
#[derive(Debug, Clone, Copy)]
pub struct CompressorContext {
    /// Whether we're compressing a sample (for ratio estimation).
    pub is_sample: bool,
    /// Remaining cascade depth allowed.
    pub allowed_cascading: usize,
}

impl Default for CompressorContext {
    fn default() -> Self {
        Self {
            is_sample: false,
            allowed_cascading: MAX_CASCADE,
        }
    }
}

impl CompressorContext {
    /// Descend one level in the cascade (decrements `allowed_cascading`).
    pub fn descend(self) -> Self {
        Self {
            allowed_cascading: self.allowed_cascading.saturating_sub(1),
            ..self
        }
    }

    /// Returns a context marked as sample compression (for ratio estimation).
    pub fn as_sample(self) -> Self {
        Self {
            is_sample: true,
            ..self
        }
    }
}
