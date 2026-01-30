// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builder for configuring `BtrBlocksCompressor` instances.

use itertools::Itertools;
use vortex_utils::aliases::hash_set::HashSet;

use crate::BtrBlocksCompressor;
use crate::FloatCode;
use crate::FloatCompressor;
use crate::IntCode;
use crate::IntCompressor;
use crate::StringCode;
use crate::StringCompressor;
use crate::float::ALL_FLOAT_SCHEMES;
use crate::float::FloatScheme;
use crate::integer::ALL_INT_SCHEMES;
use crate::integer::IntegerScheme;
use crate::string::ALL_STRING_SCHEMES;
use crate::string::StringScheme;

/// Builder for creating configured [`BtrBlocksCompressor`] instances.
///
/// Use this builder to configure which compression schemes are allowed for each data type.
/// By default, all schemes are enabled.
///
/// # Examples
///
/// ```rust
/// use vortex_btrblocks::{BtrBlocksCompressorBuilder, IntCode, FloatCode};
///
/// // Default compressor - all schemes allowed
/// let compressor = BtrBlocksCompressorBuilder::new().build();
///
/// // Exclude specific schemes
/// let compressor = BtrBlocksCompressorBuilder::new()
///     .exclude_int([IntCode::Dict])
///     .build();
///
/// // Exclude then re-include
/// let compressor = BtrBlocksCompressorBuilder::new()
///     .exclude_int([IntCode::Dict, IntCode::Rle])
///     .include_int([IntCode::Dict])
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct BtrBlocksCompressorBuilder {
    int_schemes: HashSet<&'static dyn IntegerScheme>,
    float_schemes: HashSet<&'static dyn FloatScheme>,
    string_schemes: HashSet<&'static dyn StringScheme>,
}

impl Default for BtrBlocksCompressorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BtrBlocksCompressorBuilder {
    /// Creates a new builder with all schemes enabled.
    pub fn new() -> Self {
        Self {
            int_schemes: ALL_INT_SCHEMES.iter().copied().collect(),
            float_schemes: ALL_FLOAT_SCHEMES.iter().copied().collect(),
            string_schemes: ALL_STRING_SCHEMES.iter().copied().collect(),
        }
    }

    /// Excludes the specified integer compression schemes (set difference).
    ///
    /// # Example
    ///
    /// ```rust
    /// use vortex_btrblocks::{BtrBlocksCompressorBuilder, IntCode};
    ///
    /// let compressor = BtrBlocksCompressorBuilder::new()
    ///     .exclude_int([IntCode::Dict, IntCode::Rle])
    ///     .build();
    /// ```
    pub fn exclude_int(mut self, codes: impl IntoIterator<Item = IntCode>) -> Self {
        let codes: HashSet<_> = codes.into_iter().collect();
        self.int_schemes.retain(|s| !codes.contains(&s.code()));
        self
    }

    /// Excludes the specified float compression schemes (set difference).
    ///
    /// # Example
    ///
    /// ```rust
    /// use vortex_btrblocks::{BtrBlocksCompressorBuilder, FloatCode};
    ///
    /// let compressor = BtrBlocksCompressorBuilder::new()
    ///     .exclude_float([FloatCode::Dict, FloatCode::Alp])
    ///     .build();
    /// ```
    pub fn exclude_float(mut self, codes: impl IntoIterator<Item = FloatCode>) -> Self {
        let codes: HashSet<_> = codes.into_iter().collect();
        self.float_schemes.retain(|s| !codes.contains(&s.code()));
        self
    }

    /// Excludes the specified string compression schemes (set difference).
    ///
    /// # Example
    ///
    /// ```rust
    /// use vortex_btrblocks::{BtrBlocksCompressorBuilder, StringCode};
    ///
    /// let compressor = BtrBlocksCompressorBuilder::new()
    ///     .exclude_string([StringCode::Dict, StringCode::Fsst])
    ///     .build();
    /// ```
    pub fn exclude_string(mut self, codes: impl IntoIterator<Item = StringCode>) -> Self {
        let codes: HashSet<_> = codes.into_iter().collect();
        self.string_schemes.retain(|s| !codes.contains(&s.code()));
        self
    }

    /// Includes the specified integer compression schemes (set union).
    ///
    /// # Example
    ///
    /// ```rust
    /// use vortex_btrblocks::{BtrBlocksCompressorBuilder, IntCode};
    ///
    /// let compressor = BtrBlocksCompressorBuilder::new()
    ///     .exclude_int([IntCode::Dict, IntCode::Rle])
    ///     .include_int([IntCode::Dict]) // re-enables Dict
    ///     .build();
    /// ```
    pub fn include_int(mut self, codes: impl IntoIterator<Item = IntCode>) -> Self {
        let codes: HashSet<_> = codes.into_iter().collect();
        for scheme in ALL_INT_SCHEMES {
            if codes.contains(&scheme.code()) {
                self.int_schemes.insert(*scheme);
            }
        }
        self
    }

    /// Includes the specified float compression schemes (set union).
    ///
    /// # Example
    ///
    /// ```rust
    /// use vortex_btrblocks::{BtrBlocksCompressorBuilder, FloatCode};
    ///
    /// let compressor = BtrBlocksCompressorBuilder::new()
    ///     .exclude_float([FloatCode::Alp, FloatCode::AlpRd])
    ///     .include_float([FloatCode::Alp]) // re-enables Alp
    ///     .build();
    /// ```
    pub fn include_float(mut self, codes: impl IntoIterator<Item = FloatCode>) -> Self {
        let codes: HashSet<_> = codes.into_iter().collect();
        for scheme in ALL_FLOAT_SCHEMES {
            if codes.contains(&scheme.code()) {
                self.float_schemes.insert(*scheme);
            }
        }
        self
    }

    /// Includes the specified string compression schemes (set union).
    ///
    /// # Example
    ///
    /// ```rust
    /// use vortex_btrblocks::{BtrBlocksCompressorBuilder, StringCode};
    ///
    /// let compressor = BtrBlocksCompressorBuilder::new()
    ///     .exclude_string([StringCode::Dict, StringCode::Fsst])
    ///     .include_string([StringCode::Dict]) // re-enables Dict
    ///     .build();
    /// ```
    pub fn include_string(mut self, codes: impl IntoIterator<Item = StringCode>) -> Self {
        let codes: HashSet<_> = codes.into_iter().collect();
        for scheme in ALL_STRING_SCHEMES {
            if codes.contains(&scheme.code()) {
                self.string_schemes.insert(*scheme);
            }
        }
        self
    }

    /// Builds the configured `BtrBlocksCompressor`.
    pub fn build(self) -> BtrBlocksCompressor {
        BtrBlocksCompressor {
            int_schemes: self.int_schemes.into_iter().collect_vec(),
            float_schemes: self.float_schemes.into_iter().collect_vec(),
            string_schemes: self.string_schemes.into_iter().collect_vec(),
        }
    }
}
