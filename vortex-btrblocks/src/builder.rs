// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builder for configuring `BtrBlocksCompressor` instances.

use enum_iterator::all;
use vortex_utils::aliases::hash_set::HashSet;

use crate::BtrBlocksCompressor;
use crate::BtrBlocksCompressorConfig;
use crate::FloatCode;
use crate::IntCode;
use crate::StringCode;

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
    int_schemes: HashSet<IntCode>,
    float_schemes: HashSet<FloatCode>,
    string_schemes: HashSet<StringCode>,
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
            int_schemes: all::<IntCode>().collect(),
            float_schemes: all::<FloatCode>().collect(),
            string_schemes: all::<StringCode>().collect(),
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
    pub fn exclude_int(mut self, schemes: impl IntoIterator<Item = IntCode>) -> Self {
        for scheme in schemes {
            self.int_schemes.remove(&scheme);
        }
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
    pub fn exclude_float(mut self, schemes: impl IntoIterator<Item = FloatCode>) -> Self {
        for scheme in schemes {
            self.float_schemes.remove(&scheme);
        }
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
    pub fn exclude_string(mut self, schemes: impl IntoIterator<Item = StringCode>) -> Self {
        for scheme in schemes {
            self.string_schemes.remove(&scheme);
        }
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
    pub fn include_int(mut self, schemes: impl IntoIterator<Item = IntCode>) -> Self {
        self.int_schemes.extend(schemes);
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
    pub fn include_float(mut self, schemes: impl IntoIterator<Item = FloatCode>) -> Self {
        self.float_schemes.extend(schemes);
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
    pub fn include_string(mut self, schemes: impl IntoIterator<Item = StringCode>) -> Self {
        self.string_schemes.extend(schemes);
        self
    }

    /// Builds the configured `BtrBlocksCompressor`.
    pub fn build(self) -> BtrBlocksCompressor {
        let config = BtrBlocksCompressorConfig::from_schemes(
            self.int_schemes,
            self.float_schemes,
            self.string_schemes,
        );
        BtrBlocksCompressor::from_config(config)
    }
}
