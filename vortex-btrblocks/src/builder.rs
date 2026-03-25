// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builder for configuring `BtrBlocksCompressor` instances.

use itertools::Itertools;
use vortex_utils::aliases::hash_set::HashSet;

use crate::BtrBlocksCompressor;
use crate::FloatCode;
use crate::IntCode;
use crate::StringCode;
use crate::compressor::float::ALL_FLOAT_SCHEMES;
use crate::compressor::float::FloatScheme;
use crate::compressor::integer::ALL_INT_SCHEMES;
use crate::compressor::integer::IntegerScheme;
use crate::compressor::string::ALL_STRING_SCHEMES;
use crate::compressor::string::StringScheme;

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
/// let compressor = BtrBlocksCompressorBuilder::default().build();
///
/// // Exclude specific schemes
/// let compressor = BtrBlocksCompressorBuilder::default()
///     .exclude_int([IntCode::Dict])
///     .build();
///
/// // Exclude then re-include
/// let compressor = BtrBlocksCompressorBuilder::default()
///     .exclude_int([IntCode::Dict, IntCode::Rle])
///     .include_int([IntCode::Dict])
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct BtrBlocksCompressorBuilder {
    int_schemes: HashSet<&'static dyn IntegerScheme>,
    float_schemes: HashSet<&'static dyn FloatScheme>,
    string_schemes: HashSet<&'static dyn StringScheme>,
    turboquant_config: Option<vortex_turboquant::TurboQuantConfig>,
}

impl Default for BtrBlocksCompressorBuilder {
    fn default() -> Self {
        Self {
            int_schemes: ALL_INT_SCHEMES
                .iter()
                .copied()
                .filter(|s| s.code() != IntCode::Pco)
                .collect(),
            float_schemes: ALL_FLOAT_SCHEMES
                .iter()
                .copied()
                .filter(|s| s.code() != FloatCode::Pco)
                .collect(),
            string_schemes: ALL_STRING_SCHEMES
                .iter()
                .copied()
                .filter(|s| s.code() != StringCode::Zstd && s.code() != StringCode::ZstdBuffers)
                .collect(),
            turboquant_config: None,
        }
    }
}

impl BtrBlocksCompressorBuilder {
    /// Create a new builder with no encodings enabled.
    pub fn empty() -> Self {
        Self {
            int_schemes: Default::default(),
            float_schemes: Default::default(),
            string_schemes: Default::default(),
            turboquant_config: None,
        }
    }

    /// Excludes the specified integer compression schemes.
    pub fn exclude_int(mut self, codes: impl IntoIterator<Item = IntCode>) -> Self {
        let codes: HashSet<_> = codes.into_iter().collect();
        self.int_schemes.retain(|s| !codes.contains(&s.code()));
        self
    }

    /// Excludes the specified float compression schemes.
    pub fn exclude_float(mut self, codes: impl IntoIterator<Item = FloatCode>) -> Self {
        let codes: HashSet<_> = codes.into_iter().collect();
        self.float_schemes.retain(|s| !codes.contains(&s.code()));
        self
    }

    /// Excludes the specified string compression schemes.
    pub fn exclude_string(mut self, codes: impl IntoIterator<Item = StringCode>) -> Self {
        let codes: HashSet<_> = codes.into_iter().collect();
        self.string_schemes.retain(|s| !codes.contains(&s.code()));
        self
    }

    /// Includes the specified integer compression schemes.
    pub fn include_int(mut self, codes: impl IntoIterator<Item = IntCode>) -> Self {
        let codes: HashSet<_> = codes.into_iter().collect();
        for scheme in ALL_INT_SCHEMES {
            if codes.contains(&scheme.code()) {
                self.int_schemes.insert(*scheme);
            }
        }
        self
    }

    /// Includes the specified float compression schemes.
    pub fn include_float(mut self, codes: impl IntoIterator<Item = FloatCode>) -> Self {
        let codes: HashSet<_> = codes.into_iter().collect();
        for scheme in ALL_FLOAT_SCHEMES {
            if codes.contains(&scheme.code()) {
                self.float_schemes.insert(*scheme);
            }
        }
        self
    }

    /// Includes the specified string compression schemes.
    pub fn include_string(mut self, codes: impl IntoIterator<Item = StringCode>) -> Self {
        let codes: HashSet<_> = codes.into_iter().collect();
        for scheme in ALL_STRING_SCHEMES {
            if codes.contains(&scheme.code()) {
                self.string_schemes.insert(*scheme);
            }
        }
        self
    }

    /// Enables TurboQuant lossy vector quantization for tensor extension types.
    ///
    /// When enabled, `Vector` and `FixedShapeTensor` extension columns will be
    /// quantized at the configured bit-width instead of using the default
    /// recursive storage compression.
    pub fn with_turboquant(mut self, config: vortex_turboquant::TurboQuantConfig) -> Self {
        self.turboquant_config = Some(config);
        self
    }

    /// Builds the configured `BtrBlocksCompressor`.
    pub fn build(self) -> BtrBlocksCompressor {
        // Note we should apply the schemes in the same order, in case try conflict.
        BtrBlocksCompressor {
            int_schemes: self
                .int_schemes
                .into_iter()
                .sorted_by_key(|s| s.code())
                .collect_vec(),
            float_schemes: self
                .float_schemes
                .into_iter()
                .sorted_by_key(|s| s.code())
                .collect_vec(),
            string_schemes: self
                .string_schemes
                .into_iter()
                .sorted_by_key(|s| s.code())
                .collect_vec(),
            turboquant_config: self.turboquant_config,
        }
    }
}
