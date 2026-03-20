// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builder for configuring `BtrBlocksCompressor` instances.

use vortex_utils::aliases::hash_set::HashSet;

use crate::BtrBlocksCompressor;
use crate::CascadingCompressor;
use crate::Scheme;
use crate::SchemeExt;
use crate::SchemeId;
use crate::schemes::decimal;
use crate::schemes::float;
use crate::schemes::integer;
use crate::schemes::rle;
use crate::schemes::string;
use crate::schemes::temporal;

/// All available compression schemes.
///
/// This list is order-sensitive: the builder preserves this order when constructing
/// the final scheme list, so that tie-breaking is deterministic.
pub const ALL_SCHEMES: &[&dyn Scheme] = &[
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Integer schemes.
    ////////////////////////////////////////////////////////////////////////////////////////////////
    &integer::IntConstantScheme,
    // NOTE: FoR must precede BitPacking to avoid unnecessary patches.
    &integer::FoRScheme,
    // NOTE: ZigZag should precede BitPacking because we don't want negative numbers.
    &integer::ZigZagScheme,
    &integer::BitPackingScheme,
    &integer::SparseScheme,
    &integer::IntDictScheme,
    &integer::RunEndScheme,
    &integer::SequenceScheme,
    &rle::RLE_INTEGER_SCHEME,
    #[cfg(feature = "pco")]
    &integer::PcoScheme,
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Float schemes.
    ////////////////////////////////////////////////////////////////////////////////////////////////
    &float::FloatConstantScheme,
    &float::ALPScheme,
    &float::ALPRDScheme,
    &float::FloatDictScheme,
    &float::NullDominatedSparseScheme,
    &rle::RLE_FLOAT_SCHEME,
    #[cfg(feature = "pco")]
    &float::PcoScheme,
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // String schemes.
    ////////////////////////////////////////////////////////////////////////////////////////////////
    &string::StringDictScheme,
    &string::FSSTScheme,
    &string::StringConstantScheme,
    &string::NullDominatedSparseScheme,
    #[cfg(feature = "zstd")]
    &string::ZstdScheme,
    #[cfg(all(feature = "zstd", feature = "unstable_encodings"))]
    &string::ZstdBuffersScheme,
    // Decimal schemes.
    &decimal::DecimalScheme,
    // Temporal schemes.
    &temporal::TemporalScheme,
];

/// Returns the set of scheme IDs excluded by default (behind feature gates or known-expensive).
pub fn default_excluded() -> HashSet<SchemeId> {
    #[allow(unused_mut, reason = "depends on enabled feature flags")]
    let mut excluded = HashSet::new();
    #[cfg(feature = "pco")]
    {
        excluded.insert(integer::PcoScheme.id());
        excluded.insert(float::PcoScheme.id());
    }
    #[cfg(feature = "zstd")]
    excluded.insert(string::ZstdScheme.id());
    #[cfg(all(feature = "zstd", feature = "unstable_encodings"))]
    excluded.insert(string::ZstdBuffersScheme.id());
    excluded
}

/// Builder for creating configured [`BtrBlocksCompressor`] instances.
///
/// Use this builder to configure which compression schemes are allowed.
/// By default, all schemes are enabled except those in [`default_excluded`].
///
/// # Examples
///
/// ```rust
/// use vortex_btrblocks::{BtrBlocksCompressorBuilder, Scheme, SchemeExt};
/// use vortex_btrblocks::schemes::integer::IntDictScheme;
///
/// // Default compressor - all non-excluded schemes allowed.
/// let compressor = BtrBlocksCompressorBuilder::default().build();
///
/// // Exclude specific schemes.
/// let compressor = BtrBlocksCompressorBuilder::default()
///     .exclude([IntDictScheme.id()])
///     .build();
///
/// // Exclude then re-include.
/// let compressor = BtrBlocksCompressorBuilder::default()
///     .exclude([IntDictScheme.id()])
///     .include([IntDictScheme.id()])
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct BtrBlocksCompressorBuilder {
    schemes: HashSet<&'static dyn Scheme>,
}

impl Default for BtrBlocksCompressorBuilder {
    fn default() -> Self {
        let excluded = default_excluded();
        Self {
            schemes: ALL_SCHEMES
                .iter()
                .copied()
                .filter(|s| !excluded.contains(&s.id()))
                .collect(),
        }
    }
}

impl BtrBlocksCompressorBuilder {
    /// Excludes the specified compression schemes by their [`SchemeId`].
    pub fn exclude(mut self, ids: impl IntoIterator<Item = SchemeId>) -> Self {
        let ids: HashSet<_> = ids.into_iter().collect();
        self.schemes.retain(|s| !ids.contains(&s.id()));
        self
    }

    /// Includes the specified compression schemes by their [`SchemeId`].
    ///
    /// Only schemes present in [`ALL_SCHEMES`] can be included.
    pub fn include(mut self, ids: impl IntoIterator<Item = SchemeId>) -> Self {
        let ids: HashSet<_> = ids.into_iter().collect();
        for scheme in ALL_SCHEMES {
            if ids.contains(&scheme.id()) {
                self.schemes.insert(*scheme);
            }
        }
        self
    }

    /// Adds a single scheme to the builder.
    pub fn with_scheme(mut self, scheme: &'static dyn Scheme) -> Self {
        self.schemes.insert(scheme);
        self
    }

    /// Builds the configured [`BtrBlocksCompressor`].
    ///
    /// The resulting scheme list preserves the order of [`ALL_SCHEMES`] for deterministic
    /// tie-breaking.
    pub fn build(self) -> BtrBlocksCompressor {
        let schemes = ALL_SCHEMES
            .iter()
            .copied()
            .filter(|s| self.schemes.contains(s))
            .collect();
        BtrBlocksCompressor(CascadingCompressor::new(schemes))
    }
}
