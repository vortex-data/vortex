// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builder for configuring `BtrBlocksCompressor` instances.

use vortex_array::ArrayId;
use vortex_decimal_byte_parts::decimal_byte_parts_v2_id;
use vortex_utils::aliases::hash_set::HashSet;

use crate::BtrBlocksCompressor;
use crate::CascadingCompressor;
use crate::Scheme;
use crate::SchemeExt;
use crate::SchemeId;
use crate::schemes::binary;
use crate::schemes::decimal;
use crate::schemes::float;
use crate::schemes::integer;
use crate::schemes::string;
use crate::schemes::temporal;

/// Returns the default compression schemes, including Decimal v1.
///
/// The order is preserved for deterministic tie-breaking. The builder selects compatible versions
/// and filters schemes during [`BtrBlocksCompressorBuilder::build`].
pub fn all_schemes() -> Vec<&'static dyn Scheme> {
    vec![
        ////////////////////////////////////////////////////////////////////////////////////////////////
        // Integer schemes.
        ////////////////////////////////////////////////////////////////////////////////////////////////
        // NOTE: FoR must precede BitPacking to avoid unnecessary patches.
        &integer::FoRScheme,
        // NOTE: ZigZag should precede BitPacking because we don't want negative numbers.
        &integer::ZigZagScheme,
        &integer::BitPackingScheme,
        &integer::SparseScheme,
        &integer::IntDictScheme,
        &integer::RunEndScheme,
        &integer::SequenceScheme,
        &integer::IntRLEScheme,
        // Delta is omitted here: see [`DELTA_SCHEME`].
        ////////////////////////////////////////////////////////////////////////////////////////////////
        // Float schemes.
        ////////////////////////////////////////////////////////////////////////////////////////////////
        &float::ALPScheme,
        &float::ALPRDScheme,
        &float::FloatDictScheme,
        &float::NullDominatedSparseScheme,
        &float::FloatRLEScheme,
        ////////////////////////////////////////////////////////////////////////////////////////////////
        // String schemes.
        ////////////////////////////////////////////////////////////////////////////////////////////////
        &string::StringDictScheme,
        // Both string-fragmentation schemes are registered; the sample-based
        // selector keeps whichever is smaller per column.
        &string::FSSTScheme,
        &string::OnPairScheme,
        &string::NullDominatedSparseScheme,
        ////////////////////////////////////////////////////////////////////////////////////////////////
        // Binary schemes.
        ////////////////////////////////////////////////////////////////////////////////////////////////
        &binary::BinaryDictScheme,
        &binary::VarBinScheme,
        // Decimal schemes.
        &decimal::DecimalScheme::new(false),
        // Temporal schemes.
        &temporal::TemporalScheme,
    ]
}

/// Delta, kept out of [`all_schemes`] because it is slower to decompress than the schemes that
/// would otherwise win. Callers that want it opt in with
/// [`with_new_scheme`](BtrBlocksCompressorBuilder::with_new_scheme).
///
/// TODO(robert): Return it to [`all_schemes`] once we have scheme filtering.
pub static DELTA_SCHEME: integer::DeltaScheme = integer::DeltaScheme::new(1.25);

/// Builder for creating configured [`BtrBlocksCompressor`] instances.
///
/// By default, all schemes in [`all_schemes`] are enabled in a deterministic order. Feature-gated
/// schemes (Pco, Zstd) are not in `all_schemes` and must be added explicitly via
/// [`with_new_scheme`](BtrBlocksCompressorBuilder::with_new_scheme) or `with_compact` when the
/// `zstd` feature is enabled.
/// [`Self::with_allowed_encodings`] sets the permitted serialized IDs. [`Self::build`] selects
/// compatible scheme versions and filters their outputs. Without an allowlist, the registered
/// configurations are preserved, including Decimal v1 in the defaults.
///
/// # Examples
///
/// ```rust
/// use vortex_btrblocks::{BtrBlocksCompressorBuilder, Scheme, SchemeExt};
/// use vortex_btrblocks::schemes::integer::IntDictScheme;
///
/// // Default compressor with all schemes from all_schemes.
/// let compressor = BtrBlocksCompressorBuilder::default().build();
///
/// // Remove specific schemes.
/// let compressor = BtrBlocksCompressorBuilder::default()
///     .exclude_schemes([IntDictScheme.id()])
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct BtrBlocksCompressorBuilder {
    schemes: Vec<&'static dyn Scheme>,
    allowed_serialized_ids: Option<HashSet<ArrayId>>,
}

impl Default for BtrBlocksCompressorBuilder {
    fn default() -> Self {
        Self {
            schemes: all_schemes(),
            allowed_serialized_ids: None,
        }
    }
}

impl BtrBlocksCompressorBuilder {
    /// Creates a builder with no schemes registered.
    ///
    /// Useful when the caller wants explicit, scheme-by-scheme control over the compressor.
    pub fn empty() -> Self {
        Self {
            schemes: Vec::new(),
            allowed_serialized_ids: None,
        }
    }

    /// Adds an external compression scheme not in [`all_schemes`].
    ///
    /// This allows encoding crates outside of `vortex-btrblocks` to register their own schemes
    /// with the compressor.
    ///
    /// # Panics
    ///
    /// Panics if a scheme with the same [`SchemeId`] is already present.
    pub fn with_new_scheme(mut self, scheme: &'static dyn Scheme) -> Self {
        assert!(
            !self.schemes.iter().any(|s| s.id() == scheme.id()),
            "scheme {:?} is already present in the builder",
            scheme.id(),
        );

        self.schemes.push(scheme);
        self
    }

    /// Adds compact encoding schemes (Zstd for strings and binary, Pco for numerics).
    ///
    /// This provides better compression ratios than the default, especially for floating-point
    /// heavy datasets. Requires the `zstd` feature. When the `pco` feature is also enabled,
    /// Pco schemes for integers and floats are included.
    ///
    /// # Panics
    ///
    /// Panics if any of the compact schemes are already present.
    #[cfg(feature = "zstd")]
    pub fn with_compact(self) -> Self {
        let builder = self
            .with_new_scheme(&string::ZstdScheme)
            .with_new_scheme(&binary::ZstdScheme);

        #[cfg(feature = "pco")]
        let builder = builder
            .with_new_scheme(&integer::PcoScheme)
            .with_new_scheme(&float::PcoScheme);

        builder
    }

    /// Excludes schemes without CUDA kernel support, including Decimal v2, keeps FSST for string
    /// compression, and adds Zstd for binary compression.
    ///
    /// Both the array-level and the buffer-level Zstd schemes are added. Buffer-level
    /// compression preserves binary arrays' buffer layout for zero-conversion GPU decompression,
    /// but belongs to the opt-in `zstd` edition. Set edition permissions with
    /// [`with_allowed_encodings`](Self::with_allowed_encodings) before applying this preset.
    /// This removes Decimal v2 from the stored allowlist so [`Self::build`] selects v1.
    /// Subsequent allowlists can only narrow these permissions. Without an allowlist, the default
    /// Decimal configuration stays on v1 and schemes already producing v2 are removed.
    ///
    /// This preset is intended for files that will be decoded by CUDA kernels. It may choose a
    /// larger encoded representation than the default compressor.
    pub fn only_cuda_compatible(self) -> Self {
        // Keep FSST, which has a CUDA decoder and direct Arrow offset-based export. Other
        // string fragmentation and dictionary schemes still require unsupported decode paths.
        #[cfg_attr(not(any(feature = "pco", feature = "zstd")), allow(unused_mut))]
        let mut excluded: Vec<SchemeId> = vec![
            integer::SparseScheme.id(),
            integer::IntRLEScheme.id(),
            float::ALPRDScheme.id(),
            float::FloatRLEScheme.id(),
            float::NullDominatedSparseScheme.id(),
            string::NullDominatedSparseScheme.id(),
            string::StringDictScheme.id(),
            binary::BinaryDictScheme.id(),
        ];
        // Delta now has a CUDA decode kernel, so arrays that reach the GPU already encoded with
        // it — the Delta children OnPair emits, for instance — decode there. It stays excluded
        // from this preset until GPU delta decode is benchmarked against the schemes it would
        // displace, since the preset picks encodings rather than merely decoding them.
        excluded.push(integer::DeltaScheme::default().id());
        #[cfg(feature = "pco")]
        excluded.extend([integer::PcoScheme.id(), float::PcoScheme.id()]);
        let mut builder = self.exclude_schemes(excluded);
        if let Some(allowed) = &mut builder.allowed_serialized_ids {
            allowed.remove(&decimal_byte_parts_v2_id());
        } else {
            builder
                .schemes
                .retain(|scheme| !scheme.produced_encodings().contains(&decimal_byte_parts_v2_id()));
        }

        #[cfg(feature = "zstd")]
        let builder = builder
            .with_new_scheme(&binary::ZstdScheme)
            .with_new_scheme(&binary::ZstdBuffersScheme);

        builder
    }

    /// Removes the specified compression schemes by their [`SchemeId`].
    ///
    /// Schemes registered after this call may replace the excluded instances.
    pub fn exclude_schemes(mut self, ids: impl IntoIterator<Item = SchemeId>) -> Self {
        let ids: HashSet<_> = ids.into_iter().collect();
        self.schemes.retain(|s| !ids.contains(&s.id()));
        self
    }

    /// Restricts the serialized IDs that compression schemes may produce.
    ///
    /// Repeated calls intersect the allowed sets. An empty set permits no serialized IDs.
    /// [`Self::build`] selects compatible versions and filters all registered schemes, including
    /// schemes added after this call. Apply [`Self::only_cuda_compatible`] after supplying the
    /// initial allowlist so it can remove unsupported formats from these permissions.
    pub fn with_allowed_encodings(mut self, allowed: &HashSet<ArrayId>) -> Self {
        match &mut self.allowed_serialized_ids {
            Some(current) => current.retain(|id| allowed.contains(id)),
            None => self.allowed_serialized_ids = Some(allowed.clone()),
        }
        self
    }

    /// Builds the configured [`BtrBlocksCompressor`].
    ///
    /// When allowed encodings are supplied, selects the latest compatible Decimal version and
    /// removes schemes whose declared outputs are not all permitted. This also reconfigures
    /// explicitly registered Decimal schemes. Without an allowlist, configurations are preserved.
    pub fn build(mut self) -> BtrBlocksCompressor {
        if let Some(allowed) = &self.allowed_serialized_ids {
            for scheme in &mut self.schemes {
                if scheme.id() == decimal::DecimalScheme::default().id() {
                    *scheme = if allowed.contains(&decimal_byte_parts_v2_id()) {
                        &decimal::DecimalScheme::new(true)
                    } else {
                        &decimal::DecimalScheme::new(false)
                    };
                }
            }
            self.schemes
                .retain(|s| s.produced_encodings().iter().all(|id| allowed.contains(id)));
        }
        BtrBlocksCompressor(CascadingCompressor::new(self.schemes))
    }
}

#[cfg(test)]
mod tests;
