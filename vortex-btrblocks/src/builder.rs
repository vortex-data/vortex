// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builder for configuring `BtrBlocksCompressor` instances.

use vortex_array::ArrayId;
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

/// All available compression schemes.
///
/// This list is order-sensitive: the builder preserves this order when constructing
/// the final scheme list, so that tie-breaking is deterministic.
pub const ALL_SCHEMES: &[&dyn Scheme] = &[
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
    // Prefer all other schemes above delta, for now (since its slower to decompress).
    #[cfg(feature = "unstable_encodings")]
    &integer::DeltaScheme::new(1.25),
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
    #[cfg(feature = "unstable_encodings")]
    &string::OnPairScheme,
    &string::NullDominatedSparseScheme,
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Binary schemes.
    ////////////////////////////////////////////////////////////////////////////////////////////////
    &binary::BinaryDictScheme,
    // Decimal schemes.
    &decimal::DecimalScheme,
    // Temporal schemes.
    &temporal::TemporalScheme,
];

/// Builder for creating configured [`BtrBlocksCompressor`] instances.
///
/// By default, all schemes in [`ALL_SCHEMES`] are enabled in a deterministic order. Feature-gated
/// schemes (Pco, Zstd) are not in `ALL_SCHEMES` and must be added explicitly via
/// [`with_new_scheme`](BtrBlocksCompressorBuilder::with_new_scheme) or `with_compact` when the
/// `zstd` feature is enabled.
///
/// # Examples
///
/// ```rust
/// use vortex_btrblocks::{BtrBlocksCompressorBuilder, Scheme, SchemeExt};
/// use vortex_btrblocks::schemes::integer::IntDictScheme;
///
/// // Default compressor with all schemes in ALL_SCHEMES.
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
            schemes: ALL_SCHEMES.to_vec(),
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

    /// Adds an external compression scheme not in [`ALL_SCHEMES`].
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

    /// Excludes schemes without CUDA kernel support, keeps FSST for string compression,
    /// and adds Zstd for binary compression.
    ///
    /// With the `unstable_encodings` feature, buffer-level Zstd compression is used for binary
    /// arrays, preserving their buffer layout for zero-conversion GPU decompression. Without it,
    /// interleaved binary Zstd compression is used.
    ///
    /// This preset is intended for files that will be decoded by CUDA kernels. It may choose a
    /// larger encoded representation than the default compressor.
    pub fn only_cuda_compatible(self) -> Self {
        // Keep FSST, which has a CUDA decoder and direct Arrow offset-based export. Other
        // string fragmentation and dictionary schemes still require unsupported decode paths.
        #[cfg_attr(
            not(any(feature = "pco", feature = "unstable_encodings")),
            allow(unused_mut)
        )]
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
        #[cfg(feature = "unstable_encodings")]
        excluded.push(integer::DeltaScheme::default().id());
        #[cfg(feature = "pco")]
        excluded.extend([integer::PcoScheme.id(), float::PcoScheme.id()]);
        let builder = self.exclude_schemes(excluded);
        // CUDA kernels decode each encoding's original wire format, whose ID is the encoding's
        // own. In particular bit-unpacking has no per-chunk width kernel yet, so allowing only
        // the original formats keeps one width per array.
        let original_formats = builder
            .schemes
            .iter()
            .flat_map(|scheme| scheme.produced_encodings())
            .collect();
        let builder = builder.allow_serialized_ids(&original_formats);

        #[cfg(all(feature = "zstd", feature = "unstable_encodings"))]
        let builder = builder.with_new_scheme(&binary::ZstdBuffersScheme);
        #[cfg(all(feature = "zstd", not(feature = "unstable_encodings")))]
        let builder = builder.with_new_scheme(&binary::ZstdScheme);

        builder
    }

    /// Removes the specified compression schemes by their [`SchemeId`].
    pub fn exclude_schemes(mut self, ids: impl IntoIterator<Item = SchemeId>) -> Self {
        let ids: HashSet<_> = ids.into_iter().collect();
        self.schemes.retain(|s| !ids.contains(&s.id()));
        self
    }

    /// Retains only schemes whose produced encodings all belong to `allowed`.
    ///
    /// The file writer uses this to restrict compression to the encodings of its configured
    /// editions.
    pub fn retain_allowed_encodings(mut self, allowed: &HashSet<ArrayId>) -> Self {
        self.schemes
            .retain(|s| s.produced_encodings().iter().all(|id| allowed.contains(id)));
        self
    }

    /// Restricts the output to the serialized IDs in `allowed`, intersecting with any earlier
    /// restriction, so a scheme whose encoding has several wire formats produces the newest one
    /// still allowed.
    ///
    /// The file writer passes the serialized IDs of its configured editions.
    pub fn allow_serialized_ids(mut self, allowed: &HashSet<ArrayId>) -> Self {
        self.allowed_serialized_ids = Some(match self.allowed_serialized_ids.take() {
            Some(existing) => existing.intersection(allowed).copied().collect(),
            None => allowed.clone(),
        });
        self
    }

    /// Builds the configured [`BtrBlocksCompressor`].
    pub fn build(self) -> BtrBlocksCompressor {
        let compressor = CascadingCompressor::new(self.schemes);
        BtrBlocksCompressor(match self.allowed_serialized_ids {
            Some(allowed) => compressor.with_allowed_serialized_ids(allowed),
            None => compressor,
        })
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::VTable;
    use vortex_fastlanes::BitPacked;
    use vortex_fastlanes::FoR;
    use vortex_fastlanes::bitpacked_v2_id;

    use super::*;

    #[test]
    fn empty_starts_with_no_schemes() {
        let builder = BtrBlocksCompressorBuilder::empty();
        assert!(builder.schemes.is_empty());
    }

    #[test]
    fn default_includes_all_schemes() {
        let builder = BtrBlocksCompressorBuilder::default();
        assert_eq!(builder.schemes.len(), ALL_SCHEMES.len());
    }

    #[test]
    fn retain_allowed_encodings_filters_schemes() {
        let allowed: HashSet<ArrayId> = [FoR.id()].into_iter().collect();
        let builder = BtrBlocksCompressorBuilder::default().retain_allowed_encodings(&allowed);
        assert_eq!(builder.schemes.len(), 1);
        assert_eq!(builder.schemes[0].id(), integer::FoRScheme.id());

        let none = BtrBlocksCompressorBuilder::default().retain_allowed_encodings(&HashSet::new());
        assert!(none.schemes.is_empty());
    }

    #[test]
    fn retaining_all_declared_outputs_keeps_every_scheme() {
        let allowed: HashSet<ArrayId> = ALL_SCHEMES
            .iter()
            .flat_map(|scheme| scheme.produced_encodings())
            .collect();
        let builder = BtrBlocksCompressorBuilder::default().retain_allowed_encodings(&allowed);
        assert_eq!(builder.schemes.len(), ALL_SCHEMES.len());
    }

    #[test]
    fn cuda_compatible_excludes_alprd() {
        let builder = BtrBlocksCompressorBuilder::default().only_cuda_compatible();
        assert!(
            !builder
                .schemes
                .iter()
                .any(|s| s.id() == float::ALPRDScheme.id())
        );
    }

    /// `vortex.sparse` has no CUDA decode kernel, so no sparse scheme may survive this preset.
    #[test]
    fn cuda_compatible_excludes_every_sparse_scheme() {
        let builder = BtrBlocksCompressorBuilder::default().only_cuda_compatible();
        for excluded in [
            integer::SparseScheme.id(),
            float::NullDominatedSparseScheme.id(),
            string::NullDominatedSparseScheme.id(),
        ] {
            assert!(
                !builder.schemes.iter().any(|s| s.id() == excluded),
                "{excluded} should be excluded"
            );
        }
    }

    /// Every serialized ID is allowed by default, so bit-packing picks per-chunk widths. The CUDA
    /// preset has no per-chunk kernel and allows only the original formats.
    #[test]
    fn cuda_compatible_disallows_per_chunk_bitpacking() {
        let default = BtrBlocksCompressorBuilder::default().build();
        assert!(default.0.allows_serialized_id(bitpacked_v2_id()));

        let cuda = BtrBlocksCompressorBuilder::default()
            .only_cuda_compatible()
            .build();
        assert!(!cuda.0.allows_serialized_id(bitpacked_v2_id()));
        assert!(cuda.0.allows_serialized_id(BitPacked.id()));
    }

    #[test]
    fn cuda_compatible_uses_fsst_for_strings() {
        let builder = BtrBlocksCompressorBuilder::default().only_cuda_compatible();
        assert!(
            builder
                .schemes
                .iter()
                .any(|scheme| scheme.id() == string::FSSTScheme.id())
        );
        #[cfg(feature = "zstd")]
        assert!(
            !builder
                .schemes
                .iter()
                .any(|scheme| scheme.id() == string::ZstdScheme.id())
        );
    }

    #[test]
    #[cfg(feature = "pco")]
    fn cuda_compatible_excludes_pco() {
        let builder = BtrBlocksCompressorBuilder::default()
            .with_new_scheme(&integer::PcoScheme)
            .with_new_scheme(&float::PcoScheme)
            .only_cuda_compatible();
        for scheme in [integer::PcoScheme.id(), float::PcoScheme.id()] {
            assert!(!builder.schemes.iter().any(|s| s.id() == scheme));
        }
    }
}
