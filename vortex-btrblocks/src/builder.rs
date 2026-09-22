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
    &decimal::DecimalScheme,
    // Temporal schemes.
    &temporal::TemporalScheme,
];

/// Delta, kept out of [`ALL_SCHEMES`] because it is slower to decompress than the schemes that
/// would otherwise win. Callers that want it opt in with
/// [`with_new_scheme`](BtrBlocksCompressorBuilder::with_new_scheme).
///
/// TODO(robert): Return it to [`ALL_SCHEMES`] once we have scheme filtering.
pub static DELTA_SCHEME: integer::DeltaScheme = integer::DeltaScheme::new(1.25);

/// Builder for creating configured [`BtrBlocksCompressor`] instances.
///
/// By default, all schemes in [`ALL_SCHEMES`] are enabled in a deterministic order. Feature-gated
/// schemes (Pco, Zstd) are not in `ALL_SCHEMES` and must be added explicitly via
/// [`with_new_scheme`](BtrBlocksCompressorBuilder::with_new_scheme) or `with_compact` when the
/// `zstd` feature is enabled.
///
/// [`retain_allowed_encodings`](Self::retain_allowed_encodings) and
/// [`exclude_encodings`](Self::exclude_encodings) restrict the serialized IDs the compressor may
/// emit. Both apply in [`build`](Self::build), so they cover schemes registered afterwards and may
/// be called in any order.
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
    /// Serialized IDs the writer permits, or `None` when no writer restriction was supplied.
    allowed_encodings: Option<HashSet<ArrayId>>,
    /// Serialized IDs denied regardless of what the writer permits.
    excluded_encodings: HashSet<ArrayId>,
}

impl Default for BtrBlocksCompressorBuilder {
    fn default() -> Self {
        Self {
            schemes: ALL_SCHEMES.to_vec(),
            allowed_encodings: None,
            excluded_encodings: HashSet::new(),
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
            allowed_encodings: None,
            excluded_encodings: HashSet::new(),
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

    /// Excludes schemes and wire formats without CUDA kernel support, keeps FSST for string
    /// compression, and adds Zstd for binary compression.
    ///
    /// Both the array-level and the buffer-level Zstd schemes are added. Buffer-level
    /// compression preserves binary arrays' buffer layout for zero-conversion GPU decompression,
    /// but belongs to the opt-in `zstd` edition, so callers filter the two through
    /// [`retain_allowed_encodings`](Self::retain_allowed_encodings).
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
        // Multi-part DecimalByteParts arrays have no CUDA decode kernel, so wide decimals stay
        // canonical while single-part decimals still compress under the v1 format.
        let builder = self
            .exclude_schemes(excluded)
            .exclude_encodings([decimal_byte_parts_v2_id()]);

        #[cfg(feature = "zstd")]
        let builder = builder
            .with_new_scheme(&binary::ZstdScheme)
            .with_new_scheme(&binary::ZstdBuffersScheme);

        builder
    }

    /// Removes the specified compression schemes by their [`SchemeId`].
    pub fn exclude_schemes(mut self, ids: impl IntoIterator<Item = SchemeId>) -> Self {
        let ids: HashSet<_> = ids.into_iter().collect();
        self.schemes.retain(|s| !ids.contains(&s.id()));
        self
    }

    /// Retains only schemes whose produced serialized IDs all belong to `allowed`, and permits
    /// `allowed` to the schemes that remain.
    ///
    /// `allowed` holds serialized IDs. The file writer passes the IDs its enabled editions permit.
    /// Repeated calls intersect. The restriction applies in [`build`](Self::build), so it covers
    /// schemes registered after this call, and the remaining schemes only emit optional wire
    /// formats the set contains.
    pub fn retain_allowed_encodings(mut self, allowed: &HashSet<ArrayId>) -> Self {
        match &mut self.allowed_encodings {
            Some(current) => current.retain(|id| allowed.contains(id)),
            None => self.allowed_encodings = Some(allowed.clone()),
        }
        self
    }

    /// Denies the given serialized IDs, whatever
    /// [`retain_allowed_encodings`](Self::retain_allowed_encodings) permits.
    ///
    /// Like `retain_allowed_encodings`, this applies in [`build`](Self::build).
    pub fn exclude_encodings(mut self, ids: impl IntoIterator<Item = ArrayId>) -> Self {
        self.excluded_encodings.extend(ids);
        self
    }

    /// Builds the configured [`BtrBlocksCompressor`].
    ///
    /// Without [`retain_allowed_encodings`](Self::retain_allowed_encodings), the compressor may
    /// emit exactly the serialized IDs its schemes declare, so optional wire formats need explicit
    /// permission.
    pub fn build(self) -> BtrBlocksCompressor {
        let compressor = CascadingCompressor::new(self.schemes);
        let mut allowed = self
            .allowed_encodings
            .unwrap_or_else(|| compressor.allowed_serialized_ids().clone());
        allowed.retain(|id| !self.excluded_encodings.contains(id));
        BtrBlocksCompressor(compressor.with_allowed_serialized_ids(allowed))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::VTable;
    use vortex_decimal_byte_parts::decimal_byte_parts_v1_id;
    use vortex_fastlanes::FoR;

    use super::*;

    fn declared_ids() -> HashSet<ArrayId> {
        ALL_SCHEMES
            .iter()
            .flat_map(|scheme| scheme.produced_encodings())
            .collect()
    }

    #[test]
    fn empty_starts_with_no_schemes() {
        let builder = BtrBlocksCompressorBuilder::empty();
        assert!(builder.schemes.is_empty());
    }

    #[test]
    fn default_includes_all_schemes() {
        let compressor = BtrBlocksCompressorBuilder::default().build();
        for scheme in ALL_SCHEMES {
            assert!(compressor.has_scheme(scheme.id()));
        }
    }

    /// Without a writer allowlist the compressor may emit what its schemes declare and nothing
    /// more, so an optional wire format such as DecimalByteParts v2 stays off.
    #[test]
    fn default_permits_only_declared_ids() {
        let compressor = BtrBlocksCompressorBuilder::default().build();
        assert_eq!(compressor.allowed_serialized_ids(), &declared_ids());
        assert!(
            !compressor
                .allowed_serialized_ids()
                .contains(&decimal_byte_parts_v2_id())
        );
    }

    #[test]
    fn allowed_encodings_filter_schemes_on_build() {
        let allowed: HashSet<ArrayId> = [FoR.id()].into_iter().collect();
        let compressor = BtrBlocksCompressorBuilder::default()
            .retain_allowed_encodings(&allowed)
            .build();
        assert!(compressor.has_scheme(integer::FoRScheme.id()));
        assert!(!compressor.has_scheme(integer::BitPackingScheme.id()));

        let none = BtrBlocksCompressorBuilder::default()
            .retain_allowed_encodings(&HashSet::new())
            .build();
        for scheme in ALL_SCHEMES {
            assert!(!none.has_scheme(scheme.id()));
        }
    }

    #[test]
    fn allowing_all_declared_ids_keeps_every_scheme() {
        let compressor = BtrBlocksCompressorBuilder::default()
            .retain_allowed_encodings(&declared_ids())
            .build();
        for scheme in ALL_SCHEMES {
            assert!(compressor.has_scheme(scheme.id()));
        }
    }

    #[rstest]
    fn allowed_encodings_apply_regardless_of_registration_order(
        #[values(false, true)] permitted: bool,
        #[values(false, true)] register_later: bool,
    ) {
        let allowed = if permitted {
            HashSet::from([FoR.id()])
        } else {
            HashSet::new()
        };
        let mut builder = BtrBlocksCompressorBuilder::empty();
        if !register_later {
            builder = builder.with_new_scheme(&integer::FoRScheme);
        }
        builder = builder.retain_allowed_encodings(&allowed);
        if register_later {
            builder = builder.with_new_scheme(&integer::FoRScheme);
        }
        assert_eq!(
            builder.build().has_scheme(integer::FoRScheme.id()),
            permitted
        );
    }

    #[rstest]
    fn allowed_encodings_intersect(#[values(false, true)] restrictive_first: bool) {
        let all = declared_ids();
        let restricted = HashSet::from([FoR.id()]);
        let (first, second) = if restrictive_first {
            (&restricted, &all)
        } else {
            (&all, &restricted)
        };
        let compressor = BtrBlocksCompressorBuilder::default()
            .retain_allowed_encodings(first)
            .retain_allowed_encodings(second)
            .build();
        assert!(compressor.has_scheme(integer::FoRScheme.id()));
        assert!(!compressor.has_scheme(integer::BitPackingScheme.id()));
    }

    /// A permitted optional format reaches the compressor's allowlist without being declared by any
    /// scheme, so schemes can pick it up through the compression context.
    #[test]
    fn allowed_encodings_permit_optional_ids() {
        let compressor = BtrBlocksCompressorBuilder::default()
            .retain_allowed_encodings(&HashSet::from([
                decimal_byte_parts_v1_id(),
                decimal_byte_parts_v2_id(),
            ]))
            .build();
        assert!(compressor.has_scheme(decimal::DecimalScheme.id()));
        assert!(
            compressor
                .allowed_serialized_ids()
                .contains(&decimal_byte_parts_v2_id())
        );
    }

    #[test]
    fn excluded_encodings_override_allowed_encodings() {
        let mut allowed = declared_ids();
        allowed.insert(decimal_byte_parts_v2_id());
        let compressor = BtrBlocksCompressorBuilder::default()
            .exclude_encodings([decimal_byte_parts_v2_id(), FoR.id()])
            .retain_allowed_encodings(&allowed)
            .build();
        // Denying an optional format keeps the scheme but withholds the format.
        assert!(compressor.has_scheme(decimal::DecimalScheme.id()));
        assert!(
            !compressor
                .allowed_serialized_ids()
                .contains(&decimal_byte_parts_v2_id())
        );
        // Denying a declared format drops the scheme that needs it.
        assert!(!compressor.has_scheme(integer::FoRScheme.id()));
    }

    #[rstest]
    #[case::without_allowlist(None)]
    #[case::allowlist_first(Some(true))]
    #[case::allowlist_last(Some(false))]
    fn cuda_compatible_denies_decimal_v2(#[case] allowlist_first: Option<bool>) {
        let allowed = HashSet::from([decimal_byte_parts_v1_id(), decimal_byte_parts_v2_id()]);
        let builder = match allowlist_first {
            None => BtrBlocksCompressorBuilder::default().only_cuda_compatible(),
            Some(true) => BtrBlocksCompressorBuilder::default()
                .retain_allowed_encodings(&allowed)
                .only_cuda_compatible(),
            Some(false) => BtrBlocksCompressorBuilder::default()
                .only_cuda_compatible()
                .retain_allowed_encodings(&allowed),
        };
        let compressor = builder.build();
        assert!(compressor.has_scheme(decimal::DecimalScheme.id()));
        assert!(
            !compressor
                .allowed_serialized_ids()
                .contains(&decimal_byte_parts_v2_id())
        );
    }

    #[test]
    fn cuda_compatible_excludes_alprd() {
        let compressor = BtrBlocksCompressorBuilder::default()
            .only_cuda_compatible()
            .build();
        assert!(!compressor.has_scheme(float::ALPRDScheme.id()));
    }

    /// `vortex.sparse` has no CUDA decode kernel, so no sparse scheme may survive this preset.
    #[test]
    fn cuda_compatible_excludes_every_sparse_scheme() {
        let compressor = BtrBlocksCompressorBuilder::default()
            .only_cuda_compatible()
            .build();
        for excluded in [
            integer::SparseScheme.id(),
            float::NullDominatedSparseScheme.id(),
            string::NullDominatedSparseScheme.id(),
        ] {
            assert!(
                !compressor.has_scheme(excluded),
                "{excluded} should be excluded"
            );
        }
    }

    #[test]
    fn cuda_compatible_uses_fsst_for_strings() {
        let compressor = BtrBlocksCompressorBuilder::default()
            .only_cuda_compatible()
            .build();
        assert!(compressor.has_scheme(string::FSSTScheme.id()));
        #[cfg(feature = "zstd")]
        assert!(!compressor.has_scheme(string::ZstdScheme.id()));
    }

    #[test]
    #[cfg(feature = "pco")]
    fn cuda_compatible_excludes_pco() {
        let compressor = BtrBlocksCompressorBuilder::default()
            .with_new_scheme(&integer::PcoScheme)
            .with_new_scheme(&float::PcoScheme)
            .only_cuda_compatible()
            .build();
        for scheme in [integer::PcoScheme.id(), float::PcoScheme.id()] {
            assert!(!compressor.has_scheme(scheme));
        }
    }
}
