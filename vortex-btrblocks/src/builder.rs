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

/// All default compression schemes, including Decimal v2.
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
    &decimal::DecimalScheme::v2(),
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
/// [`Self::retain_allowed_encodings`] restricts serialized IDs. During [`Self::build`], these
/// restrictions select the compatible Decimal mode and filter all registered schemes. Without
/// an allowlist, registered modes are preserved, including Decimal v2 in the defaults.
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
    forbidden_serialized_ids: HashSet<ArrayId>,
}

impl Default for BtrBlocksCompressorBuilder {
    fn default() -> Self {
        Self {
            schemes: ALL_SCHEMES.to_vec(),
            allowed_serialized_ids: None,
            forbidden_serialized_ids: HashSet::new(),
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
            forbidden_serialized_ids: HashSet::new(),
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
    /// Both the array-level and the buffer-level Zstd schemes are added. Buffer-level
    /// compression preserves binary arrays' buffer layout for zero-conversion GPU decompression,
    /// but belongs to the opt-in `zstd` edition, so callers filter the two through
    /// [`retain_allowed_encodings`](Self::retain_allowed_encodings).
    /// Decimal schemes are restricted to v1 during build, regardless of when the allowlist or
    /// Decimal scheme is supplied, because CUDA does not support lower decimal parts.
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
        builder
            .forbidden_serialized_ids
            .insert(decimal_byte_parts_v2_id());

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

    /// Restricts schemes to those whose produced serialized IDs all belong to `allowed`.
    ///
    /// `allowed` holds serialized IDs. The file writer passes the array IDs its enabled editions
    /// permit. Repeated calls intersect the allowed sets; an empty set permits no serialized IDs.
    ///
    /// Configuration and filtering are deferred until [`Self::build`], including for schemes
    /// registered after this call. Registered Decimal schemes use v2 when both decimal IDs are
    /// allowed, or v1 when only v1 is allowed. If v1 is not allowed, Decimal is removed, since
    /// single-part arrays always serialize as v1. The CUDA preset restricts Decimal to v1.
    pub fn retain_allowed_encodings(mut self, allowed: &HashSet<ArrayId>) -> Self {
        match &mut self.allowed_serialized_ids {
            Some(current) => current.retain(|id| allowed.contains(id)),
            None => self.allowed_serialized_ids = Some(allowed.clone()),
        }
        self
    }

    /// Builds the configured [`BtrBlocksCompressor`].
    pub fn build(self) -> BtrBlocksCompressor {
        BtrBlocksCompressor(CascadingCompressor::new(self.configured_schemes()))
    }

    fn configured_schemes(mut self) -> Vec<&'static dyn Scheme> {
        let allowed = self.allowed_serialized_ids.as_ref();
        let forbidden = &self.forbidden_serialized_ids;
        let is_allowed = |id: &ArrayId| {
            !forbidden.contains(id) && allowed.is_none_or(|allowed| allowed.contains(id))
        };
        if allowed.is_some() || forbidden.contains(&decimal_byte_parts_v2_id()) {
            let use_v2 = is_allowed(&decimal_byte_parts_v2_id());
            for scheme in &mut self.schemes {
                if scheme.id() == decimal::DecimalScheme::v2().id() {
                    *scheme = if use_v2 {
                        &decimal::DecimalScheme::v2()
                    } else {
                        &decimal::DecimalScheme::v1()
                    };
                }
            }
        }
        if allowed.is_some() || !forbidden.is_empty() {
            self.schemes
                .retain(|scheme| scheme.produced_encodings().iter().all(&is_allowed));
        }
        self.schemes
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::VTable;
    use vortex_fastlanes::FoR;

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
        let schemes = BtrBlocksCompressorBuilder::default()
            .retain_allowed_encodings(&allowed)
            .configured_schemes();
        assert_eq!(schemes.len(), 1);
        assert_eq!(schemes[0].id(), integer::FoRScheme.id());

        let none = BtrBlocksCompressorBuilder::default()
            .retain_allowed_encodings(&HashSet::new())
            .configured_schemes();
        assert!(none.is_empty());
    }

    #[test]
    fn retaining_all_declared_outputs_keeps_every_scheme() {
        let allowed: HashSet<ArrayId> = ALL_SCHEMES
            .iter()
            .flat_map(|scheme| scheme.produced_encodings())
            .collect();
        let schemes = BtrBlocksCompressorBuilder::default()
            .retain_allowed_encodings(&allowed)
            .configured_schemes();
        assert_eq!(schemes, ALL_SCHEMES);
    }

    #[rstest]
    fn restrictions_apply_to_later_registrations(#[values(false, true)] register_later: bool) {
        let mut builder = BtrBlocksCompressorBuilder::empty();
        if !register_later {
            builder = builder.with_new_scheme(&integer::FoRScheme);
        }
        builder = builder.retain_allowed_encodings(&HashSet::new());
        if register_later {
            builder = builder.with_new_scheme(&integer::FoRScheme);
        }
        let allowed = integer::FoRScheme.produced_encodings().into_iter().collect();
        assert!(
            builder
                .retain_allowed_encodings(&allowed)
                .configured_schemes()
                .is_empty()
        );
    }

    #[rstest]
    fn forbidden_ids_filter_schemes(
        #[values(false, true)] with_allowlist: bool,
        #[values(false, true)] register_later: bool,
    ) {
        let mut builder = BtrBlocksCompressorBuilder::empty();
        if !register_later {
            builder = builder.with_new_scheme(&integer::FoRScheme);
        }
        builder.forbidden_serialized_ids.insert(FoR.id());
        if with_allowlist {
            builder = builder.retain_allowed_encodings(&HashSet::from([FoR.id()]));
        }
        if register_later {
            builder = builder.with_new_scheme(&integer::FoRScheme);
        }
        assert!(builder.configured_schemes().is_empty());
    }

    #[test]
    fn unrelated_forbidden_ids_preserve_explicit_decimal_mode() {
        let mut builder = BtrBlocksCompressorBuilder::empty()
            .with_new_scheme(&decimal::DecimalScheme::v1());
        builder.forbidden_serialized_ids.insert(FoR.id());
        let schemes = builder.configured_schemes();
        assert_eq!(schemes.len(), 1);
        assert_eq!(
            schemes[0].produced_encodings(),
            decimal::DecimalScheme::v1().produced_encodings()
        );
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
