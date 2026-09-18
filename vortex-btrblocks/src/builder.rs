// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builder for configuring `BtrBlocksCompressor` instances.

use std::fmt;
use std::sync::Arc;

use vortex_array::ArrayId;
use vortex_decimal_byte_parts::decimal_byte_parts_v1_id;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use crate::BtrBlocksCompressor;
use crate::CascadingCompressor;
use crate::SchemeExt;
use crate::SchemeId;
use crate::SchemeRef;
use crate::schemes::binary;
use crate::schemes::decimal;
use crate::schemes::float;
use crate::schemes::integer;
use crate::schemes::string;
use crate::schemes::temporal;

/// A deferred constructor for one scheme instance.
type SchemeConstructor = dyn Fn(Option<&HashSet<ArrayId>>) -> SchemeRef + Send + Sync;

/// Constructors for all default compression schemes.
///
/// This list is order-sensitive: the builder preserves this order when constructing
/// the final scheme list, so that tie-breaking is deterministic.
pub const ALL_SCHEMES: &[&SchemeConstructor] = &[
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Integer schemes.
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // NOTE: FoR must precede BitPacking to avoid unnecessary patches.
    &|_| Arc::new(integer::FoRScheme),
    // NOTE: ZigZag should precede BitPacking because we don't want negative numbers.
    &|_| Arc::new(integer::ZigZagScheme),
    &|_| Arc::new(integer::BitPackingScheme),
    &|_| Arc::new(integer::SparseScheme),
    &|_| Arc::new(integer::IntDictScheme),
    &|_| Arc::new(integer::RunEndScheme),
    &|_| Arc::new(integer::SequenceScheme),
    &|_| Arc::new(integer::IntRLEScheme),
    // Delta is omitted here: see [`DELTA_SCHEME`].
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Float schemes.
    ////////////////////////////////////////////////////////////////////////////////////////////////
    &|_| Arc::new(float::ALPScheme),
    &|_| Arc::new(float::ALPRDScheme),
    &|_| Arc::new(float::FloatDictScheme),
    &|_| Arc::new(float::NullDominatedSparseScheme),
    &|_| Arc::new(float::FloatRLEScheme),
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // String schemes.
    ////////////////////////////////////////////////////////////////////////////////////////////////
    &|_| Arc::new(string::StringDictScheme),
    // Both string-fragmentation schemes are registered; the sample-based
    // selector keeps whichever is smaller per column.
    &|_| Arc::new(string::FSSTScheme),
    &|_| Arc::new(string::OnPairScheme),
    &|_| Arc::new(string::NullDominatedSparseScheme),
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Binary schemes.
    ////////////////////////////////////////////////////////////////////////////////////////////////
    &|_| Arc::new(binary::BinaryDictScheme),
    &|_| Arc::new(binary::VarBinScheme),
    // Decimal schemes.
    &|allowed_serialized_ids| Arc::new(decimal::DecimalScheme::new(allowed_serialized_ids)),
    // Temporal schemes.
    &|_| Arc::new(temporal::TemporalScheme),
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
#[derive(Clone)]
pub struct BtrBlocksCompressorBuilder {
    /// Constructors run once with the final settings before availability is checked.
    schemes: Vec<Arc<SchemeConstructor>>,
    /// Serialized IDs available to constructors and the availability gate.
    /// `None` permits all IDs; an empty set permits none.
    allowed_serialized_ids: Option<HashSet<ArrayId>>,
    /// For each excluded scheme, the number of registrations present when it was excluded.
    excluded: HashMap<SchemeId, usize>,
}

impl Default for BtrBlocksCompressorBuilder {
    fn default() -> Self {
        Self {
            schemes: ALL_SCHEMES
                .iter()
                .map(|factory| Arc::new(*factory) as _)
                .collect(),
            allowed_serialized_ids: None,
            excluded: HashMap::default(),
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
            excluded: HashMap::default(),
        }
    }

    /// Registers a scheme constructor, called once during [`build`](Self::build) with the final
    /// serialized-ID restrictions. The resulting instance is gated separately by
    /// [`produces_allowed_encodings`](crate::Scheme::produces_allowed_encodings).
    ///
    /// `None` means all serialized IDs are permitted. Constructors return a [`SchemeRef`];
    /// closures may capture settings or an existing scheme instance.
    pub fn with_new_scheme(
        mut self,
        constructor: impl Fn(Option<&HashSet<ArrayId>>) -> SchemeRef + Send + Sync + 'static,
    ) -> Self {
        self.schemes.push(Arc::new(constructor));
        self
    }

    /// Adds compact encoding schemes (Zstd for strings and binary, Pco for numerics).
    ///
    /// This provides better compression ratios than the default, especially for floating-point
    /// heavy datasets. Requires the `zstd` feature. When the `pco` feature is also enabled,
    /// Pco schemes for integers and floats are included.
    ///
    /// Building panics if any enabled compact scheme is registered more than once.
    #[cfg(feature = "zstd")]
    pub fn with_compact(self) -> Self {
        let builder = self
            .with_new_scheme(|_| Arc::new(string::ZstdScheme))
            .with_new_scheme(|_| Arc::new(binary::ZstdScheme));

        #[cfg(feature = "pco")]
        let builder = builder
            .with_new_scheme(|_| Arc::new(integer::PcoScheme))
            .with_new_scheme(|_| Arc::new(float::PcoScheme));

        builder
    }

    /// Excludes schemes without CUDA kernel support, keeps FSST for string compression,
    /// uses single-part decimal byte parts, and adds Zstd for binary compression.
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
        let mut builder = self.exclude_schemes(excluded);
        let decimal_v1: SchemeRef = Arc::new(decimal::DecimalScheme::new(Some(&HashSet::from([
            decimal_byte_parts_v1_id(),
        ]))));
        // Keep Decimal in its original position without adding it to builders that omit it.
        for constructor in &mut builder.schemes {
            let original = Arc::clone(constructor);
            let decimal_v1 = Arc::clone(&decimal_v1);
            *constructor = Arc::new(move |allowed_serialized_ids| {
                let scheme = original(allowed_serialized_ids);
                if scheme.id() == decimal_v1.id() {
                    Arc::clone(&decimal_v1)
                } else {
                    scheme
                }
            });
        }

        #[cfg(feature = "zstd")]
        let builder = builder
            .with_new_scheme(|_| Arc::new(binary::ZstdScheme))
            .with_new_scheme(|_| Arc::new(binary::ZstdBuffersScheme));

        builder
    }

    /// Excludes the specified schemes from registrations already in the builder.
    ///
    /// Schemes registered after this call may replace the excluded instances.
    pub fn exclude_schemes(mut self, ids: impl IntoIterator<Item = SchemeId>) -> Self {
        self.excluded
            .extend(ids.into_iter().map(|id| (id, self.schemes.len())));
        self
    }

    /// Gates schemes by their required serialized IDs and passes the IDs to their constructors.
    ///
    /// Repeated calls intersect the allowed sets. The final restriction applies to all registered
    /// constructors, including those added after this call.
    pub fn retain_allowed_encodings(mut self, allowed: &HashSet<ArrayId>) -> Self {
        match &mut self.allowed_serialized_ids {
            None => self.allowed_serialized_ids = Some(allowed.clone()),
            Some(ids) => ids.retain(|id| allowed.contains(id)),
        }
        self
    }

    /// Builds the configured [`BtrBlocksCompressor`].
    ///
    /// # Panics
    ///
    /// Panics if two enabled constructors return the same [`SchemeId`].
    pub fn build(self) -> BtrBlocksCompressor {
        let mut ids: HashSet<SchemeId> = HashSet::default();
        let schemes = self
            .schemes
            .iter()
            .map(|constructor| constructor(self.allowed_serialized_ids.as_ref()))
            .enumerate()
            .filter(|(index, scheme)| {
                self.excluded
                    .get(&scheme.id())
                    .is_none_or(|cutoff| index >= cutoff)
            })
            .map(|(_, scheme)| scheme)
            .filter(|scheme| {
                self.allowed_serialized_ids
                    .as_ref()
                    .is_none_or(|allowed| scheme.produces_allowed_encodings(allowed))
            })
            .inspect(|scheme| assert!(ids.insert(scheme.id()), "duplicate scheme {}", scheme.id()))
            .collect();
        BtrBlocksCompressor(CascadingCompressor::new(schemes))
    }
}

impl fmt::Debug for BtrBlocksCompressorBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BtrBlocksCompressorBuilder")
            .field("scheme_count", &self.schemes.len())
            .field("allowed_serialized_ids", &self.allowed_serialized_ids)
            .field("excluded", &self.excluded)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

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
        let compressor = builder.build();
        for constructor in ALL_SCHEMES {
            assert!(compressor.has_scheme(constructor(None).id()));
        }
    }

    #[test]
    fn retain_allowed_encodings_filters_schemes() {
        let allowed: HashSet<ArrayId> = [FoR.id()].into_iter().collect();
        let builder = BtrBlocksCompressorBuilder::default()
            .retain_allowed_encodings(&allowed)
            .build();
        assert!(builder.has_scheme(integer::FoRScheme.id()));
        assert!(!builder.has_scheme(integer::BitPackingScheme.id()));

        let none = BtrBlocksCompressorBuilder::default()
            .retain_allowed_encodings(&HashSet::new())
            .build();
        for constructor in ALL_SCHEMES {
            let scheme = constructor(None);
            assert!(!none.has_scheme(scheme.id()));
        }
    }

    #[test]
    fn constructors_receive_final_restrictions_once() {
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&calls);
        let builder = BtrBlocksCompressorBuilder::empty()
            .retain_allowed_encodings(&HashSet::from([FoR.id()]))
            .with_new_scheme(move |allowed_serialized_ids| {
                observed.fetch_add(1, Ordering::Relaxed);
                assert_eq!(allowed_serialized_ids, Some(&HashSet::from([FoR.id()])));
                Arc::new(integer::FoRScheme)
            })
            .with_new_scheme(|_| Arc::new(integer::BitPackingScheme))
            .retain_allowed_encodings(&HashSet::from([FoR.id(), vortex_fastlanes::BitPacked.id()]));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        let compressor = builder.build();
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert!(compressor.has_scheme(integer::FoRScheme.id()));
        assert!(!compressor.has_scheme(integer::BitPackingScheme.id()));
    }

    #[test]
    #[should_panic(expected = "duplicate scheme")]
    fn duplicate_schemes_are_rejected_at_build() {
        BtrBlocksCompressorBuilder::empty()
            .with_new_scheme(|_| Arc::new(integer::FoRScheme))
            .with_new_scheme(|_| Arc::new(integer::FoRScheme))
            .build();
    }

    #[test]
    fn excluded_scheme_can_be_replaced() {
        let id = integer::DeltaScheme::default().id();
        let builder = BtrBlocksCompressorBuilder::default()
            .exclude_schemes([id])
            .with_new_scheme(|_| Arc::new(integer::DeltaScheme::new(2.0)));
        assert!(builder.clone().build().has_scheme(id));
        let builder = builder.exclude_schemes([id]);
        assert!(!builder.clone().build().has_scheme(id));
        assert!(
            builder
                .with_new_scheme(|_| Arc::new(integer::DeltaScheme::new(3.0)))
                .build()
                .has_scheme(id)
        );
    }

    #[test]
    fn default_constructors_can_be_registered() {
        let compressor = ALL_SCHEMES
            .iter()
            .fold(
                BtrBlocksCompressorBuilder::empty(),
                |builder, constructor| builder.with_new_scheme(*constructor),
            )
            .build();
        for constructor in ALL_SCHEMES {
            assert!(compressor.has_scheme(constructor(None).id()));
        }
    }

    #[test]
    fn existing_scheme_ref_can_be_registered() {
        let scheme: SchemeRef = Arc::new(integer::FoRScheme);
        let id = scheme.id();
        let compressor = BtrBlocksCompressorBuilder::empty()
            .with_new_scheme(move |_| Arc::clone(&scheme))
            .build();
        assert!(compressor.has_scheme(id));
    }

    #[test]
    fn cuda_compatible_does_not_add_decimal() {
        let compressor = BtrBlocksCompressorBuilder::empty()
            .only_cuda_compatible()
            .build();
        assert!(!compressor.has_scheme(decimal::DecimalScheme::default().id()));
    }

    #[test]
    fn cuda_compatible_excludes_alprd() {
        let builder = BtrBlocksCompressorBuilder::default()
            .only_cuda_compatible()
            .build();
        assert!(!builder.has_scheme(float::ALPRDScheme.id()));
    }

    /// `vortex.sparse` has no CUDA decode kernel, so no sparse scheme may survive this preset.
    #[test]
    fn cuda_compatible_excludes_every_sparse_scheme() {
        let builder = BtrBlocksCompressorBuilder::default()
            .only_cuda_compatible()
            .build();
        for excluded in [
            integer::SparseScheme.id(),
            float::NullDominatedSparseScheme.id(),
            string::NullDominatedSparseScheme.id(),
        ] {
            assert!(
                !builder.has_scheme(excluded),
                "{excluded} should be excluded"
            );
        }
    }

    #[test]
    fn cuda_compatible_uses_fsst_for_strings() {
        let builder = BtrBlocksCompressorBuilder::default()
            .only_cuda_compatible()
            .build();
        assert!(builder.has_scheme(string::FSSTScheme.id()));
        #[cfg(feature = "zstd")]
        assert!(!builder.has_scheme(string::ZstdScheme.id()));
    }

    #[test]
    #[cfg(feature = "pco")]
    fn cuda_compatible_excludes_pco() {
        let builder = BtrBlocksCompressorBuilder::default()
            .with_new_scheme(|_| Arc::new(integer::PcoScheme))
            .with_new_scheme(|_| Arc::new(float::PcoScheme))
            .only_cuda_compatible()
            .build();
        for scheme in [integer::PcoScheme.id(), float::PcoScheme.id()] {
            assert!(!builder.has_scheme(scheme));
        }
    }
}
