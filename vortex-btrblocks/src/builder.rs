// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builder for configuring `BtrBlocksCompressor` instances.

use std::sync::Arc;

use vortex_compressor::cost::CostModel;
#[cfg(feature = "unstable_encodings")]
use vortex_compressor::cost::SchemePrior;
use vortex_compressor::cost::SizeCost;
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
    // Delta's selection policy (penalty + floor) lives in `default_cost_model`.
    #[cfg(feature = "unstable_encodings")]
    &integer::DELTA_SCHEME,
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
    cost_model: Arc<dyn CostModel>,
}

impl Default for BtrBlocksCompressorBuilder {
    fn default() -> Self {
        Self {
            schemes: ALL_SCHEMES.to_vec(),
            cost_model: Arc::new(default_cost_model()),
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
            cost_model: Arc::new(default_cost_model()),
        }
    }

    /// Replaces the cost model used to price candidates during scheme selection.
    ///
    /// The default is the btrblocks default model: [`SizeCost`] with Delta's selection
    /// prior. The model is attached to the compressor at [`build`](Self::build); the file
    /// writer's strategy builder carries it to both its data and stats compressors.
    pub fn with_cost_model(mut self, cost_model: Arc<dyn CostModel>) -> Self {
        self.cost_model = cost_model;
        self
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

    /// Excludes schemes without CUDA kernel support and adds Zstd for string and binary compression.
    ///
    /// With the `unstable_encodings` feature, buffer-level Zstd compression is used which
    /// preserves the array buffer layout for zero-conversion GPU decompression. Without it,
    /// interleaved Zstd compression is used.
    ///
    /// This preset is intended for files that will be decoded by CUDA kernels. It may choose a
    /// larger encoded representation than the default compressor.
    pub fn only_cuda_compatible(self) -> Self {
        // String fragmentation schemes (OnPair, FSST) require host-side
        // dictionary expansion at decode time, which is incompatible with
        // pure-GPU decompression paths. Strip whichever string-fragment
        // scheme is enabled by feature.
        #[cfg_attr(not(feature = "unstable_encodings"), allow(unused_mut))]
        let mut excluded: Vec<SchemeId> = vec![
            integer::SparseScheme.id(),
            integer::IntRLEScheme.id(),
            float::FloatRLEScheme.id(),
            float::NullDominatedSparseScheme.id(),
            string::StringDictScheme.id(),
            string::FSSTScheme.id(),
            binary::BinaryDictScheme.id(),
        ];
        #[cfg(feature = "unstable_encodings")]
        excluded.push(string::OnPairScheme.id());
        // Delta has no GPU decode kernel and its prefix-sum decode is inherently sequential, so it
        // is incompatible with pure-GPU decompression paths.
        #[cfg(feature = "unstable_encodings")]
        excluded.push(integer::DeltaScheme::default().id());
        let builder = self.exclude_schemes(excluded);

        #[cfg(all(feature = "zstd", feature = "unstable_encodings"))]
        let builder = builder
            .with_new_scheme(&string::ZstdBuffersScheme)
            .with_new_scheme(&binary::ZstdBuffersScheme);
        #[cfg(all(feature = "zstd", not(feature = "unstable_encodings")))]
        let builder = builder
            .with_new_scheme(&string::ZstdScheme)
            .with_new_scheme(&binary::ZstdScheme);

        builder
    }

    /// Removes the specified compression schemes by their [`SchemeId`].
    pub fn exclude_schemes(mut self, ids: impl IntoIterator<Item = SchemeId>) -> Self {
        let ids: HashSet<_> = ids.into_iter().collect();
        self.schemes.retain(|s| !ids.contains(&s.id()));
        self
    }

    /// Builds the configured [`BtrBlocksCompressor`].
    pub fn build(self) -> BtrBlocksCompressor {
        BtrBlocksCompressor(CascadingCompressor::new(self.schemes).with_cost_model(self.cost_model))
    }
}

/// The default cost model for btrblocks compressors: ratio-argmax ([`SizeCost`]) with Delta's
/// selection prior.
///
/// Prefer all other schemes above Delta unless it wins by a real margin: Delta is slower to
/// decompress (it breaks random access and adds a prefix-sum decode pass), so its raw ratio
/// is handicapped by the "delta tax" multiplier and gated behind a minimum effective ratio.
/// This prior is **policy, not measurement** — the scheme reports raw ratios, and this is
/// the one place the judgment lives.
pub(crate) fn default_cost_model() -> SizeCost {
    let model = SizeCost::default();
    #[cfg(feature = "unstable_encodings")]
    let model = model.with_scheme_prior(
        integer::DELTA_SCHEME.id(),
        SchemePrior {
            multiplier: integer::DELTA_PENALTY,
            min_ratio: integer::DELTA_MIN_RATIO,
        },
    );
    model
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_compressor::cost::Candidate;
    use vortex_compressor::cost::Cost;
    use vortex_compressor::scheme::CandidateEstimate;
    use vortex_compressor::scheme::SchemeEvaluation;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::*;
    use crate::ArrayAndStats;
    use crate::CompressorContext;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

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

    /// Inverts [`SizeCost`]'s preferences, so the *worst*-priced candidate wins. Only useful
    /// for proving that an injected model actually drives selection.
    #[derive(Debug)]
    struct InvertedSizeCost;

    impl CostModel for InvertedSizeCost {
        fn cost(&self, candidate: &Candidate<'_>) -> Option<Cost> {
            SizeCost::default()
                .cost(candidate)
                .map(|cost| Cost::new(-cost.value()))
        }

        fn canonical_cost(&self, _data: &ArrayAndStats, _n_values: u64) -> Cost {
            Cost::new(f64::MAX)
        }
    }

    /// Fixed-ratio scheme whose output is a constant array carrying `marker`, so the winner
    /// is observable from the compressed output.
    #[derive(Debug)]
    struct MarkerScheme {
        name: &'static str,
        ratio: f64,
        marker: i32,
    }

    impl Scheme for MarkerScheme {
        fn scheme_name(&self) -> &'static str {
            self.name
        }

        fn matches(&self, canonical: &Canonical) -> bool {
            canonical.dtype().is_int()
        }

        fn evaluate(
            &self,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> SchemeEvaluation {
            SchemeEvaluation::Candidate(CandidateEstimate::from_compression_ratio(self.ratio))
        }

        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            Ok(ConstantArray::new(
                Scalar::primitive(self.marker, Nullability::NonNullable),
                data.array_len(),
            )
            .into_array())
        }
    }

    static HIGH_RATIO: MarkerScheme = MarkerScheme {
        name: "test.high_ratio",
        ratio: 3.0,
        marker: 111,
    };
    static LOW_RATIO: MarkerScheme = MarkerScheme {
        name: "test.low_ratio",
        ratio: 2.0,
        marker: 222,
    };

    /// A model injected via [`BtrBlocksCompressorBuilder::with_cost_model`] must actually
    /// drive selection: under the default model the higher ratio wins, under the inverted
    /// model the lower ratio wins.
    #[test]
    fn cost_model_plumbing_is_live() -> VortexResult<()> {
        let array =
            PrimitiveArray::new(buffer![5i32, 6, 7, 8, 9, 10], Validity::NonNullable).into_array();
        let marker = |scheme: &MarkerScheme| {
            Some(Scalar::primitive(scheme.marker, Nullability::NonNullable))
        };

        let compressed = BtrBlocksCompressorBuilder::empty()
            .with_new_scheme(&HIGH_RATIO)
            .with_new_scheme(&LOW_RATIO)
            .build()
            .compress(&array, &mut SESSION.create_execution_ctx())?;
        assert_eq!(compressed.as_constant(), marker(&HIGH_RATIO));

        let compressed = BtrBlocksCompressorBuilder::empty()
            .with_new_scheme(&HIGH_RATIO)
            .with_new_scheme(&LOW_RATIO)
            .with_cost_model(Arc::new(InvertedSizeCost))
            .build()
            .compress(&array, &mut SESSION.create_execution_ctx())?;
        assert_eq!(compressed.as_constant(), marker(&LOW_RATIO));

        Ok(())
    }
}
