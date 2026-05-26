// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cascading array compression implementation.

use vortex_array::ArrayRef;
use vortex_array::ArraySlots;
use vortex_array::Canonical;
use vortex_array::CanonicalValidity;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::Variant;
use vortex_array::arrays::VariantArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::list::ListArrayExt;
use vortex_array::arrays::listview::ListViewArrayExt;
use vortex_array::arrays::listview::list_from_list_view;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::scalar_fn::AnyScalarFn;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::arrays::variant::VariantArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::builtins::IntDictScheme;
use crate::ctx::CompressorContext;
use crate::estimate::CompressionEstimate;
use crate::estimate::DeferredEstimate;
use crate::estimate::EstimateScore;
use crate::estimate::EstimateVerdict;
use crate::estimate::WinnerEstimate;
use crate::estimate::estimate_compression_ratio_with_sampling;
use crate::estimate::is_better_score;
use crate::scheme::ChildSelection;
use crate::scheme::DescendantExclusion;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::scheme::SchemeId;
use crate::stats::ArrayAndStats;
use crate::stats::GenerateStatsOptions;
use crate::trace;

/// Synthetic scheme ID used for the compressor's own root-level cascading.
pub(crate) const ROOT_SCHEME_ID: SchemeId = SchemeId {
    name: "vortex.compressor.root",
};

/// Child indices for the compressor's list/listview compression.
mod root_list_children {
    /// List/ListView offsets child.
    pub const OFFSETS: usize = 1;
    /// ListView sizes child.
    pub const SIZES: usize = 2;
}

/// The main compressor type implementing cascading adaptive compression.
///
/// This compressor applies adaptive compression [`Scheme`]s to arrays based on their data types and
/// characteristics. It recursively compresses nested structures like structs and lists, and chooses
/// optimal compression schemes for leaf types.
///
/// The compressor works by:
/// 1. Canonicalizing input arrays to a standard representation.
/// 2. Pre-filtering schemes by [`Scheme::matches`] and exclusion rules.
/// 3. Evaluating each matching scheme's compression estimate and resolving deferred work.
/// 4. Compressing with the best scheme and verifying the result is smaller.
///
/// No scheme may appear twice in a cascade chain. The compressor enforces this automatically
/// along with push/pull exclusion rules declared by each scheme.
#[derive(Debug, Clone)]
pub struct CascadingCompressor {
    /// The enabled compression schemes.
    schemes: Vec<&'static dyn Scheme>,

    /// Descendant exclusion rules for the compressor's own cascading (e.g. excluding Dict from
    /// list offsets).
    root_exclusions: Vec<DescendantExclusion>,
}

impl CascadingCompressor {
    /// Creates a new compressor with the given schemes.
    ///
    /// Root-level exclusion rules (e.g. excluding Dict from list offsets) are built
    /// automatically.
    pub fn new(schemes: Vec<&'static dyn Scheme>) -> Self {
        // Root exclusion: exclude IntDict from list/listview offsets (monotonically
        // increasing data where dictionary encoding is wasteful).
        let root_exclusions = vec![DescendantExclusion {
            excluded: IntDictScheme.id(),
            children: ChildSelection::One(root_list_children::OFFSETS),
        }];
        Self {
            schemes,
            root_exclusions,
        }
    }

    /// Compresses an array using cascading adaptive compression.
    ///
    /// First canonicalizes and compacts the array, then applies optimal compression schemes.
    ///
    /// # Errors
    ///
    /// Returns an error if canonicalization or compression fails.
    pub fn compress(
        &self,
        array: &ArrayRef,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let before_nbytes = array.nbytes();
        let span = trace::compress_span(array.len(), array.dtype(), before_nbytes);
        let _enter = span.enter();

        let canonical = array.clone().execute::<CanonicalValidity>(exec_ctx)?.0;
        let compact = canonical.compact()?;
        let compressed = self.compress_canonical(compact, CompressorContext::new(), exec_ctx)?;

        trace::record_compress_outcome(&span, before_nbytes, compressed.nbytes());

        Ok(compressed)
    }

    /// Compresses a child array produced by a cascading scheme.
    ///
    /// If the cascade budget is exhausted, the canonical array is returned as-is. Otherwise, the
    /// child context is created by descending and recording the parent scheme + child index, and
    /// compression proceeds normally.
    ///
    /// # Errors
    ///
    /// Returns an error if compression fails.
    pub fn compress_child(
        &self,
        child: &ArrayRef,
        parent_ctx: &CompressorContext,
        parent_id: SchemeId,
        child_index: usize,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        if parent_ctx.finished_cascading() {
            trace::cascade_exhausted(parent_id, child_index);
            return Ok(child.clone());
        }

        let canonical = child.clone().execute::<CanonicalValidity>(exec_ctx)?.0;
        let compact = canonical.compact()?;

        let child_ctx = parent_ctx
            .clone()
            .descend_with_scheme(parent_id, child_index);
        self.compress_canonical(compact, child_ctx, exec_ctx)
    }

    /// Compresses a canonical array by dispatching to type-specific logic.
    ///
    /// # Errors
    ///
    /// Returns an error if compression of any sub-array fails.
    fn compress_canonical(
        &self,
        array: Canonical,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        match array {
            Canonical::Null(null_array) => Ok(null_array.into_array()),
            Canonical::Bool(bool_array) => {
                self.choose_and_compress(Canonical::Bool(bool_array), compress_ctx, exec_ctx)
            }
            Canonical::Primitive(primitive) => {
                self.choose_and_compress(Canonical::Primitive(primitive), compress_ctx, exec_ctx)
            }
            Canonical::Decimal(decimal) => {
                self.choose_and_compress(Canonical::Decimal(decimal), compress_ctx, exec_ctx)
            }
            Canonical::Struct(struct_array) => {
                let fields = struct_array
                    .iter_unmasked_fields()
                    .map(|field| self.compress(field, exec_ctx))
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(StructArray::try_new(
                    struct_array.names().clone(),
                    fields,
                    struct_array.len(),
                    struct_array.validity()?,
                )?
                .into_array())
            }
            Canonical::List(list_view_array) => {
                if list_view_array.is_zero_copy_to_list() || list_view_array.elements().is_empty() {
                    let list_array = list_from_list_view(list_view_array)?;
                    self.compress_list_array(list_array, compress_ctx, exec_ctx)
                } else {
                    self.compress_list_view_array(list_view_array, compress_ctx, exec_ctx)
                }
            }
            Canonical::FixedSizeList(fsl_array) => {
                let compressed_elems = self.compress(fsl_array.elements(), exec_ctx)?;

                Ok(FixedSizeListArray::try_new(
                    compressed_elems,
                    fsl_array.list_size(),
                    fsl_array.validity()?,
                    fsl_array.len(),
                )?
                .into_array())
            }
            Canonical::VarBinView(strings) => {
                if strings
                    .dtype()
                    .eq_ignore_nullability(&DType::Utf8(Nullability::NonNullable))
                {
                    self.choose_and_compress(Canonical::VarBinView(strings), compress_ctx, exec_ctx)
                } else {
                    // We do not compress binary arrays.
                    Ok(strings.into_array())
                }
            }
            Canonical::Extension(ext_array) => {
                let before_nbytes = ext_array.as_ref().nbytes();

                // Try scheme-based compression first.
                let result = self.choose_and_compress(
                    Canonical::Extension(ext_array.clone()),
                    compress_ctx,
                    exec_ctx,
                )?;
                if result.nbytes() < before_nbytes {
                    return Ok(result);
                }

                // TODO(connor): HACK TO SUPPORT L2 DENORMALIZATION!!!
                if result.is::<AnyScalarFn>() {
                    return Ok(result);
                }

                // Otherwise, fall back to compressing the underlying storage array.
                let compressed_storage = self.compress(ext_array.storage_array(), exec_ctx)?;

                Ok(
                    ExtensionArray::new(ext_array.ext_dtype().clone(), compressed_storage)
                        .into_array(),
                )
            }
            Canonical::Variant(variant_array) => {
                let core_storage =
                    self.compress_physical_slots(variant_array.core_storage(), exec_ctx)?;
                let shredded = variant_array
                    .shredded()
                    .map(|arr| {
                        // Avoid stack-overflow for variant shredded values
                        if arr.is::<Variant>() {
                            self.compress_physical_slots(arr, exec_ctx)
                        } else {
                            self.compress(arr, exec_ctx)
                        }
                    })
                    .transpose()?;

                Ok(VariantArray::try_new(core_storage, shredded)?.into_array())
            }
        }
    }

    /// The main scheme-selection entry point for a single leaf array.
    ///
    /// Filters allowed schemes by [`matches`] and exclusion rules, merges their [`stats_options`]
    /// into a single [`GenerateStatsOptions`], and picks the winner by estimated compression
    /// ratio.
    ///
    /// If a winner is found and its compressed output is actually smaller, that output is
    /// returned. Otherwise, the original array is returned unchanged.
    ///
    /// Empty and all-null arrays are short-circuited before any scheme evaluation.
    ///
    /// [`matches`]: Scheme::matches
    /// [`stats_options`]: Scheme::stats_options
    fn choose_and_compress(
        &self,
        canonical: Canonical,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let eligible_schemes: Vec<&'static dyn Scheme> = self
            .schemes
            .iter()
            .copied()
            .filter(|s| s.matches(&canonical) && !self.is_excluded(*s, &compress_ctx))
            .collect();

        let array: ArrayRef = canonical.into();

        if eligible_schemes.is_empty() || array.is_empty() {
            return Ok(array);
        }

        if array.all_invalid(exec_ctx)? {
            return Ok(
                ConstantArray::new(Scalar::null(array.dtype().clone()), array.len()).into_array(),
            );
        }

        let before_nbytes = array.nbytes();

        let merged_opts = eligible_schemes
            .iter()
            .fold(GenerateStatsOptions::default(), |acc, s| {
                acc.merge(s.stats_options())
            });
        let compress_ctx = compress_ctx.with_merged_stats_options(merged_opts);

        let data = ArrayAndStats::new(array, merged_opts);

        let Some((winner, winner_estimate)) =
            self.choose_best_scheme(&eligible_schemes, &data, compress_ctx.clone(), exec_ctx)?
        else {
            return Ok(data.into_array());
        };

        // Run the winning scheme's `compress`. On failure, emit an ERROR event carrying the
        // scheme name and cascade history before propagating.
        let error_ctx = trace::enabled_error_context(&compress_ctx);
        let _winner_span = trace::winner_compress_span(winner.id(), before_nbytes).entered();
        let compressed = winner
            .compress(self, &data, compress_ctx, exec_ctx)
            .inspect_err(|err| {
                // NB: this is the only way we can tell which scheme panicked / bailed on their
                // data, especially for third-party schemes where the error site may not carry any
                // compressor context.
                trace::scheme_compress_failed(winner.id(), before_nbytes, error_ctx.as_ref(), err);
            })?;

        let after_nbytes = compressed.nbytes();
        let actual_ratio = (after_nbytes != 0).then(|| before_nbytes as f64 / after_nbytes as f64);

        // TODO(connor): HACK TO SUPPORT L2 DENORMALIZATION!!!
        let accepted = after_nbytes < before_nbytes || compressed.is::<AnyScalarFn>();

        trace::record_winner_compress_result(
            after_nbytes,
            winner_estimate.trace_ratio(),
            actual_ratio,
            accepted,
        );

        if accepted {
            Ok(compressed)
        } else {
            Ok(data.into_array())
        }
    }

    /// Calls [`expected_compression_ratio`] on each candidate and returns the winning scheme along
    /// with its resolved winner estimate, or `None` if no scheme beats the canonical encoding.
    ///
    /// Selection runs in two passes. Pass 1 evaluates every immediate
    /// [`CompressionEstimate::Verdict`] and tracks the running best. [`Scheme`]s returning
    /// [`CompressionEstimate::Deferred`] are stashed for pass 2 so that we do not make any
    /// expensive computations if we don't have to.
    ///
    /// Pass 2 evaluates the deferred work and, for each [`DeferredEstimate::Callback`], passes the
    /// current best [`EstimateScore`] as an early-exit hint so the callback can return
    /// [`EstimateVerdict::Skip`] without doing expensive work when it cannot beat the threshold.
    ///
    /// Ties are broken by registration order within each pass.
    ///
    /// [`expected_compression_ratio`]: Scheme::expected_compression_ratio
    fn choose_best_scheme(
        &self,
        schemes: &[&'static dyn Scheme],
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<(&'static dyn Scheme, WinnerEstimate)>> {
        let mut best: Option<(&'static dyn Scheme, EstimateScore)> = None;
        let mut deferred: Vec<(&'static dyn Scheme, DeferredEstimate)> = Vec::new();

        // Pass 1: evaluate every immediate verdict. Stash deferred work for pass 2.
        {
            let _verdict_pass = trace::verdict_pass_span().entered();
            for &scheme in schemes {
                match scheme.expected_compression_ratio(data, compress_ctx.clone(), exec_ctx) {
                    CompressionEstimate::Verdict(EstimateVerdict::Skip) => {}
                    CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse) => {
                        return Ok(Some((scheme, WinnerEstimate::AlwaysUse)));
                    }
                    CompressionEstimate::Verdict(EstimateVerdict::Ratio(ratio)) => {
                        let score = EstimateScore::FiniteCompression(ratio);

                        if is_better_score(score, best.as_ref()) {
                            best = Some((scheme, score));
                        }
                    }
                    CompressionEstimate::Deferred(deferred_estimate) => {
                        deferred.push((scheme, deferred_estimate));
                    }
                }
            }
        }

        // Pass 2: run deferred work. Callbacks receive the current best as a threshold so they can
        // short-circuit with `Skip` when they cannot beat it.
        for (scheme, deferred_estimate) in deferred {
            let _span = trace::scheme_eval_span(scheme.id()).entered();
            let threshold: Option<EstimateScore> = best.map(|(_, score)| score);
            match deferred_estimate {
                DeferredEstimate::Sample => {
                    let score = estimate_compression_ratio_with_sampling(
                        self,
                        scheme,
                        data.array(),
                        compress_ctx.clone(),
                        exec_ctx,
                    )?;

                    if is_better_score(score, best.as_ref()) {
                        best = Some((scheme, score));
                    }
                }
                DeferredEstimate::Callback(callback) => {
                    match callback(self, data, threshold, compress_ctx.clone(), exec_ctx)? {
                        EstimateVerdict::Skip => {}
                        EstimateVerdict::AlwaysUse => {
                            return Ok(Some((scheme, WinnerEstimate::AlwaysUse)));
                        }
                        EstimateVerdict::Ratio(ratio) => {
                            let score = EstimateScore::FiniteCompression(ratio);

                            if is_better_score(score, best.as_ref()) {
                                best = Some((scheme, score));
                            }
                        }
                    }
                }
            }
        }

        Ok(best.map(|(scheme, score)| (scheme, WinnerEstimate::Score(score))))
    }

    // TODO(connor): Lots of room for optimization here.
    /// Returns `true` if the candidate scheme should be excluded based on the cascade history and
    /// exclusion rules.
    fn is_excluded(&self, candidate: &dyn Scheme, ctx: &CompressorContext) -> bool {
        let id = candidate.id();
        let history = ctx.cascade_history();

        // Self-exclusion: no scheme appears twice in any chain.
        if history.iter().any(|&(sid, _)| sid == id) {
            return true;
        }

        let mut iter = history.iter().copied().peekable();

        // The root entry is always first in the history (if present). Check if the root has
        // excluded us.
        if let Some((_, child_idx)) = iter.next_if(|&(sid, _)| sid == ROOT_SCHEME_ID)
            && self
                .root_exclusions
                .iter()
                .any(|rule| rule.excluded == id && rule.children.contains(child_idx))
        {
            return true;
        }

        // Push rules: Check if any of our ancestors have excluded us.
        for (ancestor_id, child_idx) in iter {
            if let Some(ancestor) = self.schemes.iter().find(|s| s.id() == ancestor_id)
                && ancestor
                    .descendant_exclusions()
                    .iter()
                    .any(|rule| rule.excluded == id && rule.children.contains(child_idx))
            {
                return true;
            }
        }

        // Pull rules: Check if we have excluded ourselves because of our ancestors.
        for rule in candidate.ancestor_exclusions() {
            if history
                .iter()
                .any(|(sid, cidx)| *sid == rule.ancestor && rule.children.contains(*cidx))
            {
                return true;
            }
        }

        false
    }

    /// Compresses a [`ListArray`] by narrowing offsets and recursively compressing elements.
    fn compress_list_array(
        &self,
        list_array: ListArray,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let list_array = list_array.reset_offsets(true)?;

        let compressed_elems = self.compress(list_array.elements(), exec_ctx)?;

        // Record the root scheme with the offsets child index so root exclusion rules apply.
        let offset_ctx =
            compress_ctx.descend_with_scheme(ROOT_SCHEME_ID, root_list_children::OFFSETS);
        let list_offsets_primitive = list_array
            .offsets()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?;
        let compressed_offsets = self.compress_canonical(
            Canonical::Primitive(list_offsets_primitive),
            offset_ctx,
            exec_ctx,
        )?;

        Ok(
            ListArray::try_new(compressed_elems, compressed_offsets, list_array.validity()?)?
                .into_array(),
        )
    }

    /// Compresses a [`ListViewArray`] by narrowing offsets/sizes and recursively compressing
    /// elements.
    fn compress_list_view_array(
        &self,
        list_view: ListViewArray,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let compressed_elems = self.compress(list_view.elements(), exec_ctx)?;

        let offset_ctx = compress_ctx
            .clone()
            .descend_with_scheme(ROOT_SCHEME_ID, root_list_children::OFFSETS);
        let list_view_offsets_primitive = list_view
            .offsets()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?;
        let compressed_offsets = self.compress_canonical(
            Canonical::Primitive(list_view_offsets_primitive),
            offset_ctx,
            exec_ctx,
        )?;

        let sizes_ctx = compress_ctx.descend_with_scheme(ROOT_SCHEME_ID, root_list_children::SIZES);
        let list_view_sizes_primitive = list_view
            .sizes()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?;
        let compressed_sizes = self.compress_canonical(
            Canonical::Primitive(list_view_sizes_primitive),
            sizes_ctx,
            exec_ctx,
        )?;

        Ok(ListViewArray::try_new(
            compressed_elems,
            compressed_offsets,
            compressed_sizes,
            list_view.validity()?,
        )?
        .into_array())
    }

    /// Compress very child slot of the array, then re-build it from them.
    fn compress_physical_slots(
        &self,
        array: &ArrayRef,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let slots = array
            .slots()
            .iter()
            .map(|slot| {
                slot.as_ref()
                    .map(|child| self.compress(child, exec_ctx))
                    .transpose()
            })
            .collect::<VortexResult<ArraySlots>>()?;

        array.clone().with_slots(slots)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use parking_lot::Mutex;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::Constant;
    use vortex_array::arrays::NullArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use super::*;
    use crate::builtins::FloatDictScheme;
    use crate::builtins::IntDictScheme;
    use crate::builtins::StringDictScheme;
    use crate::ctx::CompressorContext;
    use crate::estimate::CompressionEstimate;
    use crate::estimate::DeferredEstimate;
    use crate::estimate::EstimateScore;
    use crate::estimate::EstimateVerdict;
    use crate::estimate::WinnerEstimate;
    use crate::scheme::SchemeExt;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    fn compressor() -> CascadingCompressor {
        CascadingCompressor::new(vec![&IntDictScheme, &FloatDictScheme, &StringDictScheme])
    }

    fn estimate_test_data() -> ArrayAndStats {
        let array = PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array();
        ArrayAndStats::new(array, GenerateStatsOptions::default())
    }

    fn matches_integer_primitive(canonical: &Canonical) -> bool {
        matches!(canonical, Canonical::Primitive(primitive) if primitive.ptype().is_int())
    }

    #[derive(Debug)]
    struct DirectRatioScheme;

    impl Scheme for DirectRatioScheme {
        fn scheme_name(&self) -> &'static str {
            "test.direct_ratio"
        }

        fn matches(&self, canonical: &Canonical) -> bool {
            matches_integer_primitive(canonical)
        }

        fn expected_compression_ratio(
            &self,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> CompressionEstimate {
            CompressionEstimate::Verdict(EstimateVerdict::Ratio(2.0))
        }

        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            unreachable!("test helper should never be selected for compression")
        }
    }

    #[derive(Debug)]
    struct ImmediateAlwaysUseScheme;

    impl Scheme for ImmediateAlwaysUseScheme {
        fn scheme_name(&self) -> &'static str {
            "test.immediate_always_use"
        }

        fn matches(&self, canonical: &Canonical) -> bool {
            matches_integer_primitive(canonical)
        }

        fn expected_compression_ratio(
            &self,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> CompressionEstimate {
            CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse)
        }

        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            unreachable!("test helper should never be selected for compression")
        }
    }

    #[derive(Debug)]
    struct CallbackAlwaysUseScheme;

    impl Scheme for CallbackAlwaysUseScheme {
        fn scheme_name(&self) -> &'static str {
            "test.callback_always_use"
        }

        fn matches(&self, canonical: &Canonical) -> bool {
            matches_integer_primitive(canonical)
        }

        fn expected_compression_ratio(
            &self,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> CompressionEstimate {
            CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
                |_compressor, _data, _ctx, _exec_ctx, _best_so_far| Ok(EstimateVerdict::AlwaysUse),
            )))
        }

        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            unreachable!("test helper should never be selected for compression")
        }
    }

    #[derive(Debug)]
    struct CallbackSkipScheme;

    impl Scheme for CallbackSkipScheme {
        fn scheme_name(&self) -> &'static str {
            "test.callback_skip"
        }

        fn matches(&self, canonical: &Canonical) -> bool {
            matches_integer_primitive(canonical)
        }

        fn expected_compression_ratio(
            &self,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> CompressionEstimate {
            CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
                |_compressor, _data, _ctx, _exec_ctx, _best_so_far| Ok(EstimateVerdict::Skip),
            )))
        }

        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            unreachable!("test helper should never be selected for compression")
        }
    }

    #[derive(Debug)]
    struct CallbackRatioScheme;

    impl Scheme for CallbackRatioScheme {
        fn scheme_name(&self) -> &'static str {
            "test.callback_ratio"
        }

        fn matches(&self, canonical: &Canonical) -> bool {
            matches_integer_primitive(canonical)
        }

        fn expected_compression_ratio(
            &self,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> CompressionEstimate {
            CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
                |_compressor, _data, _ctx, _exec_ctx, _best_so_far| Ok(EstimateVerdict::Ratio(3.0)),
            )))
        }

        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            unreachable!("test helper should never be selected for compression")
        }
    }

    #[derive(Debug)]
    struct HugeRatioScheme;

    impl Scheme for HugeRatioScheme {
        fn scheme_name(&self) -> &'static str {
            "test.huge_ratio"
        }

        fn matches(&self, canonical: &Canonical) -> bool {
            matches_integer_primitive(canonical)
        }

        fn expected_compression_ratio(
            &self,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> CompressionEstimate {
            CompressionEstimate::Verdict(EstimateVerdict::Ratio(100.0))
        }

        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            unreachable!("test helper should never be selected for compression")
        }
    }

    #[derive(Debug)]
    struct ZeroBytesSamplingScheme;

    impl Scheme for ZeroBytesSamplingScheme {
        fn scheme_name(&self) -> &'static str {
            "test.zero_bytes_sampling"
        }

        fn matches(&self, canonical: &Canonical) -> bool {
            matches_integer_primitive(canonical)
        }

        fn expected_compression_ratio(
            &self,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> CompressionEstimate {
            CompressionEstimate::Deferred(DeferredEstimate::Sample)
        }

        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            Ok(NullArray::new(data.array().len()).into_array())
        }
    }

    #[test]
    fn test_self_exclusion() {
        let c = compressor();
        let ctx = CompressorContext::default().descend_with_scheme(IntDictScheme.id(), 0);

        // IntDictScheme is in the history, so it should be excluded.
        assert!(c.is_excluded(&IntDictScheme, &ctx));
    }

    #[test]
    fn test_root_exclusion_list_offsets() {
        let c = compressor();
        let ctx = CompressorContext::default()
            .descend_with_scheme(ROOT_SCHEME_ID, root_list_children::OFFSETS);

        // IntDict should be excluded for list offsets.
        assert!(c.is_excluded(&IntDictScheme, &ctx));
    }

    #[test]
    fn test_push_rule_float_dict_excludes_int_dict_from_codes() {
        let c = compressor();
        // FloatDict cascading through codes (child 1).
        let ctx = CompressorContext::default().descend_with_scheme(FloatDictScheme.id(), 1);

        // IntDict should be excluded from FloatDict's codes child.
        assert!(c.is_excluded(&IntDictScheme, &ctx));
    }

    #[test]
    fn test_push_rule_float_dict_excludes_int_dict_from_values() {
        let c = compressor();
        // FloatDict cascading through values (child 0).
        let ctx = CompressorContext::default().descend_with_scheme(FloatDictScheme.id(), 0);

        // IntDict should also be excluded from FloatDict's values child (ALP propagation
        // replacement).
        assert!(c.is_excluded(&IntDictScheme, &ctx));
    }

    #[test]
    fn test_no_exclusion_without_history() {
        let c = compressor();
        let ctx = CompressorContext::default();

        // No history means no exclusions.
        assert!(!c.is_excluded(&IntDictScheme, &ctx));
    }

    #[test]
    fn immediate_always_use_wins_immediately() -> VortexResult<()> {
        let compressor =
            CascadingCompressor::new(vec![&DirectRatioScheme, &ImmediateAlwaysUseScheme]);
        let schemes: [&'static dyn Scheme; 2] = [&DirectRatioScheme, &ImmediateAlwaysUseScheme];
        let data = estimate_test_data();
        let mut exec_ctx = SESSION.create_execution_ctx();

        let winner = compressor.choose_best_scheme(
            &schemes,
            &data,
            CompressorContext::new(),
            &mut exec_ctx,
        )?;

        assert!(matches!(
            winner,
            Some((scheme, WinnerEstimate::AlwaysUse))
                if scheme.id() == ImmediateAlwaysUseScheme.id()
        ));
        Ok(())
    }

    #[test]
    fn callback_always_use_wins_immediately() -> VortexResult<()> {
        let compressor =
            CascadingCompressor::new(vec![&DirectRatioScheme, &CallbackAlwaysUseScheme]);
        let schemes: [&'static dyn Scheme; 2] = [&DirectRatioScheme, &CallbackAlwaysUseScheme];
        let data = estimate_test_data();
        let mut exec_ctx = SESSION.create_execution_ctx();

        let winner = compressor.choose_best_scheme(
            &schemes,
            &data,
            CompressorContext::new(),
            &mut exec_ctx,
        )?;

        assert!(matches!(
            winner,
            Some((scheme, WinnerEstimate::AlwaysUse))
                if scheme.id() == CallbackAlwaysUseScheme.id()
        ));
        Ok(())
    }

    #[test]
    fn callback_skip_is_ignored() -> VortexResult<()> {
        let compressor = CascadingCompressor::new(vec![&CallbackSkipScheme, &DirectRatioScheme]);
        let schemes: [&'static dyn Scheme; 2] = [&CallbackSkipScheme, &DirectRatioScheme];
        let data = estimate_test_data();
        let mut exec_ctx = SESSION.create_execution_ctx();

        let winner = compressor.choose_best_scheme(
            &schemes,
            &data,
            CompressorContext::new(),
            &mut exec_ctx,
        )?;

        assert!(matches!(
            winner,
            Some((scheme, WinnerEstimate::Score(EstimateScore::FiniteCompression(2.0))))
                if scheme.id() == DirectRatioScheme.id()
        ));
        Ok(())
    }

    #[test]
    fn callback_ratio_competes_numerically() -> VortexResult<()> {
        let compressor = CascadingCompressor::new(vec![&DirectRatioScheme, &CallbackRatioScheme]);
        let schemes: [&'static dyn Scheme; 2] = [&DirectRatioScheme, &CallbackRatioScheme];
        let data = estimate_test_data();
        let mut exec_ctx = SESSION.create_execution_ctx();

        let winner = compressor.choose_best_scheme(
            &schemes,
            &data,
            CompressorContext::new(),
            &mut exec_ctx,
        )?;

        assert!(matches!(
            winner,
            Some((scheme, WinnerEstimate::Score(EstimateScore::FiniteCompression(3.0))))
                if scheme.id() == CallbackRatioScheme.id()
        ));
        Ok(())
    }

    #[test]
    fn zero_byte_sample_loses_to_finite_ratio() -> VortexResult<()> {
        let compressor = CascadingCompressor::new(vec![&HugeRatioScheme, &ZeroBytesSamplingScheme]);
        let schemes: [&'static dyn Scheme; 2] = [&HugeRatioScheme, &ZeroBytesSamplingScheme];
        let data = estimate_test_data();
        let mut exec_ctx = SESSION.create_execution_ctx();

        let winner = compressor.choose_best_scheme(
            &schemes,
            &data,
            CompressorContext::new(),
            &mut exec_ctx,
        )?;

        assert!(matches!(
            winner,
            Some((scheme, WinnerEstimate::Score(EstimateScore::FiniteCompression(100.0))))
                if scheme.id() == HugeRatioScheme.id()
        ));
        Ok(())
    }

    #[test]
    fn finite_ratio_displaces_zero_byte_sample() -> VortexResult<()> {
        let compressor = CascadingCompressor::new(vec![&ZeroBytesSamplingScheme, &HugeRatioScheme]);
        let schemes: [&'static dyn Scheme; 2] = [&ZeroBytesSamplingScheme, &HugeRatioScheme];
        let data = estimate_test_data();
        let mut exec_ctx = SESSION.create_execution_ctx();

        let winner = compressor.choose_best_scheme(
            &schemes,
            &data,
            CompressorContext::new(),
            &mut exec_ctx,
        )?;

        assert!(matches!(
            winner,
            Some((scheme, WinnerEstimate::Score(EstimateScore::FiniteCompression(100.0))))
                if scheme.id() == HugeRatioScheme.id()
        ));
        Ok(())
    }

    #[test]
    fn zero_byte_sample_alone_selects_no_scheme() -> VortexResult<()> {
        let compressor = CascadingCompressor::new(vec![&ZeroBytesSamplingScheme]);
        let schemes: [&'static dyn Scheme; 1] = [&ZeroBytesSamplingScheme];
        let data = estimate_test_data();
        let mut exec_ctx = SESSION.create_execution_ctx();

        let winner = compressor.choose_best_scheme(
            &schemes,
            &data,
            CompressorContext::new(),
            &mut exec_ctx,
        )?;

        assert!(winner.is_none());
        Ok(())
    }

    // Observer helper used by threshold-related tests. Captures the `best_so_far` value the
    // compressor passes to its deferred callback. `OBSERVER_LOCK` serializes tests that share
    // `OBSERVED_THRESHOLD` so they do not race.
    static OBSERVER_LOCK: Mutex<()> = Mutex::new(());
    static OBSERVED_THRESHOLD: Mutex<Option<Option<EstimateScore>>> = Mutex::new(None);

    #[derive(Debug)]
    struct ThresholdObservingScheme;

    impl Scheme for ThresholdObservingScheme {
        fn scheme_name(&self) -> &'static str {
            "test.threshold_observing"
        }

        fn matches(&self, canonical: &Canonical) -> bool {
            matches_integer_primitive(canonical)
        }

        fn expected_compression_ratio(
            &self,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> CompressionEstimate {
            CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
                |_compressor, _data, best_so_far, _ctx, _exec_ctx| {
                    *OBSERVED_THRESHOLD.lock() = Some(best_so_far);
                    Ok(EstimateVerdict::Skip)
                },
            )))
        }

        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            unreachable!("test helper should never be selected for compression")
        }
    }

    #[derive(Debug)]
    struct CallbackMatchingRatioScheme;

    impl Scheme for CallbackMatchingRatioScheme {
        fn scheme_name(&self) -> &'static str {
            "test.callback_matching_ratio"
        }

        fn matches(&self, canonical: &Canonical) -> bool {
            matches_integer_primitive(canonical)
        }

        fn expected_compression_ratio(
            &self,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> CompressionEstimate {
            CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
                |_compressor, _data, _ctx, _exec_ctx, _best_so_far| Ok(EstimateVerdict::Ratio(2.0)),
            )))
        }

        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            unreachable!("test helper should never be selected for compression")
        }
    }

    #[test]
    fn callback_always_use_overrides_pass_one_best() -> VortexResult<()> {
        // `HugeRatioScheme` returns an immediate `Ratio(100.0)` in pass 1;
        // `CallbackAlwaysUseScheme` returns `AlwaysUse` from its deferred callback in pass 2.
        // The deferred `AlwaysUse` must still win.
        let compressor = CascadingCompressor::new(vec![&HugeRatioScheme, &CallbackAlwaysUseScheme]);
        let schemes: [&'static dyn Scheme; 2] = [&HugeRatioScheme, &CallbackAlwaysUseScheme];
        let data = estimate_test_data();
        let mut exec_ctx = SESSION.create_execution_ctx();

        let winner = compressor.choose_best_scheme(
            &schemes,
            &data,
            CompressorContext::new(),
            &mut exec_ctx,
        )?;

        assert!(matches!(
            winner,
            Some((scheme, WinnerEstimate::AlwaysUse))
                if scheme.id() == CallbackAlwaysUseScheme.id()
        ));
        Ok(())
    }

    #[test]
    fn threshold_reflects_pass_one_best() -> VortexResult<()> {
        let _guard = OBSERVER_LOCK.lock();
        *OBSERVED_THRESHOLD.lock() = None;

        let compressor =
            CascadingCompressor::new(vec![&DirectRatioScheme, &ThresholdObservingScheme]);
        let schemes: [&'static dyn Scheme; 2] = [&DirectRatioScheme, &ThresholdObservingScheme];
        let data = estimate_test_data();
        let mut exec_ctx = SESSION.create_execution_ctx();

        compressor.choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;

        let observed = *OBSERVED_THRESHOLD.lock();
        assert!(matches!(
            observed,
            Some(Some(EstimateScore::FiniteCompression(r))) if r == 2.0
        ));
        Ok(())
    }

    #[test]
    fn threshold_is_none_when_only_prior_is_zero_bytes() -> VortexResult<()> {
        let _guard = OBSERVER_LOCK.lock();
        *OBSERVED_THRESHOLD.lock() = None;

        let compressor =
            CascadingCompressor::new(vec![&ZeroBytesSamplingScheme, &ThresholdObservingScheme]);
        let schemes: [&'static dyn Scheme; 2] =
            [&ZeroBytesSamplingScheme, &ThresholdObservingScheme];
        let data = estimate_test_data();
        let mut exec_ctx = SESSION.create_execution_ctx();

        compressor.choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;

        // The observing callback was invoked (outer `Some`) and `best_so_far` was `None` (inner
        // `None`) because the zero-byte sample is never stored as the best.
        let observed = *OBSERVED_THRESHOLD.lock();
        assert_eq!(observed, Some(None));
        Ok(())
    }

    #[test]
    fn threshold_is_none_when_no_prior_scheme() -> VortexResult<()> {
        let _guard = OBSERVER_LOCK.lock();
        *OBSERVED_THRESHOLD.lock() = None;

        let compressor = CascadingCompressor::new(vec![&ThresholdObservingScheme]);
        let schemes: [&'static dyn Scheme; 1] = [&ThresholdObservingScheme];
        let data = estimate_test_data();
        let mut exec_ctx = SESSION.create_execution_ctx();

        compressor.choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;

        let observed = *OBSERVED_THRESHOLD.lock();
        assert_eq!(observed, Some(None));
        Ok(())
    }

    #[test]
    fn threshold_updates_from_earlier_deferred_callback() -> VortexResult<()> {
        let _guard = OBSERVER_LOCK.lock();
        *OBSERVED_THRESHOLD.lock() = None;

        // Both schemes are deferred. The first callback registers `Ratio(3.0)`; the second
        // callback must observe it as its threshold.
        let compressor =
            CascadingCompressor::new(vec![&CallbackRatioScheme, &ThresholdObservingScheme]);
        let schemes: [&'static dyn Scheme; 2] = [&CallbackRatioScheme, &ThresholdObservingScheme];
        let data = estimate_test_data();
        let mut exec_ctx = SESSION.create_execution_ctx();

        compressor.choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;

        let observed = *OBSERVED_THRESHOLD.lock();
        assert!(matches!(
            observed,
            Some(Some(EstimateScore::FiniteCompression(r))) if r == 3.0
        ));
        Ok(())
    }

    #[test]
    fn ratio_tie_between_immediate_and_deferred_favors_immediate() -> VortexResult<()> {
        // Both schemes produce the same `Ratio(2.0)`, one from pass 1 (immediate) and one from
        // pass 2 (deferred callback). Pass 1 locks in first, and strict `>` tie-breaking means
        // the deferred callback's equal ratio cannot displace it.
        let compressor =
            CascadingCompressor::new(vec![&CallbackMatchingRatioScheme, &DirectRatioScheme]);
        let schemes: [&'static dyn Scheme; 2] = [&CallbackMatchingRatioScheme, &DirectRatioScheme];
        let data = estimate_test_data();
        let mut exec_ctx = SESSION.create_execution_ctx();

        let winner = compressor.choose_best_scheme(
            &schemes,
            &data,
            CompressorContext::new(),
            &mut exec_ctx,
        )?;

        assert!(matches!(
            winner,
            Some((scheme, WinnerEstimate::Score(EstimateScore::FiniteCompression(r))))
                if scheme.id() == DirectRatioScheme.id() && r == 2.0
        ));
        Ok(())
    }

    #[test]
    fn all_null_array_compresses_to_constant() -> VortexResult<()> {
        let array = PrimitiveArray::new(
            buffer![0i32, 0, 0, 0, 0],
            Validity::Array(BoolArray::from_iter([false, false, false, false, false]).into_array()),
        )
        .into_array();

        // The compressor should produce a `ConstantArray` for an all-null array regardless of
        // which schemes are registered.
        let compressor = CascadingCompressor::new(vec![&IntDictScheme]);
        let mut exec_ctx = SESSION.create_execution_ctx();
        let compressed = compressor.compress(&array, &mut exec_ctx)?;
        assert!(compressed.is::<Constant>());
        Ok(())
    }

    /// Regression test for <https://github.com/vortex-data/vortex/issues/7227>.
    ///
    /// `estimate_compression_ratio_with_sampling` must use the *scheme's* stats options
    /// (which request distinct-value counting) rather than the context's stats options
    /// (which may not). With the old code this panicked inside `dictionary_encode` because
    /// distinct values were never computed for the sample.
    #[test]
    fn sampling_uses_scheme_stats_options() -> VortexResult<()> {
        // Low-cardinality float array so FloatDictScheme considers it compressible.
        let array = PrimitiveArray::new(
            buffer![1.0f32, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            Validity::NonNullable,
        )
        .into_array();

        let compressor = CascadingCompressor::new(vec![&FloatDictScheme]);

        // A context with default stats_options (count_distinct_values = false) and
        // marked as a sample so the function skips the sampling step and compresses
        // the array directly.
        let ctx = CompressorContext::new().with_sampling();

        // Before the fix this panicked with:
        //   "this must be present since `DictScheme` declared that we need distinct values"
        let mut exec_ctx = SESSION.create_execution_ctx();
        let score = estimate_compression_ratio_with_sampling(
            &compressor,
            &FloatDictScheme,
            &array,
            ctx,
            &mut exec_ctx,
        )?;
        assert!(matches!(score, EstimateScore::FiniteCompression(ratio) if ratio.is_finite()));
        Ok(())
    }
}
