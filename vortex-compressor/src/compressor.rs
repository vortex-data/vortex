// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cascading array compression implementation.

use std::sync::Arc;

use parking_lot::Mutex;
use parking_lot::MutexGuard;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::CanonicalValidity;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
#[expect(deprecated)]
use vortex_array::ToCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::list::ListArrayExt;
use vortex_array::arrays::listview::ListViewArrayExt;
use vortex_array::arrays::listview::list_from_list_view;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::builtins::IntDictScheme;
use crate::ctx::CompressorContext;
use crate::estimate::CompressionEstimate;
use crate::estimate::DeferredEstimate;
use crate::estimate::EstimateVerdict;
use crate::estimate::estimate_compression_ratio_with_sampling;
use crate::estimate::is_better_ratio;
use crate::scheme::ChildSelection;
use crate::scheme::DescendantExclusion;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::scheme::SchemeId;
use crate::stats::ArrayAndStats;
use crate::stats::GenerateStatsOptions;

/// The implicit root scheme ID for the compressor's own cascading (e.g. list offset compression).
///
/// This is the **only** [`SchemeId`] that is not auto-provided via [`SchemeExt`].
const ROOT_SCHEME_ID: SchemeId = SchemeId {
    name: "vortex.compressor.root",
};

/// Child indices for the compressor's list/listview compression.
mod root_list_children {
    /// List/ListView offsets child.
    pub const OFFSETS: usize = 1;
    /// ListView sizes child.
    pub const SIZES: usize = 2;
}

/// The winning estimate for a scheme after all deferred work has been resolved.
#[derive(Debug, Clone, Copy, PartialEq)]
enum WinnerEstimate {
    /// The scheme must be used immediately.
    AlwaysUse,
    /// The scheme won by numeric compression ratio.
    Ratio(f64),
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

    /// Shared execution context for array operations during compression.
    ///
    /// This should have low contention as we only execute arrays one at a time during compression.
    ctx: Arc<Mutex<ExecutionCtx>>,
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
            // TODO(connor): The caller should probably pass this in.
            ctx: Arc::new(Mutex::new(LEGACY_SESSION.create_execution_ctx())),
        }
    }

    /// Returns a mutable borrow of the execution context.
    pub fn execution_ctx(&self) -> MutexGuard<'_, ExecutionCtx> {
        self.ctx.lock()
    }

    /// Compresses an array using cascading adaptive compression.
    ///
    /// First canonicalizes and compacts the array, then applies optimal compression schemes.
    ///
    /// # Errors
    ///
    /// Returns an error if canonicalization or compression fails.
    pub fn compress(&self, array: &ArrayRef) -> VortexResult<ArrayRef> {
        let canonical = array
            .clone()
            .execute::<CanonicalValidity>(&mut self.execution_ctx())?
            .0;

        // Compact it, removing any wasted space before we attempt to compress it.
        let compact = canonical.compact()?;

        self.compress_canonical(compact, CompressorContext::new())
    }

    /// Compresses a child array produced by a cascading scheme.
    ///
    /// If the cascade budget is exhausted, the canonical array is returned as-is. Otherwise,
    /// the child context is created by descending and recording the parent scheme + child
    /// index, and compression proceeds normally.
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
    ) -> VortexResult<ArrayRef> {
        if parent_ctx.finished_cascading() {
            return Ok(child.clone());
        }

        let canonical = child
            .clone()
            .execute::<CanonicalValidity>(&mut self.execution_ctx())?
            .0;
        let compact = canonical.compact()?;

        let child_ctx = parent_ctx
            .clone()
            .descend_with_scheme(parent_id, child_index);
        self.compress_canonical(compact, child_ctx)
    }

    /// Compresses a canonical array by dispatching to type-specific logic.
    ///
    /// # Errors
    ///
    /// Returns an error if compression of any sub-array fails.
    fn compress_canonical(
        &self,
        array: Canonical,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        match array {
            Canonical::Null(null_array) => Ok(null_array.into_array()),
            Canonical::Bool(bool_array) => {
                self.choose_and_compress(Canonical::Bool(bool_array), ctx)
            }
            Canonical::Primitive(primitive) => {
                self.choose_and_compress(Canonical::Primitive(primitive), ctx)
            }
            Canonical::Decimal(decimal) => {
                self.choose_and_compress(Canonical::Decimal(decimal), ctx)
            }
            Canonical::Struct(struct_array) => {
                let fields = struct_array
                    .iter_unmasked_fields()
                    .map(|field| self.compress(field))
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
                    self.compress_list_array(list_array, ctx)
                } else {
                    self.compress_list_view_array(list_view_array, ctx)
                }
            }
            Canonical::FixedSizeList(fsl_array) => {
                let compressed_elems = self.compress(fsl_array.elements())?;

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
                    self.choose_and_compress(Canonical::VarBinView(strings), ctx)
                } else {
                    // We do not compress binary arrays.
                    Ok(strings.into_array())
                }
            }
            Canonical::Extension(ext_array) => {
                let before_nbytes = ext_array.as_ref().nbytes();

                // Try scheme-based compression first.
                let result =
                    self.choose_and_compress(Canonical::Extension(ext_array.clone()), ctx)?;
                if result.nbytes() < before_nbytes {
                    return Ok(result);
                }

                // Otherwise, fall back to compressing the underlying storage array.
                let compressed_storage = self.compress(ext_array.storage_array())?;

                Ok(
                    ExtensionArray::new(ext_array.ext_dtype().clone(), compressed_storage)
                        .into_array(),
                )
            }
            Canonical::Variant(_) => {
                vortex_bail!("Variant arrays can not be compressed")
            }
        }
    }

    /// The main scheme-selection entry point for a single leaf array.
    ///
    /// Filters allowed schemes by [`matches`] and exclusion rules, merges their [`stats_options`]
    /// into a single [`GenerateStatsOptions`], then delegates to [`choose_scheme`] to pick the
    /// winner by estimated compression ratio.
    ///
    /// If a winner is found and its compressed output is actually smaller, that output is returned.
    /// Otherwise, the original array is returned unchanged.
    ///
    /// Empty and all-null arrays are short-circuited before any scheme evaluation.
    ///
    /// [`matches`]: Scheme::matches
    /// [`stats_options`]: Scheme::stats_options
    /// [`choose_scheme`]: Self::choose_scheme
    fn choose_and_compress(
        &self,
        canonical: Canonical,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let eligible_schemes: Vec<&'static dyn Scheme> = self
            .schemes
            .iter()
            .copied()
            .filter(|s| s.matches(&canonical) && !self.is_excluded(*s, &ctx))
            .collect();

        let array: ArrayRef = canonical.into();

        // If there are no schemes that we can compress into, then just return it uncompressed.
        if eligible_schemes.is_empty() {
            return Ok(array);
        }

        // Nothing to compress if empty or all-null.
        if array.is_empty() {
            return Ok(array);
        }
        if array.all_invalid(&mut LEGACY_SESSION.create_execution_ctx())? {
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
        let ctx = ctx.with_merged_stats_options(merged_opts);

        let mut data = ArrayAndStats::new(array, merged_opts);

        // TODO(connor): Add tracing support for logging the winner estimate.
        if let Some((winner, _winner_estimate)) =
            self.choose_best_scheme(&eligible_schemes, &mut data, ctx.clone())?
        {
            // Sampling and estimation chose a scheme, so let's compress the whole array with it.
            let compressed = winner.compress(self, &mut data, ctx)?;

            // TODO(connor): Add a tracing warning here if compression with the chosen scheme
            // failed, since there was likely more we could have done while choosing schemes.

            // Only choose the compressed array if it is smaller than the canonical one.
            if compressed.nbytes() < before_nbytes {
                // TODO(connor): Add a tracing warning here too.
                return Ok(compressed);
            }
        }

        // No scheme improved on the original.
        Ok(data.into_array())
    }

    /// Calls [`expected_compression_ratio`] on each candidate and returns the winning scheme and
    /// resolved winner estimate, or `None` if no scheme exceeds 1.0. Ties are broken by
    /// registration order (earlier in the list wins).
    ///
    /// [`expected_compression_ratio`]: Scheme::expected_compression_ratio
    fn choose_best_scheme(
        &self,
        schemes: &[&'static dyn Scheme],
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<Option<(&'static dyn Scheme, WinnerEstimate)>> {
        let mut best: Option<(&'static dyn Scheme, f64)> = None;

        // TODO(connor): Might want to use an `im` data structure inside of `ctx` if the clones here
        // are expensive.
        for &scheme in schemes {
            let estimate = scheme.expected_compression_ratio(data, ctx.clone());

            // TODO(connor): Rather than computing the deferred estimates eagerly, it would be
            // better to look at all quick estimates and see if it makes sense to sample at all.
            match estimate {
                CompressionEstimate::Verdict(verdict) => {
                    if let Some(winner_estimate) =
                        Self::check_and_update_estimate_verdict(&mut best, scheme, verdict)
                    {
                        return Ok(Some((scheme, winner_estimate)));
                    }
                }
                CompressionEstimate::Deferred(DeferredEstimate::Sample) => {
                    let sample_ratio = estimate_compression_ratio_with_sampling(
                        scheme,
                        self,
                        data.array(),
                        ctx.clone(),
                    )?;

                    if is_better_ratio(sample_ratio, &best) {
                        best = Some((scheme, sample_ratio));
                    }
                }
                CompressionEstimate::Deferred(DeferredEstimate::Callback(estimate_callback)) => {
                    let verdict = estimate_callback(self, data, ctx.clone())?;
                    if let Some(winner_estimate) =
                        Self::check_and_update_estimate_verdict(&mut best, scheme, verdict)
                    {
                        return Ok(Some((scheme, winner_estimate)));
                    }
                }
            }
        }

        Ok(best.map(|(scheme, ratio)| (scheme, WinnerEstimate::Ratio(ratio))))
    }

    /// Updates `best` from a terminal estimate verdict.
    fn check_and_update_estimate_verdict(
        best: &mut Option<(&'static dyn Scheme, f64)>,
        scheme: &'static dyn Scheme,
        verdict: EstimateVerdict,
    ) -> Option<WinnerEstimate> {
        match verdict {
            EstimateVerdict::Skip => None,
            EstimateVerdict::AlwaysUse => Some(WinnerEstimate::AlwaysUse),
            EstimateVerdict::Ratio(ratio) => {
                if is_better_ratio(ratio, &*best) {
                    *best = Some((scheme, ratio));
                }
                None
            }
        }
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
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let list_array = list_array.reset_offsets(true)?;

        let compressed_elems = self.compress(list_array.elements())?;

        // Record the root scheme with the offsets child index so root exclusion rules apply.
        let offset_ctx = ctx.descend_with_scheme(ROOT_SCHEME_ID, root_list_children::OFFSETS);
        #[expect(deprecated)]
        let list_offsets_primitive = list_array.offsets().to_primitive().narrow()?;
        let compressed_offsets =
            self.compress_canonical(Canonical::Primitive(list_offsets_primitive), offset_ctx)?;

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
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let compressed_elems = self.compress(list_view.elements())?;

        let offset_ctx = ctx
            .clone()
            .descend_with_scheme(ROOT_SCHEME_ID, root_list_children::OFFSETS);
        #[expect(deprecated)]
        let list_view_offsets_primitive = list_view.offsets().to_primitive().narrow()?;
        let compressed_offsets = self.compress_canonical(
            Canonical::Primitive(list_view_offsets_primitive),
            offset_ctx,
        )?;

        let sizes_ctx = ctx.descend_with_scheme(ROOT_SCHEME_ID, root_list_children::SIZES);
        #[expect(deprecated)]
        let list_view_sizes_primitive = list_view.sizes().to_primitive().narrow()?;
        let compressed_sizes =
            self.compress_canonical(Canonical::Primitive(list_view_sizes_primitive), sizes_ctx)?;

        Ok(ListViewArray::try_new(
            compressed_elems,
            compressed_offsets,
            compressed_sizes,
            list_view.validity()?,
        )?
        .into_array())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::Constant;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use super::*;
    use crate::builtins::FloatDictScheme;
    use crate::builtins::IntDictScheme;
    use crate::builtins::StringDictScheme;
    use crate::ctx::CompressorContext;
    use crate::estimate::CompressionEstimate;
    use crate::estimate::DeferredEstimate;
    use crate::estimate::EstimateVerdict;
    use crate::scheme::SchemeExt;

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
            _data: &mut ArrayAndStats,
            _ctx: CompressorContext,
        ) -> CompressionEstimate {
            CompressionEstimate::Verdict(EstimateVerdict::Ratio(2.0))
        }

        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            _data: &mut ArrayAndStats,
            _ctx: CompressorContext,
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
            _data: &mut ArrayAndStats,
            _ctx: CompressorContext,
        ) -> CompressionEstimate {
            CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse)
        }

        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            _data: &mut ArrayAndStats,
            _ctx: CompressorContext,
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
            _data: &mut ArrayAndStats,
            _ctx: CompressorContext,
        ) -> CompressionEstimate {
            CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
                |_compressor, _data, _ctx| Ok(EstimateVerdict::AlwaysUse),
            )))
        }

        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            _data: &mut ArrayAndStats,
            _ctx: CompressorContext,
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
            _data: &mut ArrayAndStats,
            _ctx: CompressorContext,
        ) -> CompressionEstimate {
            CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
                |_compressor, _data, _ctx| Ok(EstimateVerdict::Skip),
            )))
        }

        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            _data: &mut ArrayAndStats,
            _ctx: CompressorContext,
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
            _data: &mut ArrayAndStats,
            _ctx: CompressorContext,
        ) -> CompressionEstimate {
            CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
                |_compressor, _data, _ctx| Ok(EstimateVerdict::Ratio(3.0)),
            )))
        }

        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            _data: &mut ArrayAndStats,
            _ctx: CompressorContext,
        ) -> VortexResult<ArrayRef> {
            unreachable!("test helper should never be selected for compression")
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
        let mut data = estimate_test_data();

        let winner =
            compressor.choose_best_scheme(&schemes, &mut data, CompressorContext::new())?;

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
        let mut data = estimate_test_data();

        let winner =
            compressor.choose_best_scheme(&schemes, &mut data, CompressorContext::new())?;

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
        let mut data = estimate_test_data();

        let winner =
            compressor.choose_best_scheme(&schemes, &mut data, CompressorContext::new())?;

        assert!(matches!(
            winner,
            Some((scheme, WinnerEstimate::Ratio(2.0)))
                if scheme.id() == DirectRatioScheme.id()
        ));
        Ok(())
    }

    #[test]
    fn callback_ratio_competes_numerically() -> VortexResult<()> {
        let compressor = CascadingCompressor::new(vec![&DirectRatioScheme, &CallbackRatioScheme]);
        let schemes: [&'static dyn Scheme; 2] = [&DirectRatioScheme, &CallbackRatioScheme];
        let mut data = estimate_test_data();

        let winner =
            compressor.choose_best_scheme(&schemes, &mut data, CompressorContext::new())?;

        assert!(matches!(
            winner,
            Some((scheme, WinnerEstimate::Ratio(3.0)))
                if scheme.id() == CallbackRatioScheme.id()
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
        let compressed = compressor.compress(&array)?;
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
        let ratio =
            estimate_compression_ratio_with_sampling(&FloatDictScheme, &compressor, &array, ctx)?;
        assert!(ratio.is_finite());
        Ok(())
    }
}
