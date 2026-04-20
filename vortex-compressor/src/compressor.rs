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
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::list::ListArrayExt;
use vortex_array::arrays::listview::ListViewArrayExt;
use vortex_array::arrays::listview::list_from_list_view;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::scalar_fn::AnyScalarFn;
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
        let before_nbytes = array.nbytes();
        let span = trace::compress_span(array.len(), array.dtype(), before_nbytes);
        let _enter = span.enter();

        let canonical = array
            .clone()
            .execute::<CanonicalValidity>(&mut self.execution_ctx())?
            .0;
        let compact = canonical.compact()?;
        let compressed = self.compress_canonical(compact, CompressorContext::new())?;

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
    ) -> VortexResult<ArrayRef> {
        if parent_ctx.finished_cascading() {
            trace::cascade_exhausted(parent_id, child_index);
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

                // TODO(connor): HACK TO SUPPORT L2 DENORMALIZATION!!!
                if result.is::<AnyScalarFn>() {
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
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let eligible_schemes: Vec<&'static dyn Scheme> = self
            .schemes
            .iter()
            .copied()
            .filter(|s| s.matches(&canonical) && !self.is_excluded(*s, &ctx))
            .collect();

        let array: ArrayRef = canonical.into();

        if eligible_schemes.is_empty() || array.is_empty() {
            return Ok(array);
        }

        if array.all_invalid(&mut self.execution_ctx())? {
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

        let Some((winner, winner_estimate)) =
            self.choose_best_scheme(&eligible_schemes, &mut data, ctx.clone())?
        else {
            return Ok(data.into_array());
        };

        // Run the winning scheme's `compress`. On failure, emit an ERROR event carrying the
        // scheme name and cascade history before propagating.
        let error_ctx = trace::enabled_error_context(&ctx);
        let compressed = match winner.compress(self, &mut data, ctx) {
            Ok(compressed) => compressed,
            Err(err) => {
                // NB: this is the only way we can tell which scheme panicked / bailed on their
                // data, especially for third-party schemes where the error site may not carry any
                // compressor context.
                trace::scheme_compress_failed(winner.id(), before_nbytes, error_ctx.as_ref(), &err);
                return Err(err);
            }
        };

        let after_nbytes = compressed.nbytes();
        let actual_ratio = (after_nbytes != 0).then(|| before_nbytes as f64 / after_nbytes as f64);

        // TODO(connor): HACK TO SUPPORT L2 DENORMALIZATION!!!
        let accepted = after_nbytes < before_nbytes || compressed.is::<AnyScalarFn>();

        trace::scheme_compress_result(
            winner.id(),
            before_nbytes,
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
    /// Ties are broken by registration order (earlier in the list wins).
    ///
    /// [`expected_compression_ratio`]: Scheme::expected_compression_ratio
    fn choose_best_scheme(
        &self,
        schemes: &[&'static dyn Scheme],
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<Option<(&'static dyn Scheme, WinnerEstimate)>> {
        let mut best: Option<(&'static dyn Scheme, EstimateScore)> = None;

        // TODO(connor): Rather than computing the deferred estimates eagerly, it would be better to
        // look at all quick estimates and see if it makes sense to sample at all.
        for &scheme in schemes {
            let verdict = match scheme.expected_compression_ratio(data, ctx.clone()) {
                CompressionEstimate::Verdict(verdict) => verdict,
                CompressionEstimate::Deferred(DeferredEstimate::Sample) => {
                    let score = estimate_compression_ratio_with_sampling(
                        scheme,
                        self,
                        data.array(),
                        ctx.clone(),
                    )?;
                    if is_better_score(score, &best) {
                        best = Some((scheme, score));
                    }
                    continue;
                }
                CompressionEstimate::Deferred(DeferredEstimate::Callback(callback)) => {
                    callback(self, data, ctx.clone())?
                }
            };

            match verdict {
                EstimateVerdict::Skip => {}
                EstimateVerdict::AlwaysUse => return Ok(Some((scheme, WinnerEstimate::AlwaysUse))),
                EstimateVerdict::Ratio(ratio) => {
                    let score = EstimateScore::FiniteCompression(ratio);
                    if is_better_score(score, &best) {
                        best = Some((scheme, score));
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
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let list_array = list_array.reset_offsets(true)?;

        let compressed_elems = self.compress(list_array.elements())?;

        // Record the root scheme with the offsets child index so root exclusion rules apply.
        let offset_ctx = ctx.descend_with_scheme(ROOT_SCHEME_ID, root_list_children::OFFSETS);
        let list_offsets_primitive = list_array
            .offsets()
            .clone()
            .execute::<PrimitiveArray>(&mut self.execution_ctx())?
            .narrow()?;
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
        let list_view_offsets_primitive = list_view
            .offsets()
            .clone()
            .execute::<PrimitiveArray>(&mut self.execution_ctx())?
            .narrow()?;
        let compressed_offsets = self.compress_canonical(
            Canonical::Primitive(list_view_offsets_primitive),
            offset_ctx,
        )?;

        let sizes_ctx = ctx.descend_with_scheme(ROOT_SCHEME_ID, root_list_children::SIZES);
        let list_view_sizes_primitive = list_view
            .sizes()
            .clone()
            .execute::<PrimitiveArray>(&mut self.execution_ctx())?
            .narrow()?;
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
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use tracing::Event;
    use tracing::Subscriber;
    use tracing::field::Field;
    use tracing::field::Visit;
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::Context;
    use tracing_subscriber::prelude::*;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::Constant;
    use vortex_array::arrays::NullArray;
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
    use crate::estimate::EstimateScore;
    use crate::estimate::EstimateVerdict;
    use crate::estimate::WinnerEstimate;
    use crate::scheme::SchemeExt;
    use crate::trace::TARGET_TRACE;

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

    fn test_integer_array() -> ArrayRef {
        PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array()
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct RecordedEvent {
        target: String,
        fields: BTreeMap<String, String>,
    }

    #[derive(Default)]
    struct EventVisitor {
        fields: BTreeMap<String, String>,
    }

    impl Visit for EventVisitor {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            self.fields
                .insert(field.name().to_owned(), format!("{value:?}"));
        }

        fn record_i64(&mut self, field: &Field, value: i64) {
            self.fields
                .insert(field.name().to_owned(), value.to_string());
        }

        fn record_u64(&mut self, field: &Field, value: u64) {
            self.fields
                .insert(field.name().to_owned(), value.to_string());
        }

        fn record_bool(&mut self, field: &Field, value: bool) {
            self.fields
                .insert(field.name().to_owned(), value.to_string());
        }

        fn record_str(&mut self, field: &Field, value: &str) {
            self.fields
                .insert(field.name().to_owned(), value.to_owned());
        }
    }

    struct RecordingLayer {
        events: Arc<Mutex<Vec<RecordedEvent>>>,
    }

    impl RecordingLayer {
        fn new(events: Arc<Mutex<Vec<RecordedEvent>>>) -> Self {
            Self { events }
        }
    }

    impl<S> Layer<S> for RecordingLayer
    where
        S: Subscriber,
    {
        fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
            let mut visitor = EventVisitor::default();
            event.record(&mut visitor);
            self.events.lock().push(RecordedEvent {
                target: event.metadata().target().to_owned(),
                fields: visitor.fields,
            });
        }
    }

    fn record_events<T>(f: impl FnOnce() -> T) -> (T, Vec<RecordedEvent>) {
        let events = Arc::new(Mutex::new(Vec::new()));
        let subscriber =
            tracing_subscriber::registry().with(RecordingLayer::new(Arc::clone(&events)));
        let result = tracing::subscriber::with_default(subscriber, f);
        let recorded = events.lock().clone();
        (result, recorded)
    }

    fn find_event<'a>(
        events: &'a [RecordedEvent],
        target: &str,
        message: &str,
    ) -> &'a RecordedEvent {
        events
            .iter()
            .find(|event| {
                event.target == target
                    && event
                        .fields
                        .get("message")
                        .is_some_and(|value| value == message)
            })
            .expect("expected event not found")
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
            _data: &mut ArrayAndStats,
            _ctx: CompressorContext,
        ) -> CompressionEstimate {
            CompressionEstimate::Verdict(EstimateVerdict::Ratio(100.0))
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
            _data: &mut ArrayAndStats,
            _ctx: CompressorContext,
        ) -> CompressionEstimate {
            CompressionEstimate::Deferred(DeferredEstimate::Sample)
        }

        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            data: &mut ArrayAndStats,
            _ctx: CompressorContext,
        ) -> VortexResult<ArrayRef> {
            Ok(NullArray::new(data.array().len()).into_array())
        }
    }

    #[derive(Debug)]
    struct NestedFailureParentScheme;

    impl Scheme for NestedFailureParentScheme {
        fn scheme_name(&self) -> &'static str {
            "test.nested_failure_parent"
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
            compressor: &CascadingCompressor,
            data: &mut ArrayAndStats,
            ctx: CompressorContext,
        ) -> VortexResult<ArrayRef> {
            compressor.compress_child(data.array(), &ctx, self.id(), 1)
        }
    }

    #[derive(Debug)]
    struct NestedFailureLeafScheme;

    impl Scheme for NestedFailureLeafScheme {
        fn scheme_name(&self) -> &'static str {
            "test.nested_failure_leaf"
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
            vortex_error::vortex_bail!("nested failure")
        }
    }

    #[derive(Debug)]
    struct SamplingFailureScheme;

    impl Scheme for SamplingFailureScheme {
        fn scheme_name(&self) -> &'static str {
            "test.sampling_failure"
        }

        fn matches(&self, canonical: &Canonical) -> bool {
            matches_integer_primitive(canonical)
        }

        fn expected_compression_ratio(
            &self,
            _data: &mut ArrayAndStats,
            _ctx: CompressorContext,
        ) -> CompressionEstimate {
            CompressionEstimate::Deferred(DeferredEstimate::Sample)
        }

        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            _data: &mut ArrayAndStats,
            _ctx: CompressorContext,
        ) -> VortexResult<ArrayRef> {
            vortex_error::vortex_bail!("sample failure")
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
            Some((scheme, WinnerEstimate::Score(EstimateScore::FiniteCompression(2.0))))
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
            Some((scheme, WinnerEstimate::Score(EstimateScore::FiniteCompression(3.0))))
                if scheme.id() == CallbackRatioScheme.id()
        ));
        Ok(())
    }

    #[test]
    fn zero_byte_sample_loses_to_finite_ratio() -> VortexResult<()> {
        let compressor = CascadingCompressor::new(vec![&HugeRatioScheme, &ZeroBytesSamplingScheme]);
        let schemes: [&'static dyn Scheme; 2] = [&HugeRatioScheme, &ZeroBytesSamplingScheme];
        let mut data = estimate_test_data();

        let winner =
            compressor.choose_best_scheme(&schemes, &mut data, CompressorContext::new())?;

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
        let mut data = estimate_test_data();

        let winner =
            compressor.choose_best_scheme(&schemes, &mut data, CompressorContext::new())?;

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
        let mut data = estimate_test_data();

        let winner =
            compressor.choose_best_scheme(&schemes, &mut data, CompressorContext::new())?;

        assert!(winner.is_none());
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
        let score =
            estimate_compression_ratio_with_sampling(&FloatDictScheme, &compressor, &array, ctx)?;
        assert!(matches!(score, EstimateScore::FiniteCompression(ratio) if ratio.is_finite()));
        Ok(())
    }

    #[test]
    fn compress_failure_event_includes_cascade_path_and_depth() {
        let compressor =
            CascadingCompressor::new(vec![&NestedFailureParentScheme, &NestedFailureLeafScheme]);
        let array = test_integer_array();

        let (result, events) = record_events(|| compressor.compress(&array));

        assert!(result.is_err());
        let event = find_event(&events, TARGET_TRACE, "scheme.compress_failed");
        assert_eq!(
            event.fields.get("scheme").map(String::as_str),
            Some("test.nested_failure_leaf")
        );
        assert_eq!(
            event.fields.get("cascade_path").map(String::as_str),
            Some("test.nested_failure_parent[1]")
        );
        assert_eq!(
            event.fields.get("cascade_depth").map(String::as_str),
            Some("1")
        );
    }

    #[test]
    fn sample_failure_event_includes_cascade_path_and_depth() {
        let compressor = CascadingCompressor::new(vec![&SamplingFailureScheme]);
        let array = test_integer_array();

        let (result, events) = record_events(|| compressor.compress(&array));

        assert!(result.is_err());
        let event = find_event(&events, TARGET_TRACE, "sample.compress_failed");
        assert_eq!(
            event.fields.get("scheme").map(String::as_str),
            Some("test.sampling_failure")
        );
        assert_eq!(
            event.fields.get("cascade_path").map(String::as_str),
            Some("root")
        );
        assert_eq!(
            event.fields.get("cascade_depth").map(String::as_str),
            Some("0")
        );
    }

    #[test]
    fn zero_byte_sample_result_omits_ratio_fields_and_selects_no_scheme() {
        let compressor = CascadingCompressor::new(vec![&ZeroBytesSamplingScheme]);
        let array = test_integer_array();

        let (result, events) = record_events(|| compressor.compress(&array));

        assert!(result.is_ok());

        let sample_event = find_event(&events, TARGET_TRACE, "sample.result");
        assert_eq!(
            sample_event.fields.get("sampled_after").map(String::as_str),
            Some("0")
        );
        assert!(!sample_event.fields.contains_key("sampled_ratio"));

        assert!(!events.iter().any(|event| {
            event.target == TARGET_TRACE
                && event
                    .fields
                    .get("message")
                    .is_some_and(|value| value == "scheme.compress_result")
        }));
    }
}
