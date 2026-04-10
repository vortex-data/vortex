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
use vortex_error::vortex_panic;

use crate::builtins::IntDictScheme;
use crate::ctx::CompressorContext;
use crate::estimate::CompressionEstimate;
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

/// Tracing target for scheme selection events (eligibility, evaluation, winner, short-circuits).
///
/// See the crate-level `Observability` section of [`crate`] for the full target taxonomy.
const TARGET_SELECT: &str = "vortex_compressor::select";

/// Tracing target for per-scheme encoding events (the `scheme.compress` span and the
/// `scheme.compress_result` event reporting estimated vs actual compression ratios).
const TARGET_ENCODE: &str = "vortex_compressor::encode";

/// Tracing target for cascade-tree events (top-level `compress` and `compress_child` spans,
/// cascade-exhausted short-circuits).
const TARGET_CASCADE: &str = "vortex_compressor::cascade";

/// Emits a structured `scheme.evaluated` trace event on [`TARGET_SELECT`] for one scheme's
/// initial estimation verdict.
///
/// For `Ratio(r)` the numeric estimate is recorded directly. For `Sample` and `Estimate`
/// the ratio is not yet known at this point; a follow-up `scheme.evaluated.resolved` event
/// is emitted by the caller after the deferred computation finishes.
///
/// Defined as a standalone helper (rather than inlined) because the `match` expression that
/// extracts `kind` and the optional `ratio` field is the only repetition worth factoring out
/// of [`CascadingCompressor::choose_best_scheme`].
fn emit_scheme_evaluated(scheme: &'static dyn Scheme, estimate: &CompressionEstimate) {
    let (kind, ratio): (&'static str, Option<f64>) = match estimate {
        CompressionEstimate::Skip => ("Skip", None),
        CompressionEstimate::AlwaysUse => ("AlwaysUse", None),
        CompressionEstimate::Ratio(r) => ("Ratio", Some(*r)),
        CompressionEstimate::Sample => ("Sample", None),
        CompressionEstimate::Estimate(_) => ("Estimate", None),
    };
    tracing::trace!(
        target: TARGET_SELECT,
        scheme = %scheme.id(),
        kind,
        ratio = ?ratio,
        "scheme.evaluated",
    );
}

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
/// 3. Evaluating each matching scheme's compression ratio on a sample.
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
    #[tracing::instrument(
        target = "vortex_compressor::cascade",
        name = "CascadingCompressor::compress",
        level = "trace",
        skip_all,
        fields(
            len = array.len(),
            nbytes = array.nbytes(),
            dtype = %array.dtype(),
        ),
    )]
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
        let _span = tracing::trace_span!(
            target: TARGET_CASCADE,
            "compress_child",
            parent = %parent_id,
            child_index,
            cascade_depth = parent_ctx.cascade_history().len(),
            len = child.len(),
        )
        .entered();

        if parent_ctx.finished_cascading() {
            tracing::debug!(
                target: TARGET_CASCADE,
                reason = "cascade_exhausted",
                parent = %parent_id,
                child_index,
                "short_circuit",
            );
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
        // Capture span-facing metadata before we move `canonical` into an `ArrayRef`.
        let len = canonical.len();
        let cascade_depth = ctx.cascade_history().len();

        // `eligible_count` is recorded after filtering; pre-declare it so `Span::record`
        // works.
        let _span = tracing::trace_span!(
            target: TARGET_SELECT,
            "choose_and_compress",
            dtype = %canonical.dtype(),
            len,
            cascade_depth,
            eligible_count = tracing::field::Empty,
        )
        .entered();

        let eligible_schemes: Vec<&'static dyn Scheme> = self
            .schemes
            .iter()
            .copied()
            .filter(|s| s.matches(&canonical) && !self.is_excluded(*s, &ctx))
            .collect();

        tracing::Span::current().record("eligible_count", eligible_schemes.len());

        let array: ArrayRef = canonical.into();

        // If there are no schemes that we can compress into, then just return it uncompressed.
        if eligible_schemes.is_empty() {
            tracing::debug!(
                target: TARGET_SELECT,
                reason = "no_schemes",
                "short_circuit",
            );
            return Ok(array);
        }

        // Nothing to compress if empty or all-null.
        if array.is_empty() {
            tracing::debug!(
                target: TARGET_SELECT,
                reason = "empty",
                "short_circuit",
            );
            return Ok(array);
        }
        if array.all_invalid()? {
            tracing::debug!(
                target: TARGET_SELECT,
                reason = "all_null",
                "short_circuit",
            );
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

        let Some((winner, estimated_ratio)) =
            self.choose_best_scheme(&eligible_schemes, &mut data, ctx.clone())?
        else {
            // No scheme beat the canonical encoding.
            tracing::debug!(
                target: TARGET_SELECT,
                reason = "fell_through",
                candidate_count = eligible_schemes.len(),
                "short_circuit",
            );
            return Ok(data.into_array());
        };

        tracing::debug!(
            target: TARGET_SELECT,
            scheme = %winner.id(),
            estimated_ratio,
            candidate_count = eligible_schemes.len(),
            "scheme.winner",
        );

        // Wrap the actual encode in its own span so tracing-perfetto /
        // tracing-timing get a distinct timing frame per scheme compression.
        let compressed = {
            let _encode_span = tracing::trace_span!(
                target: TARGET_ENCODE,
                "scheme.compress",
                scheme = %winner.id(),
                before_nbytes,
            )
            .entered();
            winner.compress(self, &mut data, ctx)?
        };

        let after_nbytes = compressed.nbytes();
        // Guard against division by zero: a zero-byte output is legal (e.g. constant
        // arrays) so we clamp to 1 for the display ratio rather than emit NaN/Inf.
        let actual_ratio = before_nbytes as f64 / after_nbytes.max(1) as f64;
        let accepted = after_nbytes < before_nbytes;

        tracing::debug!(
            target: TARGET_ENCODE,
            scheme = %winner.id(),
            before_nbytes,
            after_nbytes,
            estimated_ratio,
            actual_ratio,
            accepted,
            "scheme.compress_result",
        );

        if accepted {
            return Ok(compressed);
        }

        // Winner was picked but its output was not smaller than the canonical input.
        // This is silent in the old code and hides real compressor bugs (bad estimate,
        // pathological data). Surface it explicitly.
        tracing::debug!(
            target: TARGET_SELECT,
            reason = "larger_output",
            scheme = %winner.id(),
            before_nbytes,
            after_nbytes,
            estimated_ratio,
            actual_ratio,
            "short_circuit",
        );

        Ok(data.into_array())
    }

    /// Calls [`expected_compression_ratio`] on each candidate and returns the scheme with the
    /// highest estimated compression ratio, or `None` if no scheme exceeds 1.0. Ties are broken
    /// by registration order (earlier in the list wins).
    ///
    /// The returned `f64` is the scheme's estimated compression ratio. Schemes that returned
    /// [`CompressionEstimate::AlwaysUse`] do not provide a numeric estimate; we use
    /// [`f64::INFINITY`] as a "perfect ratio" sentinel for them so the caller can still
    /// short-circuit and so observability events can emit a single consistent field.
    ///
    /// [`expected_compression_ratio`]: Scheme::expected_compression_ratio
    fn choose_best_scheme(
        &self,
        schemes: &[&'static dyn Scheme],
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<Option<(&'static dyn Scheme, f64)>> {
        let mut best: Option<(&'static dyn Scheme, f64)> = None;

        // TODO(connor): Might want to use an `im` data structure inside of `ctx` if the clones here
        // are expensive.
        for &scheme in schemes {
            let estimate = scheme.expected_compression_ratio(data, ctx.clone());

            // Emit the initial estimate verdict for every scheme the compressor looks at.
            // For `Ratio` this carries the numeric estimate directly; for `Sample` and
            // `Estimate` the ratio is unknown at this point and will be reported via a
            // follow-up `scheme.evaluated.resolved` event once computed.
            emit_scheme_evaluated(scheme, &estimate);

            match estimate {
                CompressionEstimate::Skip => {}
                CompressionEstimate::AlwaysUse => return Ok(Some((scheme, f64::INFINITY))),
                CompressionEstimate::Ratio(ratio) => {
                    if is_better_ratio(ratio, &best) {
                        best = Some((scheme, ratio));
                    }
                }
                CompressionEstimate::Sample => {
                    let sample_ratio = estimate_compression_ratio_with_sampling(
                        scheme,
                        self,
                        data.array(),
                        ctx.clone(),
                    )?;

                    tracing::trace!(
                        target: TARGET_SELECT,
                        scheme = %scheme.id(),
                        kind = "Sample",
                        ratio = sample_ratio,
                        "scheme.evaluated.resolved",
                    );

                    if is_better_ratio(sample_ratio, &best) {
                        best = Some((scheme, sample_ratio));
                    }
                }
                // TODO(connor): Is there a way to deduplicate some of this code?
                CompressionEstimate::Estimate(estimate_callback) => {
                    let estimate = estimate_callback(self, data, ctx.clone())?;

                    match estimate {
                        CompressionEstimate::Skip => {
                            tracing::trace!(
                                target: TARGET_SELECT,
                                scheme = %scheme.id(),
                                kind = "Estimate",
                                resolved_kind = "Skip",
                                "scheme.evaluated.resolved",
                            );
                        }
                        CompressionEstimate::AlwaysUse => {
                            return Ok(Some((scheme, f64::INFINITY)));
                        }
                        CompressionEstimate::Ratio(ratio) => {
                            tracing::trace!(
                                target: TARGET_SELECT,
                                scheme = %scheme.id(),
                                kind = "Estimate",
                                resolved_kind = "Ratio",
                                ratio,
                                "scheme.evaluated.resolved",
                            );
                            if is_better_ratio(ratio, &best) {
                                best = Some((scheme, ratio));
                            }
                        }
                        e @ (CompressionEstimate::Sample | CompressionEstimate::Estimate(_)) => {
                            vortex_panic!(
                                "an estimation function returned an invalid variant {e:?}"
                            )
                        }
                    }
                }
            }
        }

        Ok(best)
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
        let compressed_offsets = self.compress_canonical(
            Canonical::Primitive(list_array.offsets().to_primitive().narrow()?),
            offset_ctx,
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
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let compressed_elems = self.compress(list_view.elements())?;

        let offset_ctx = ctx
            .clone()
            .descend_with_scheme(ROOT_SCHEME_ID, root_list_children::OFFSETS);
        let compressed_offsets = self.compress_canonical(
            Canonical::Primitive(list_view.offsets().to_primitive().narrow()?),
            offset_ctx,
        )?;

        let sizes_ctx = ctx.descend_with_scheme(ROOT_SCHEME_ID, root_list_children::SIZES);
        let compressed_sizes = self.compress_canonical(
            Canonical::Primitive(list_view.sizes().to_primitive().narrow()?),
            sizes_ctx,
        )?;

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
    use crate::scheme::SchemeExt;

    fn compressor() -> CascadingCompressor {
        CascadingCompressor::new(vec![&IntDictScheme, &FloatDictScheme, &StringDictScheme])
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
