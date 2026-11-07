// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Core cascading compression flow.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::CanonicalValidity;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::Masked;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::UnionArray;
use vortex_array::arrays::Variant;
use vortex_array::arrays::VariantArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::listview::list_from_list_view;
use vortex_array::arrays::masked::MaskedArraySlotsExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::arrays::union::UnionArrayExt;
use vortex_array::arrays::union::UnionArraySlotsExt;
use vortex_array::arrays::variant::VariantArraySlotsExt;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::CascadingCompressor;
use super::constant;
use super::plan::Decision;
use super::plan::PlanState;
use super::plan::Planned;
use super::select::WinnerEstimate;
use crate::scheme::CompressionEstimate;
use crate::scheme::CompressorContext;
use crate::scheme::DeferredEstimate;
use crate::scheme::EstimateVerdict;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::scheme::SchemeId;
use crate::stats::ArrayAndStats;
use crate::stats::GenerateStatsOptions;
use crate::trace;

/// The outcome of [`CascadingCompressor::choose_and_compress`].
enum Selection {
    /// A scheme, or the compressor's own empty, null or constant handling, produced a new array.
    Compressed(ArrayRef),
    /// Nothing beat the input, which is handed back as it was.
    Unchanged(ArrayRef),
}

impl Selection {
    /// Returns the compression ratio against `before_nbytes`.
    fn ratio(&self, before_nbytes: u64) -> f64 {
        match self {
            Selection::Compressed(array) => before_nbytes as f64 / array.nbytes().max(1) as f64,
            Selection::Unchanged(_) => 1.0,
        }
    }

    /// Returns the decision for this selection, when `scheme` made it.
    fn decision(&self, scheme: SchemeId) -> Decision {
        match self {
            Selection::Compressed(_) => Decision::Scheme(scheme),
            Selection::Unchanged(_) => Decision::Unchanged,
        }
    }

    /// Returns the array, compressed or not.
    fn into_array(self) -> ArrayRef {
        match self {
            Selection::Compressed(array) | Selection::Unchanged(array) => array,
        }
    }
}

impl CascadingCompressor {
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
        self.compress_root(array, CompressorContext::new(), exec_ctx)
    }

    /// Like [`compress`](Self::compress), but replays and records decisions through `plan`.
    pub(super) fn compress_in_plan(
        &self,
        array: &ArrayRef,
        plan: Arc<PlanState>,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        self.compress_root(array, CompressorContext::new().with_plan(plan), exec_ctx)
    }

    /// Canonicalizes and compacts an array, then compresses it with `compress_ctx`.
    fn compress_root(
        &self,
        array: &ArrayRef,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let before_nbytes = exact_uncompressed_size(array, exec_ctx)?;
        let span = trace::compress_span(array.len(), array.dtype(), before_nbytes);
        let _enter = span.enter();

        let canonical = array.clone().execute::<CanonicalValidity>(exec_ctx)?.0;
        let compact = canonical.compact(exec_ctx)?;
        let compressed = self.compress_canonical(compact, compress_ctx, exec_ctx)?;

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
        let compact = canonical.compact(exec_ctx)?;

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
    pub(super) fn compress_canonical(
        &self,
        array: Canonical,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        match array {
            Canonical::Null(null_array) => Ok(null_array.into_array()),
            Canonical::Bool(bool_array) => self
                .choose_and_compress(Canonical::Bool(bool_array), compress_ctx, exec_ctx)
                .map(Selection::into_array),
            Canonical::Primitive(primitive) => self
                .choose_and_compress(Canonical::Primitive(primitive), compress_ctx, exec_ctx)
                .map(Selection::into_array),
            Canonical::Decimal(decimal) => self
                .choose_and_compress(Canonical::Decimal(decimal), compress_ctx, exec_ctx)
                .map(Selection::into_array),
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
            Canonical::Union(union_array) => {
                let type_ids = self.compress(union_array.type_ids(), exec_ctx)?;
                let children = union_array
                    .iter_children()
                    .map(|child| self.compress(child, exec_ctx))
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(
                    UnionArray::try_new(type_ids, union_array.variants().clone(), children)?
                        .into_array(),
                )
            }
            Canonical::List(list_view_array) => {
                if list_view_array.is_zero_copy_to_list() || list_view_array.elements().is_empty() {
                    let list_array = list_from_list_view(list_view_array, exec_ctx)?;
                    self.compress_list_array(list_array, compress_ctx, exec_ctx)
                } else {
                    self.compress_list_view_array(list_view_array, compress_ctx, exec_ctx)
                }
            }
            Canonical::Map(map_array) => self.compress_map_array(map_array, compress_ctx, exec_ctx),
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
            Canonical::VarBinView(varbinview) => self
                .choose_and_compress(Canonical::VarBinView(varbinview), compress_ctx, exec_ctx)
                .map(Selection::into_array),
            Canonical::Extension(ext_array) => {
                // Try scheme-based compression first.
                let selection = self.choose_and_compress(
                    Canonical::Extension(ext_array.clone()),
                    compress_ctx,
                    exec_ctx,
                )?;

                // A constant extension array (that might be masked) is already in its terminal
                // representation, and compressing the storage separately cannot do better.
                if let Selection::Compressed(compressed) = &selection
                    && (compressed.is::<Constant>()
                        || compressed
                            .as_opt::<Masked>()
                            .is_some_and(|masked| masked.child().is::<Constant>()))
                {
                    return Ok(selection.into_array());
                }

                // Also compress the underlying storage array. Some extension schemes can beat the
                // extension storage but still lose to ordinary storage compression.
                let compressed_storage = self.compress(ext_array.storage_array(), exec_ctx)?;
                let storage_compressed =
                    ExtensionArray::new(ext_array.ext_dtype().clone(), compressed_storage)
                        .into_array();

                match selection {
                    Selection::Compressed(scheme_compressed)
                        if scheme_compressed.nbytes() < storage_compressed.nbytes() =>
                    {
                        Ok(scheme_compressed)
                    }
                    // A canonical extension array's storage is only canonical at its top level,
                    // so an unchanged input can still hold lazy, unserializable arrays.
                    Selection::Compressed(_) | Selection::Unchanged(_) => Ok(storage_compressed),
                }
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
    /// If a winner is found and its compressed output is actually smaller than the exact
    /// uncompressed size, that output is returned as [`Selection::Compressed`]. Otherwise, the
    /// original array is returned as [`Selection::Unchanged`], and the caller decides whether it
    /// is a valid result: it is for leaf arrays, but not for arrays like extensions whose children
    /// have yet to be compressed.
    ///
    /// Empty, all-null, and constant arrays are handled by the compressor itself before any
    /// scheme evaluation (constant detection is skipped while compressing samples).
    ///
    /// [`matches`]: Scheme::matches
    /// [`stats_options`]: Scheme::stats_options
    fn choose_and_compress(
        &self,
        canonical: Canonical,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Selection> {
        let eligible_schemes: Vec<&'static dyn Scheme> = self
            .schemes
            .iter()
            .copied()
            .filter(|s| s.matches(&canonical) && !self.is_excluded(*s, &compress_ctx))
            .collect();

        let array: ArrayRef = canonical.into();

        if array.is_empty() {
            return Ok(Selection::Unchanged(array));
        }

        if array.all_invalid(exec_ctx)? {
            return Ok(Selection::Compressed(
                ConstantArray::new(Scalar::null(array.dtype().clone()), array.len()).into_array(),
            ));
        }

        // Constant detection is built into the compressor: a constant leaf always short-circuits
        // scheme selection. Samples are exempt because a constant sample does not imply that the
        // full array is constant.
        //
        // Compare valid values directly so even nullable constants avoid distinct-value counting.
        if !compress_ctx.is_sample() && constant::is_constant_for_compression(&array, exec_ctx)? {
            return constant::compress_as_constant(array, exec_ctx).map(Selection::Compressed);
        }

        // A scheme competes against the bytes the rows reference, not dead bytes in the buffers.
        let before_nbytes = exact_uncompressed_size(&array, exec_ctx)?;

        if eligible_schemes.is_empty() {
            return Ok(Selection::Unchanged(array));
        }

        // A planned scheme that is no longer eligible, or that rejects the array, falls back to
        // the full selection.
        let Some(plan) = compress_ctx.plan() else {
            return self.select_and_compress(
                &eligible_schemes,
                array,
                before_nbytes,
                compress_ctx,
                exec_ctx,
            );
        };

        match plan.planned(compress_ctx.cascade_history()) {
            Some(Planned {
                decision: Decision::Unchanged,
                ..
            }) => {
                record_decision(&compress_ctx, Decision::Unchanged, 1.0);
                return Ok(Selection::Unchanged(array));
            }
            Some(
                planned @ Planned {
                    decision: Decision::Scheme(id),
                    ..
                },
            ) => {
                if let Some(&scheme) = eligible_schemes.iter().find(|s| s.id() == id)
                    && let Some(replayed) =
                        self.replay(scheme, &array, before_nbytes, &compress_ctx, exec_ctx)?
                {
                    // `compress_reusing` checks the root ratio itself.
                    let replayed_ratio = replayed.ratio(before_nbytes);
                    if compress_ctx.cascade_history().is_empty()
                        || plan.accepts(replayed_ratio, &planned)
                    {
                        return Ok(replayed);
                    }

                    // The child data changed: select again and keep the smaller result.
                    let replayed_decision = replayed.decision(id);
                    let selected = self.select_and_compress(
                        &eligible_schemes,
                        array,
                        before_nbytes,
                        compress_ctx.clone(),
                        exec_ctx,
                    )?;
                    if selected.ratio(before_nbytes) > replayed_ratio {
                        return Ok(selected);
                    }

                    record_decision(&compress_ctx, replayed_decision, replayed_ratio);
                    return Ok(replayed);
                }
            }
            None => {}
        }

        self.select_and_compress(
            &eligible_schemes,
            array,
            before_nbytes,
            compress_ctx,
            exec_ctx,
        )
    }

    /// Selects the best of `eligible_schemes` for `array` and compresses with it.
    fn select_and_compress(
        &self,
        eligible_schemes: &[&'static dyn Scheme],
        array: ArrayRef,
        before_nbytes: u64,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Selection> {
        let merged_opts = eligible_schemes
            .iter()
            .fold(GenerateStatsOptions::default(), |acc, s| {
                acc.merge(s.stats_options())
            });
        let compress_ctx = compress_ctx.with_merged_stats_options(merged_opts);

        let data = ArrayAndStats::new(array, merged_opts);

        let Some((winner, winner_estimate)) =
            self.choose_best_scheme(eligible_schemes, &data, compress_ctx.clone(), exec_ctx)?
        else {
            record_decision(&compress_ctx, Decision::Unchanged, 1.0);
            return Ok(Selection::Unchanged(data.into_array()));
        };

        self.compress_with_winner(
            winner,
            data,
            before_nbytes,
            winner_estimate,
            compress_ctx,
            exec_ctx,
        )
    }

    /// Compresses `array` with the replayed `scheme`, or returns `None` if the scheme rejects it.
    ///
    /// Only sampling is skipped: some schemes check preconditions in their verdict or callback.
    fn replay(
        &self,
        scheme: &'static dyn Scheme,
        array: &ArrayRef,
        before_nbytes: u64,
        compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Selection>> {
        let opts = scheme.stats_options();
        let compress_ctx = compress_ctx.clone().with_merged_stats_options(opts);
        let data = ArrayAndStats::new(array.clone(), opts);

        let usable = match scheme.expected_compression_ratio(&data, compress_ctx.clone(), exec_ctx)
        {
            CompressionEstimate::Verdict(EstimateVerdict::Skip) => false,
            CompressionEstimate::Verdict(_)
            | CompressionEstimate::Deferred(DeferredEstimate::Sample) => true,
            CompressionEstimate::Deferred(DeferredEstimate::Callback(callback)) => !matches!(
                callback(self, &data, None, compress_ctx.clone(), exec_ctx)?,
                EstimateVerdict::Skip
            ),
        };

        if !usable {
            return Ok(None);
        }

        // On failure, the caller falls back to the full selection.
        Ok(self
            .compress_with_winner(
                scheme,
                data,
                before_nbytes,
                WinnerEstimate::Replayed,
                compress_ctx,
                exec_ctx,
            )
            .ok())
    }

    /// Compresses `data` with `winner` and keeps the result if it is smaller than `before_nbytes`.
    fn compress_with_winner(
        &self,
        winner: &'static dyn Scheme,
        data: ArrayAndStats,
        before_nbytes: u64,
        winner_estimate: WinnerEstimate,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Selection> {
        // Run the winning scheme's `compress`. On failure, emit an ERROR event carrying the
        // scheme name and cascade history before propagating. Replays are not reported, because
        // the caller falls back to the full selection.
        let error_ctx = trace::enabled_error_context(&compress_ctx);
        let plan_ctx = compress_ctx.clone();
        let _winner_span = trace::winner_compress_span(winner.id(), before_nbytes).entered();
        let compressed = winner
            .compress(self, &data, compress_ctx, exec_ctx)
            .inspect_err(|err| {
                // NB: this is the only way we can tell which scheme panicked / bailed on their
                // data, especially for third-party schemes where the error site may not carry any
                // compressor context.
                if winner_estimate != WinnerEstimate::Replayed {
                    trace::scheme_compress_failed(
                        winner.id(),
                        before_nbytes,
                        error_ctx.as_ref(),
                        err,
                    );
                }
            })?;

        let after_nbytes = compressed.nbytes();
        let actual_ratio = (after_nbytes != 0).then(|| before_nbytes as f64 / after_nbytes as f64);

        let accepted = after_nbytes < before_nbytes;

        trace::record_winner_compress_result(
            after_nbytes,
            winner_estimate.trace_ratio(),
            actual_ratio,
            accepted,
        );

        if accepted {
            let ratio = before_nbytes as f64 / after_nbytes.max(1) as f64;
            record_decision(&plan_ctx, Decision::Scheme(winner.id()), ratio);
            Ok(Selection::Compressed(compressed))
        } else {
            record_decision(&plan_ctx, Decision::Unchanged, 1.0);
            Ok(Selection::Unchanged(data.into_array()))
        }
    }
}

/// Records `decision` at the cascade position of `compress_ctx`, if it carries a plan.
fn record_decision(compress_ctx: &CompressorContext, decision: Decision, ratio: f64) {
    if let Some(plan) = compress_ctx.plan() {
        plan.record(compress_ctx.cascade_history(), decision, ratio);
    }
}

/// The bytes `array`'s rows use once canonical, read from the exact statistic the writer caches
/// on each chunk and computed otherwise. A dtype that cannot be measured, such as variant, is
/// sized by the bytes it holds.
pub(super) fn exact_uncompressed_size(
    array: &ArrayRef,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    if !UncompressedSizeInBytes::supports_dtype(array.dtype()) {
        return Ok(array.nbytes());
    }
    let size = uncompressed_size_in_bytes(array, exec_ctx)?;
    u64::try_from(size).map_err(|e| vortex_err!("uncompressed size does not fit in u64: {e}"))
}
