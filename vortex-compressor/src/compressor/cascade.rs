// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Core cascading compression flow.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::CanonicalValidity;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
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

use std::sync::Arc;

use super::CascadingCompressor;
use super::constant;
use crate::CompressionPlan;
use crate::plan::PlanDecision;
use crate::plan::PlanState;
use crate::scheme::CompressionEstimate;
use crate::scheme::CompressorContext;
use crate::scheme::DeferredEstimate;
use crate::scheme::EstimateScore;
use crate::scheme::EstimateVerdict;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::scheme::SchemeId;
use crate::stats::ArrayAndStats;
use crate::stats::GenerateStatsOptions;
use crate::trace;

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

    /// Compresses an array like [`compress`](Self::compress), replaying the scheme decisions of
    /// `hint` where they still apply, and returns the [`CompressionPlan`] of this compression.
    ///
    /// Pass the returned plan as the `hint` for the next similar array (e.g. the next chunk of the
    /// same column) to skip scheme selection. A hinted scheme is reused only if it still applies
    /// to the data and achieves at least 80% of the compression ratio it achieved when it was
    /// recorded. Otherwise that site falls back to a full search.
    ///
    /// # Errors
    ///
    /// Returns an error if canonicalization or compression fails.
    pub fn compress_with_plan(
        &self,
        array: &ArrayRef,
        hint: Option<Arc<CompressionPlan>>,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<(ArrayRef, CompressionPlan)> {
        let plan = Arc::new(PlanState::new(hint));
        let compressed =
            self.compress_root(array, CompressorContext::with_plan(Arc::clone(&plan)), exec_ctx)?;
        Ok((compressed, plan.finish()))
    }

    /// Canonicalizes, compacts and compresses a root array in the given context.
    fn compress_root(
        &self,
        array: &ArrayRef,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let before_nbytes = array.nbytes();
        let span = trace::compress_span(array.len(), array.dtype(), before_nbytes);
        let _enter = span.enter();

        let canonical = array.clone().execute::<CanonicalValidity>(exec_ctx)?.0;
        let compact = canonical.compact(exec_ctx)?;
        let compressed = self.compress_canonical(compact, compress_ctx, exec_ctx)?;

        trace::record_compress_outcome(&span, before_nbytes, compressed.nbytes());

        Ok(compressed)
    }

    /// Compresses a structural child (struct field, list elements, ...) of the array compressed
    /// in `parent_ctx`.
    ///
    /// The child starts a fresh cascade, exactly like a root array, but keeps its place in the
    /// compression plan.
    ///
    /// # Errors
    ///
    /// Returns an error if canonicalization or compression fails.
    pub(super) fn compress_structural_child(
        &self,
        child: &ArrayRef,
        parent_ctx: &CompressorContext,
        child_index: usize,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        self.compress_root(child, parent_ctx.descend_structural(child_index), exec_ctx)
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
                    .enumerate()
                    .map(|(idx, field)| {
                        self.compress_structural_child(field, &compress_ctx, idx, exec_ctx)
                    })
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
                let type_ids = self.compress_structural_child(
                    union_array.type_ids(),
                    &compress_ctx,
                    0,
                    exec_ctx,
                )?;
                let children = union_array
                    .iter_children()
                    .enumerate()
                    .map(|(idx, child)| {
                        self.compress_structural_child(child, &compress_ctx, idx + 1, exec_ctx)
                    })
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
                let compressed_elems = self.compress_structural_child(
                    fsl_array.elements(),
                    &compress_ctx,
                    0,
                    exec_ctx,
                )?;

                Ok(FixedSizeListArray::try_new(
                    compressed_elems,
                    fsl_array.list_size(),
                    fsl_array.validity()?,
                    fsl_array.len(),
                )?
                .into_array())
            }
            Canonical::VarBinView(varbinview) => {
                self.choose_and_compress(Canonical::VarBinView(varbinview), compress_ctx, exec_ctx)
            }
            Canonical::Extension(ext_array) => {
                // Try scheme-based compression first.
                let scheme_compressed = self.choose_and_compress(
                    Canonical::Extension(ext_array.clone()),
                    compress_ctx.clone(),
                    exec_ctx,
                )?;

                // A constant extension array (that might be masked) is already in its terminal
                // representation, and compressing the storage separately cannot do better.
                if scheme_compressed.is::<Constant>() {
                    return Ok(scheme_compressed);
                }
                if let Some(masked) = scheme_compressed.as_opt::<Masked>()
                    && masked.child().is::<Constant>()
                {
                    return Ok(scheme_compressed);
                }

                // Also compress the underlying storage array. Some extension schemes can beat the
                // extension storage but still lose to ordinary storage compression.
                let compressed_storage = self.compress_structural_child(
                    ext_array.storage_array(),
                    &compress_ctx,
                    0,
                    exec_ctx,
                )?;
                let storage_compressed =
                    ExtensionArray::new(ext_array.ext_dtype().clone(), compressed_storage)
                        .into_array();

                if scheme_compressed.nbytes() < storage_compressed.nbytes() {
                    Ok(scheme_compressed)
                } else {
                    Ok(storage_compressed)
                }
            }
            Canonical::Variant(variant_array) => {
                let core_storage = self.compress_physical_slots(
                    variant_array.core_storage(),
                    &compress_ctx.descend_structural(0),
                    exec_ctx,
                )?;
                let shredded = variant_array
                    .shredded()
                    .map(|arr| {
                        // Avoid stack-overflow for variant shredded values
                        if arr.is::<Variant>() {
                            self.compress_physical_slots(
                                arr,
                                &compress_ctx.descend_structural(1),
                                exec_ctx,
                            )
                        } else {
                            self.compress_structural_child(arr, &compress_ctx, 1, exec_ctx)
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
    ) -> VortexResult<ArrayRef> {
        let eligible_schemes: Vec<&'static dyn Scheme> = self
            .schemes
            .iter()
            .copied()
            .filter(|s| s.matches(&canonical) && !self.is_excluded(*s, &compress_ctx))
            .collect();

        let array: ArrayRef = canonical.into();

        if array.is_empty() {
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

        // Constant detection is built into the compressor: a constant leaf always short-circuits
        // scheme selection. Samples are exempt because a constant sample does not imply that the
        // full array is constant.
        if !compress_ctx.is_sample() && constant::is_constant_for_compression(&data, exec_ctx)? {
            let _winner_span =
                trace::winner_compress_span(constant::CONSTANT_SCHEME_ID, before_nbytes).entered();
            let compressed = constant::compress_constant(data.array(), exec_ctx)?;

            let after_nbytes = compressed.nbytes();
            let actual_ratio =
                (after_nbytes != 0).then(|| before_nbytes as f64 / after_nbytes as f64);
            let accepted = after_nbytes < before_nbytes;
            trace::record_winner_compress_result(after_nbytes, None, actual_ratio, accepted);

            return if accepted {
                Ok(compressed)
            } else {
                Ok(data.into_array())
            };
        }

        if eligible_schemes.is_empty() {
            return Ok(data.into_array());
        }

        if let Some(decision) = compress_ctx.plan_hint()
            && let Some(compressed) = self.replay_plan_decision(
                decision,
                &eligible_schemes,
                &data,
                &compress_ctx,
                exec_ctx,
            )?
        {
            return Ok(compressed);
        }

        let Some((winner, winner_estimate)) =
            self.choose_best_scheme(&eligible_schemes, &data, compress_ctx.clone(), exec_ctx)?
        else {
            compress_ctx.record_plan_decision(PlanDecision::Uncompressed);
            return Ok(data.into_array());
        };

        // Run the winning scheme's `compress`. On failure, emit an ERROR event carrying the
        // scheme name and cascade history before propagating.
        let error_ctx = trace::enabled_error_context(&compress_ctx);
        let plan_ctx = compress_ctx.clone();
        let checkpoint = plan_ctx.plan_checkpoint();
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

        let accepted = after_nbytes < before_nbytes;

        trace::record_winner_compress_result(
            after_nbytes,
            winner_estimate.trace_ratio(),
            actual_ratio,
            accepted,
        );

        if accepted {
            plan_ctx.record_plan_decision(PlanDecision::Scheme {
                id: winner.id(),
                ratio: actual_ratio.unwrap_or(f64::INFINITY),
            });
            Ok(compressed)
        } else {
            plan_ctx.plan_rollback(checkpoint);
            plan_ctx.record_plan_decision(PlanDecision::Uncompressed);
            Ok(data.into_array())
        }
    }

    /// Replays a hinted plan decision for a single leaf array, skipping scheme selection.
    ///
    /// Returns `None` if the decision no longer applies, in which case the caller performs a full
    /// scheme search. A hinted scheme no longer applies if it is not eligible, if its estimate
    /// rejects the data, or if it compresses to less than [`PLAN_REPLAY_MIN_RATIO_FRACTION`] of
    /// the ratio it achieved when the plan was recorded. Sampling estimates are not run, and
    /// decisions recorded while compressing with a rejected scheme are discarded.
    fn replay_plan_decision(
        &self,
        decision: PlanDecision,
        eligible_schemes: &[&'static dyn Scheme],
        data: &ArrayAndStats,
        compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let (id, recorded_ratio) = match decision {
            PlanDecision::Uncompressed => {
                compress_ctx.record_plan_decision(PlanDecision::Uncompressed);
                return Ok(Some(data.array().clone()));
            }
            PlanDecision::Scheme { id, ratio } => (id, ratio),
        };

        let Some(scheme) = eligible_schemes.iter().copied().find(|s| s.id() == id) else {
            return Ok(None);
        };

        // Schemes reject data they cannot encode from their estimate, so it must still run. Only
        // sampling is skipped, and callbacks run without a threshold to compete against.
        let applies = match scheme.expected_compression_ratio(data, compress_ctx.clone(), exec_ctx)
        {
            CompressionEstimate::Verdict(verdict) => verdict_applies(&verdict),
            CompressionEstimate::Deferred(DeferredEstimate::Sample) => true,
            CompressionEstimate::Deferred(DeferredEstimate::Callback(callback)) => {
                verdict_applies(&callback(self, data, None, compress_ctx.clone(), exec_ctx)?)
            }
        };
        if !applies {
            return Ok(None);
        }

        let checkpoint = compress_ctx.plan_checkpoint();

        let before_nbytes = data.array().nbytes();
        let error_ctx = trace::enabled_error_context(compress_ctx);
        let _winner_span = trace::winner_compress_span(id, before_nbytes).entered();
        let compressed = scheme
            .compress(self, data, compress_ctx.clone(), exec_ctx)
            .inspect_err(|err| {
                trace::scheme_compress_failed(id, before_nbytes, error_ctx.as_ref(), err);
            })?;

        let after_nbytes = compressed.nbytes();
        let actual_ratio = if after_nbytes == 0 {
            f64::INFINITY
        } else {
            before_nbytes as f64 / after_nbytes as f64
        };
        let accepted = after_nbytes < before_nbytes
            && actual_ratio >= recorded_ratio * PLAN_REPLAY_MIN_RATIO_FRACTION;

        trace::record_winner_compress_result(
            after_nbytes,
            Some(recorded_ratio),
            actual_ratio.is_finite().then_some(actual_ratio),
            accepted,
        );

        if !accepted {
            compress_ctx.plan_rollback(checkpoint);
            return Ok(None);
        }

        compress_ctx.record_plan_decision(PlanDecision::Scheme {
            id,
            ratio: actual_ratio,
        });
        Ok(Some(compressed))
    }
}

/// The fraction of a plan decision's recorded compression ratio that replaying it must achieve.
///
/// Below this, the data has drifted enough that a full scheme search is worth its cost.
const PLAN_REPLAY_MIN_RATIO_FRACTION: f64 = 0.8;

/// Returns `true` if a scheme's estimate verdict allows replaying it on the data.
fn verdict_applies(verdict: &EstimateVerdict) -> bool {
    match verdict {
        EstimateVerdict::Skip => false,
        EstimateVerdict::AlwaysUse => true,
        EstimateVerdict::Ratio(ratio) => EstimateScore::FiniteCompression(*ratio).is_valid(),
    }
}
