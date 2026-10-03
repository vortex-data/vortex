// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Core cascading compression flow.

use std::iter;
use std::sync::Arc;

use parking_lot::Mutex;
use rand::RngExt;
use rand::prelude::StdRng;
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

use super::CascadingCompressor;
use super::constant;
use super::select::WinnerEstimate;
use crate::plan::Plan;
use crate::plan::PlanRecorder;
use crate::plan::Selection;
use crate::scheme::CompressionEstimate;
use crate::scheme::CompressorContext;
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
        let before_nbytes = array.nbytes();
        let span = trace::compress_span(array.len(), array.dtype(), before_nbytes);
        let _enter = span.enter();

        let compact = Self::canonicalize_input(array, exec_ctx)?;
        let compressed = self.compress_canonical(compact, CompressorContext::new(), exec_ctx)?;

        trace::record_compress_outcome(&span, before_nbytes, compressed.nbytes());

        Ok(compressed)
    }

    /// Compresses an array following `plan`.
    ///
    /// Sites the plan leaves [`Plan::Adaptive`], and sites where its scheme does not apply or
    /// fails, are compressed by estimate-based selection as in [`compress`](Self::compress).
    /// Following a plan skips sampling, and computes only the stats of the planned schemes.
    ///
    /// The plan applies to leaf arrays. The fields of nested arrays are compressed adaptively.
    ///
    /// # Errors
    ///
    /// Returns an error if canonicalization or compression fails.
    pub fn compress_with_plan(
        &self,
        array: &ArrayRef,
        plan: &Plan,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let compact = Self::canonicalize_input(array, exec_ctx)?;
        let compress_ctx = CompressorContext::new()
            .with_selection(Selection::follow(plan.clone(), Selection::Estimate));
        self.compress_canonical(compact, compress_ctx, exec_ctx)
    }

    /// Compresses an array like [`compress`](Self::compress), and returns the plan it applied.
    ///
    /// Following the returned plan with [`compress_with_plan`](Self::compress_with_plan)
    /// reproduces the same compressed array. Nested arrays return [`Plan::Adaptive`].
    ///
    /// # Errors
    ///
    /// Returns an error if canonicalization or compression fails.
    pub fn compress_recording_plan(
        &self,
        array: &ArrayRef,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<(ArrayRef, Plan)> {
        let compact = Self::canonicalize_input(array, exec_ctx)?;
        let recorder = Arc::new(PlanRecorder::new(None));
        let compress_ctx = CompressorContext::new().with_recorder(Some(Arc::clone(&recorder)));
        let compressed = self.compress_canonical(compact, compress_ctx, exec_ctx)?;
        Ok((compressed, recorder.get(0).unwrap_or(Plan::Adaptive)))
    }

    /// Canonicalizes and compacts an input array before compression.
    pub(crate) fn canonicalize_input(
        array: &ArrayRef,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Canonical> {
        let canonical = array.clone().execute::<CanonicalValidity>(exec_ctx)?.0;
        canonical.compact(exec_ctx)
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
            parent_ctx.record_child(parent_id, child_index, Plan::Canonical);
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
    pub(crate) fn compress_canonical(
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
                let compressed_storage = self.compress(ext_array.storage_array(), exec_ctx)?;
                let storage_compressed =
                    ExtensionArray::new(ext_array.ext_dtype().clone(), compressed_storage)
                        .into_array();

                if scheme_compressed.nbytes() < storage_compressed.nbytes() {
                    Ok(scheme_compressed)
                } else {
                    // The storage was compressed adaptively, so no plan describes it.
                    compress_ctx.record(Plan::Adaptive);
                    Ok(storage_compressed)
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
    /// Filters allowed schemes by [`matches`] and exclusion rules, then chooses a scheme as the
    /// context's selection mode directs: by estimated compression ratio (the default), by
    /// following a [`Plan`], at random, or by exhaustive search.
    ///
    /// If the chosen scheme's compressed output is actually smaller, that output is returned.
    /// Otherwise, the original array is returned unchanged.
    ///
    /// Empty, all-null, and constant arrays are handled by the compressor itself before any
    /// scheme evaluation (constant detection is skipped while compressing samples).
    ///
    /// [`matches`]: Scheme::matches
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
            compress_ctx.record(Plan::Canonical);
            return Ok(array);
        }

        if array.all_invalid(exec_ctx)? {
            compress_ctx.record(Plan::Constant);
            return Ok(
                ConstantArray::new(Scalar::null(array.dtype().clone()), array.len()).into_array(),
            );
        }

        let Selection::Follow { plan, fallback } = compress_ctx.selection().clone() else {
            return self.select_and_compress(array, &eligible_schemes, compress_ctx, exec_ctx);
        };

        match plan {
            Plan::Scheme {
                scheme: planned, ..
            } => {
                if let Some(&scheme) = eligible_schemes.iter().find(|s| s.id() == planned) {
                    // Only the planned scheme's stats are needed, so only those are computed.
                    let (data, compress_ctx) =
                        Self::prepare_site(array.clone(), &[scheme], compress_ctx.clone());
                    if let Some(compressed) =
                        Self::try_compress_constant(&data, &compress_ctx, exec_ctx)?
                    {
                        return Ok(compressed);
                    }
                    if let Ok(compressed) = self.compress_with_scheme(
                        scheme,
                        &data,
                        compress_ctx.clone(),
                        None,
                        true,
                        exec_ctx,
                    ) {
                        return Ok(compressed);
                    }
                }
            }
            Plan::Canonical => {
                let (data, compress_ctx) = Self::prepare_site(array, &[], compress_ctx);
                if let Some(compressed) =
                    Self::try_compress_constant(&data, &compress_ctx, exec_ctx)?
                {
                    return Ok(compressed);
                }
                compress_ctx.record(Plan::Canonical);
                return Ok(data.into_array());
            }
            Plan::Adaptive | Plan::Constant => {}
        }

        // The plan does not decide this site, or its scheme cannot be applied here.
        let compress_ctx = compress_ctx.with_selection(fallback.as_ref().clone());
        self.select_and_compress(array, &eligible_schemes, compress_ctx, exec_ctx)
    }

    /// Chooses and applies a scheme by estimate, at random, or by exhaustive search.
    ///
    /// The caller must have handled empty and all-null arrays.
    fn select_and_compress(
        &self,
        array: ArrayRef,
        eligible_schemes: &[&'static dyn Scheme],
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let (data, compress_ctx) = Self::prepare_site(array, eligible_schemes, compress_ctx);

        if let Some(compressed) = Self::try_compress_constant(&data, &compress_ctx, exec_ctx)? {
            return Ok(compressed);
        }

        if eligible_schemes.is_empty() {
            compress_ctx.record(Plan::Canonical);
            return Ok(data.into_array());
        }

        match compress_ctx.selection().clone() {
            Selection::Random(rng) => {
                self.compress_randomly(&data, eligible_schemes, compress_ctx, &rng, exec_ctx)
            }
            Selection::Exhaustive { cost, iterations } => self.compress_exhaustively(
                &data,
                eligible_schemes,
                compress_ctx,
                &cost,
                iterations,
                exec_ctx,
            ),
            Selection::Estimate | Selection::Follow { .. } => {
                // Estimation compresses samples, which must not record into this site.
                let estimate_ctx = compress_ctx.clone().with_recorder(None);
                let Some((winner, winner_estimate)) =
                    self.choose_best_scheme(eligible_schemes, &data, estimate_ctx, exec_ctx)?
                else {
                    compress_ctx.record(Plan::Canonical);
                    return Ok(data.into_array());
                };
                self.compress_with_scheme(
                    winner,
                    &data,
                    compress_ctx,
                    Some(winner_estimate),
                    false,
                    exec_ctx,
                )
            }
        }
    }

    /// Bundles `array` with the stats that `schemes` need, and records those stats options in the
    /// context.
    fn prepare_site(
        array: ArrayRef,
        schemes: &[&'static dyn Scheme],
        compress_ctx: CompressorContext,
    ) -> (ArrayAndStats, CompressorContext) {
        let merged_opts = schemes
            .iter()
            .fold(GenerateStatsOptions::default(), |acc, s| {
                acc.merge(s.stats_options())
            });
        (
            ArrayAndStats::new(array, merged_opts),
            compress_ctx.with_merged_stats_options(merged_opts),
        )
    }

    /// Encodes the array as a constant if it is one, returning `None` otherwise.
    ///
    /// Constant detection is built into the compressor: a constant leaf always short-circuits
    /// scheme selection. Samples are exempt because a constant sample does not imply that the
    /// full array is constant.
    fn try_compress_constant(
        data: &ArrayAndStats,
        compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if compress_ctx.is_sample() || !constant::is_constant_for_compression(data, exec_ctx)? {
            return Ok(None);
        }

        let before_nbytes = data.array().nbytes();
        let _winner_span =
            trace::winner_compress_span(constant::CONSTANT_SCHEME_ID, before_nbytes).entered();
        let compressed = constant::compress_constant(data.array(), exec_ctx)?;

        let after_nbytes = compressed.nbytes();
        let actual_ratio = (after_nbytes != 0).then(|| before_nbytes as f64 / after_nbytes as f64);
        let accepted = after_nbytes < before_nbytes;
        trace::record_winner_compress_result(after_nbytes, None, actual_ratio, accepted);

        if accepted {
            compress_ctx.record(Plan::Constant);
            Ok(Some(compressed))
        } else {
            compress_ctx.record(Plan::Canonical);
            Ok(Some(data.array().clone()))
        }
    }

    /// Compresses the array with `scheme`, keeping the result only if it is smaller.
    ///
    /// Records the applied plan, including the plans of the scheme's children, when the context
    /// is recording. Failures are reported as error events, or as debug events when `exploring`
    /// (the scheme was chosen by a plan or at random, and the caller recovers from failures).
    ///
    /// # Errors
    ///
    /// Returns an error if the scheme fails to compress the array.
    fn compress_with_scheme(
        &self,
        scheme: &'static dyn Scheme,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        estimate: Option<WinnerEstimate>,
        exploring: bool,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let before_nbytes = data.array().nbytes();
        let children = compress_ctx
            .recorder()
            .map(|_| Arc::new(PlanRecorder::new(Some(scheme.id()))));
        let scheme_ctx = compress_ctx.clone().with_recorder(children.clone());

        // Run the scheme's `compress`. On failure, emit an event carrying the scheme name and
        // cascade history before propagating.
        let error_ctx = trace::enabled_error_context(&scheme_ctx);
        let _winner_span = trace::winner_compress_span(scheme.id(), before_nbytes).entered();
        let compressed = scheme
            .compress(self, data, scheme_ctx, exec_ctx)
            .inspect_err(|err| {
                // NB: this is the only way we can tell which scheme panicked / bailed on their
                // data, especially for third-party schemes where the error site may not carry any
                // compressor context.
                if exploring {
                    trace::candidate_failed(scheme.id(), err);
                } else {
                    trace::scheme_compress_failed(
                        scheme.id(),
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
            estimate.and_then(WinnerEstimate::trace_ratio),
            actual_ratio,
            accepted,
        );

        if accepted {
            if let Some(children) = children {
                compress_ctx.record(Plan::Scheme {
                    scheme: scheme.id(),
                    children: children.children(scheme.num_children()),
                });
            }
            Ok(compressed)
        } else {
            compress_ctx.record(Plan::Canonical);
            Ok(data.array().clone())
        }
    }

    /// Compresses the array with canonical or a randomly chosen scheme.
    ///
    /// Schemes that fail on the array are dropped and another is drawn, so this always succeeds.
    fn compress_randomly(
        &self,
        data: &ArrayAndStats,
        eligible_schemes: &[&'static dyn Scheme],
        compress_ctx: CompressorContext,
        rng: &Mutex<StdRng>,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let probe_ctx = compress_ctx.clone().with_recorder(None);
        let mut options: Vec<Option<&'static dyn Scheme>> = iter::once(None)
            .chain(
                self.candidate_schemes(eligible_schemes, data, &probe_ctx, exec_ctx)
                    .into_iter()
                    .map(Some),
            )
            .collect();

        loop {
            let index = rng.lock().random_range(0..options.len());
            let Some(scheme) = options.swap_remove(index) else {
                compress_ctx.record(Plan::Canonical);
                return Ok(data.array().clone());
            };
            if let Ok(compressed) =
                self.compress_with_scheme(scheme, data, compress_ctx.clone(), None, true, exec_ctx)
            {
                return Ok(compressed);
            }
        }
    }

    /// Returns the schemes worth trying on the array: those whose estimate does not rule them
    /// out.
    ///
    /// Only immediate [`EstimateVerdict::Skip`] verdicts are honored. They encode preconditions
    /// such as "FoR needs a child to bit-pack" or "the minimum is already zero". Deferred
    /// estimates are not resolved, since the caller compresses the full array anyway.
    pub(crate) fn candidate_schemes(
        &self,
        eligible_schemes: &[&'static dyn Scheme],
        data: &ArrayAndStats,
        compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> Vec<&'static dyn Scheme> {
        eligible_schemes
            .iter()
            .copied()
            .filter(|scheme| {
                !matches!(
                    scheme.expected_compression_ratio(data, compress_ctx.clone(), exec_ctx),
                    CompressionEstimate::Verdict(EstimateVerdict::Skip)
                )
            })
            .collect()
    }
}
