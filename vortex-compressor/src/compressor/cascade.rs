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
use super::select::WinnerEstimate;
use crate::plan::Plan;
use crate::plan::PlanRecorder;
use crate::plan::SelectionMode;
use crate::scheme::CompressionEstimate;
use crate::scheme::CompressorContext;
use crate::scheme::EstimateVerdict;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::scheme::SchemeId;
use crate::stats::ArrayAndStats;
use crate::stats::GenerateStatsOptions;
use crate::trace;

/// A complete compressed output that a scheme produced while estimating its ratio.
///
/// An estimate callback may store one in its [`ArrayAndStats`]. Its size is a measurement, while
/// the winning scheme was picked on an estimate, so the compressor keeps the trial output
/// whenever it is smaller than what the winner actually produced.
#[derive(Debug)]
pub(crate) struct TrialOutput {
    /// The scheme that produced the output.
    pub(crate) scheme: SchemeId,
    /// The compressed array.
    pub(crate) array: ArrayRef,
}

/// The outcome of [`CascadingCompressor::choose_and_compress`].
enum Selection {
    /// A scheme, or the compressor's own empty, null or constant handling, produced a new array.
    Compressed(ArrayRef),
    /// Nothing beat the input, which is handed back as it was.
    Unchanged(ArrayRef),
}

impl Selection {
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
        let before_nbytes = exact_uncompressed_size(array, exec_ctx)?;
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
            .with_selection(SelectionMode::follow(plan.clone(), SelectionMode::Estimate));
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
    /// If the cascade budget is exhausted, the child is returned as-is, apart from the validity of
    /// a canonical leaf, which is still compressed. Otherwise, the child context is created by
    /// descending and recording the parent scheme + child index, and compression proceeds
    /// normally.
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
            return self.with_compressed_validity(child, exec_ctx);
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
            leaf @ (Canonical::Bool(_)
            | Canonical::Primitive(_)
            | Canonical::Decimal(_)
            | Canonical::VarBinView(_)) => self.compress_leaf(leaf, compress_ctx, exec_ctx),
            Canonical::Struct(struct_array) => {
                let fields = struct_array
                    .iter_unmasked_fields()
                    .map(|field| self.compress(field, exec_ctx))
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(StructArray::try_new(
                    struct_array.names().clone(),
                    fields,
                    struct_array.len(),
                    self.compress_validity(struct_array.validity()?, exec_ctx)?,
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
                // Canonicalize the elements first, so that schemes see the list's real size and
                // their output is accepted only if it beats it.
                let elements = fsl_array
                    .elements()
                    .clone()
                    .execute::<CanonicalValidity>(exec_ctx)?
                    .0
                    .compact(exec_ctx)?
                    .into_array();
                let fsl_array = FixedSizeListArray::try_new(
                    elements,
                    fsl_array.list_size(),
                    fsl_array.validity()?,
                    fsl_array.len(),
                )?;

                // Schemes over the whole list can drop the elements of null rows, which
                // compressing the elements alone has to keep.
                if let Selection::Compressed(compressed) = self.choose_and_compress(
                    Canonical::FixedSizeList(fsl_array.clone()),
                    compress_ctx,
                    exec_ctx,
                )? {
                    return Ok(compressed);
                }

                let compressed_elems = self.compress(fsl_array.elements(), exec_ctx)?;

                Ok(FixedSizeListArray::try_new(
                    compressed_elems,
                    fsl_array.list_size(),
                    self.compress_validity(fsl_array.validity()?, exec_ctx)?,
                    fsl_array.len(),
                )?
                .into_array())
            }
            Canonical::Extension(ext_array) => {
                // Try scheme-based compression first.
                let selection = self.choose_and_compress(
                    Canonical::Extension(ext_array.clone()),
                    compress_ctx.clone(),
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
                    Selection::Compressed(_) | Selection::Unchanged(_) => {
                        // The storage was compressed adaptively, so no plan describes it.
                        compress_ctx.record(Plan::Adaptive);
                        Ok(storage_compressed)
                    }
                }
            }
            Canonical::Variant(variant_array) => {
                // Storage under a lazy array, such as a slice, would be written with it.
                // Executing it yields the storage encoding, which executes to a Variant over
                // itself, and builds chunks of storage into one.
                let executed = variant_array
                    .core_storage()
                    .clone()
                    .execute::<VariantArray>(exec_ctx)?;
                let core_storage = if executed.shredded().is_none() {
                    executed.core_storage().clone()
                } else {
                    variant_array.core_storage().clone()
                };
                let core_storage = self.compress_physical_slots(&core_storage, exec_ctx)?;
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

    /// Compresses a bool, primitive, decimal or string leaf with the best scheme.
    ///
    /// A scheme that stores the leaf's validity compresses it itself. A leaf no scheme compresses
    /// keeps its values, and its validity bitmap is compressed on its own.
    fn compress_leaf(
        &self,
        leaf: Canonical,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        match self.choose_and_compress(leaf, compress_ctx, exec_ctx)? {
            Selection::Compressed(compressed) => Ok(compressed),
            Selection::Unchanged(array) => self.with_compressed_validity(&array, exec_ctx),
        }
    }

    /// The main scheme-selection entry point for a single leaf array.
    ///
    /// Filters allowed schemes by [`matches`] and exclusion rules, merges their [`stats_options`]
    /// into a single [`GenerateStatsOptions`], and chooses a scheme as the context's selection
    /// mode directs: by estimated compression ratio (the default), by following a [`Plan`], at
    /// random, or by exhaustive search.
    ///
    /// If a scheme is chosen and its compressed output is actually smaller than the exact
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
            compress_ctx.record(Plan::Canonical);
            return Ok(Selection::Unchanged(array));
        }

        if array.all_invalid(exec_ctx)? {
            compress_ctx.record(Plan::Constant);
            return Ok(Selection::Compressed(
                ConstantArray::new(Scalar::null(array.dtype().clone()), array.len()).into_array(),
            ));
        }

        // Constant detection is built into the compressor: a constant leaf always short-circuits
        // scheme selection, whatever the selection mode. Samples are exempt because a constant
        // sample does not imply that the full array is constant.
        //
        // Compare valid values directly so even nullable constants avoid distinct-value counting.
        if !compress_ctx.is_sample() && constant::is_constant_for_compression(&array, exec_ctx)? {
            let compressed = constant::compress_as_constant(
                array.clone(),
                |validity, exec_ctx| self.compress_validity(validity, exec_ctx),
                exec_ctx,
            )?;
            // The constant encoding hands the array back when it is not smaller.
            return Ok(if ArrayRef::ptr_eq(&compressed, &array) {
                compress_ctx.record(Plan::Canonical);
                Selection::Unchanged(array)
            } else {
                compress_ctx.record(Plan::Constant);
                Selection::Compressed(compressed)
            });
        }

        // A scheme competes against the bytes the rows reference, not dead bytes in the buffers.
        let before_nbytes = exact_uncompressed_size(&array, exec_ctx)?;

        let SelectionMode::Follow { plan, fallback } = compress_ctx.selection().clone() else {
            return self.select_and_compress(
                array,
                &eligible_schemes,
                before_nbytes,
                compress_ctx,
                exec_ctx,
            );
        };

        match plan {
            Plan::Scheme {
                scheme: planned, ..
            } => {
                if let Some(&scheme) = eligible_schemes.iter().find(|s| s.id() == planned) {
                    // Only the planned scheme's stats are needed, so only those are computed.
                    let (data, compress_ctx) =
                        Self::prepare_site(array.clone(), &[scheme], compress_ctx.clone());
                    if let Ok(selection) = self.compress_with_scheme(
                        scheme,
                        &data,
                        before_nbytes,
                        compress_ctx,
                        None,
                        true,
                        exec_ctx,
                    ) {
                        return Ok(selection);
                    }
                }
            }
            Plan::Canonical => {
                compress_ctx.record(Plan::Canonical);
                return Ok(Selection::Unchanged(array));
            }
            Plan::Adaptive | Plan::Constant => {}
        }

        // The plan does not decide this site, or its scheme cannot be applied here.
        let compress_ctx = compress_ctx.with_selection(fallback.as_ref().clone());
        self.select_and_compress(
            array,
            &eligible_schemes,
            before_nbytes,
            compress_ctx,
            exec_ctx,
        )
    }

    /// Chooses and applies a scheme by estimate, at random, or by exhaustive search.
    ///
    /// The caller must have handled empty, all-null and constant arrays.
    fn select_and_compress(
        &self,
        array: ArrayRef,
        eligible_schemes: &[&'static dyn Scheme],
        before_nbytes: u64,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Selection> {
        let (data, compress_ctx) = Self::prepare_site(array, eligible_schemes, compress_ctx);

        if eligible_schemes.is_empty() {
            compress_ctx.record(Plan::Canonical);
            return Ok(Selection::Unchanged(data.into_array()));
        }

        match compress_ctx.selection().clone() {
            SelectionMode::Random(rng) => self.compress_randomly(
                &data,
                eligible_schemes,
                before_nbytes,
                compress_ctx,
                &rng,
                exec_ctx,
            ),
            SelectionMode::Exhaustive { cost, iterations } => {
                let compressed = self.compress_exhaustively(
                    &data,
                    eligible_schemes,
                    compress_ctx,
                    &cost,
                    iterations,
                    exec_ctx,
                )?;
                Ok(if ArrayRef::ptr_eq(&compressed, data.array()) {
                    Selection::Unchanged(compressed)
                } else {
                    Selection::Compressed(compressed)
                })
            }
            SelectionMode::Estimate | SelectionMode::Follow { .. } => {
                // Estimation compresses samples, which must not record into this site.
                let estimate_ctx = compress_ctx.clone().with_recorder(None);
                let Some((winner, winner_estimate)) =
                    self.choose_best_scheme(eligible_schemes, &data, estimate_ctx, exec_ctx)?
                else {
                    compress_ctx.record(Plan::Canonical);
                    return Ok(Selection::Unchanged(data.into_array()));
                };
                self.compress_with_scheme(
                    winner,
                    &data,
                    before_nbytes,
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

    /// Compresses the array with `scheme`, keeping the result only if it is smaller than
    /// `before_nbytes`.
    ///
    /// Records the applied plan, including the plans of the scheme's children, when the context
    /// is recording. Failures are reported as error events, or as debug events when `exploring`
    /// (the scheme was chosen by a plan or at random, and the caller recovers from failures).
    ///
    /// # Errors
    ///
    /// Returns an error if the scheme fails to compress the array.
    #[expect(
        clippy::too_many_arguments,
        reason = "internal helper threading selection state"
    )]
    fn compress_with_scheme(
        &self,
        scheme: &'static dyn Scheme,
        data: &ArrayAndStats,
        before_nbytes: u64,
        compress_ctx: CompressorContext,
        estimate: Option<WinnerEstimate>,
        exploring: bool,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Selection> {
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

        // The winner was picked on estimates. A trial output is measured, so it wins when smaller.
        let (compressed, trial_scheme) = match data.get::<TrialOutput>() {
            Some(trial) if trial.array.nbytes() < compressed.nbytes() => {
                (trial.array.clone(), Some(trial.scheme))
            }
            _ => (compressed, None),
        };

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
            match (trial_scheme, children) {
                // A trial output's children were not recorded, so replay chooses them adaptively.
                (Some(trial_scheme), _) => {
                    compress_ctx.record(Plan::scheme(trial_scheme, Vec::new()));
                }
                (None, Some(children)) => compress_ctx.record(Plan::Scheme {
                    scheme: scheme.id(),
                    children: children.children(scheme.num_children()),
                }),
                (None, None) => {}
            }
            Ok(Selection::Compressed(compressed))
        } else {
            compress_ctx.record(Plan::Canonical);
            Ok(Selection::Unchanged(data.array().clone()))
        }
    }

    /// Compresses the array with canonical or a randomly chosen scheme.
    ///
    /// Schemes that fail on the array are dropped and another is drawn, so this always succeeds.
    fn compress_randomly(
        &self,
        data: &ArrayAndStats,
        eligible_schemes: &[&'static dyn Scheme],
        before_nbytes: u64,
        compress_ctx: CompressorContext,
        rng: &Mutex<StdRng>,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Selection> {
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
                return Ok(Selection::Unchanged(data.array().clone()));
            };
            if let Ok(selection) = self.compress_with_scheme(
                scheme,
                data,
                before_nbytes,
                compress_ctx.clone(),
                None,
                true,
                exec_ctx,
            ) {
                return Ok(selection);
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

/// The bytes `array`'s rows use once canonical, read from the exact statistic the writer caches
/// on each chunk and computed otherwise. A dtype that cannot be measured, such as variant, is
/// sized by the bytes it holds.
pub(crate) fn exact_uncompressed_size(
    array: &ArrayRef,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    if !UncompressedSizeInBytes::supports_dtype(array.dtype()) {
        return Ok(array.nbytes());
    }
    let size = uncompressed_size_in_bytes(array, exec_ctx)?;
    u64::try_from(size).map_err(|e| vortex_err!("uncompressed size does not fit in u64: {e}"))
}
