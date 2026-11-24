// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Core cascading compression flow.

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
use crate::scheme::CompressionPolicy;
use crate::scheme::CompressorContext;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::scheme::SchemeId;
use crate::scheme::SchemeSelection;
use crate::scheme::SiteOutcome;
use crate::scheme::SiteRequest;
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
    /// Nothing beat the input, which is handed back as it was, with its null count when the
    /// statistics generated while choosing a scheme already hold it.
    Unchanged(ArrayRef, Option<u32>),
}

impl Selection {
    /// Hands back the input of an evaluation that nothing beat, keeping the null count its
    /// statistics found.
    fn unchanged(data: ArrayAndStats) -> Self {
        let null_count = data.cached_null_count();
        Selection::Unchanged(data.into_array(), null_count)
    }

    /// Returns the array, compressed or not.
    fn into_array(self) -> ArrayRef {
        match self {
            Selection::Compressed(array) | Selection::Unchanged(array, _) => array,
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

        let canonical = array.clone().execute::<CanonicalValidity>(exec_ctx)?.0;
        let compact = canonical.compact(exec_ctx)?;
        let compressed = self.compress_canonical(compact, CompressorContext::new(), exec_ctx)?;

        trace::record_compress_outcome(&span, before_nbytes, compressed.nbytes());

        Ok(compressed)
    }

    /// Compresses a child array of a cascading scheme.
    ///
    /// The compressor calls this for the children a scheme registers in its
    /// [`Encoded`](crate::scheme::Encoded). A scheme that implements only
    /// [`Scheme::compress`] calls it for its own children.
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
            if !child.is_canonical() {
                return Ok(child.clone());
            }
            let canonical = child.clone().execute::<Canonical>(exec_ctx)?;
            return Ok(self
                .compress_leaf_validity(canonical, None, parent_ctx, exec_ctx)?
                .into_array());
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
                    self.resolve_validity(struct_array.validity()?, &compress_ctx, exec_ctx)?,
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
                    compress_ctx.clone(),
                    exec_ctx,
                )? {
                    return Ok(compressed);
                }

                let compressed_elems = self.compress(fsl_array.elements(), exec_ctx)?;

                Ok(FixedSizeListArray::try_new(
                    compressed_elems,
                    fsl_array.list_size(),
                    self.resolve_validity(fsl_array.validity()?, &compress_ctx, exec_ctx)?,
                    fsl_array.len(),
                )?
                .into_array())
            }
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
                    Selection::Compressed(_) | Selection::Unchanged(..) => Ok(storage_compressed),
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
    /// The winning scheme registers the leaf's validity itself. A leaf no scheme compresses keeps
    /// its values, and its validity bitmap is compressed on its own.
    fn compress_leaf(
        &self,
        leaf: Canonical,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        match self.choose_and_compress(leaf, compress_ctx.clone(), exec_ctx)? {
            Selection::Compressed(compressed) => Ok(compressed),
            Selection::Unchanged(array, null_count) => {
                let canonical = array.execute::<Canonical>(exec_ctx)?;
                Ok(self
                    .compress_leaf_validity(canonical, null_count, &compress_ctx, exec_ctx)?
                    .into_array())
            }
        }
    }

    /// The main scheme-selection entry point for a single leaf array.
    ///
    /// Filters allowed schemes by [`matches`] and exclusion rules, merges their [`stats_options`]
    /// into a single [`GenerateStatsOptions`], and picks the winner by estimated compression
    /// ratio.
    ///
    /// The winner's [`encode`] output has its registered children compressed by
    /// [`compress_encoded`](Self::compress_encoded).
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
    /// [`encode`]: Scheme::encode
    fn choose_and_compress(
        &self,
        canonical: Canonical,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Selection> {
        let mut eligible_schemes: Vec<&'static dyn Scheme> = self
            .schemes
            .iter()
            .copied()
            .filter(|s| s.matches(&canonical) && !self.is_excluded(*s, &compress_ctx))
            .collect();

        let array: ArrayRef = canonical.into();

        if array.is_empty() {
            return Ok(Selection::Unchanged(array, None));
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
            let compressed = constant::compress_as_constant(
                array.clone(),
                |validity, exec_ctx| self.resolve_validity(validity, &compress_ctx, exec_ctx),
                exec_ctx,
            )?;
            // The constant encoding hands the array back when it is not smaller.
            return Ok(if ArrayRef::ptr_eq(&compressed, &array) {
                Selection::Unchanged(array, None)
            } else {
                Selection::Compressed(compressed)
            });
        }

        // A scheme competes against the bytes the rows reference, not dead bytes in the buffers.
        let before_nbytes = exact_uncompressed_size(&array, exec_ctx)?;

        let merged_opts = eligible_schemes
            .iter()
            .fold(GenerateStatsOptions::default(), |acc, s| {
                acc.merge(s.stats_options())
            });
        let compress_ctx = compress_ctx.with_merged_stats_options(merged_opts);

        let data = ArrayAndStats::new(array, merged_opts);

        // Samples are compressed only to estimate, so the policy neither chooses nor observes them.
        let policy = self.policy.as_deref().filter(|_| !compress_ctx.is_sample());

        if eligible_schemes.is_empty() {
            return Ok(observed(
                policy,
                None,
                &compress_ctx,
                Selection::unchanged(data),
            ));
        }

        if let Some(policy) = policy {
            let eligible: Vec<SchemeId> = eligible_schemes.iter().map(|s| s.id()).collect();
            let request = SiteRequest {
                array: data.array(),
                eligible: &eligible,
                ctx: &compress_ctx,
            };
            match policy.select_scheme_with_stats(&request, &data, exec_ctx) {
                SchemeSelection::Estimate => {}
                SchemeSelection::EstimateAmong(ids) => {
                    let among: Vec<&'static dyn Scheme> = eligible_schemes
                        .iter()
                        .copied()
                        .filter(|s| ids.contains(&s.id()))
                        .collect();
                    if !among.is_empty() {
                        eligible_schemes = among;
                    }
                }
                SchemeSelection::Canonical => {
                    return Ok(observed(
                        Some(policy),
                        None,
                        &compress_ctx,
                        Selection::unchanged(data),
                    ));
                }
                SchemeSelection::Scheme(id) => {
                    if let Some(&scheme) = eligible_schemes.iter().find(|s| s.id() == id)
                        && let Some(compressed) =
                            self.try_scheme(scheme, &data, &compress_ctx, before_nbytes, exec_ctx)
                    {
                        return Ok(observed(
                            Some(policy),
                            Some(id),
                            &compress_ctx,
                            Selection::Compressed(compressed),
                        ));
                    }
                }
                SchemeSelection::Best(ids) => {
                    let selection = self.compress_best(
                        policy,
                        &eligible_schemes,
                        &ids,
                        data,
                        &compress_ctx,
                        before_nbytes,
                        exec_ctx,
                    );
                    return Ok(selection);
                }
            }
        }

        let Some((winner, winner_estimate)) =
            self.choose_best_scheme(&eligible_schemes, &data, compress_ctx.clone(), exec_ctx)?
        else {
            return Ok(observed(
                policy,
                None,
                &compress_ctx,
                Selection::unchanged(data),
            ));
        };

        // Run the winning scheme's `encode` and compress the children it registers, unless an
        // estimate already produced its complete output. On failure, emit an ERROR event carrying
        // the scheme name and cascade history before propagating.
        let error_ctx = trace::enabled_error_context(&compress_ctx);
        let _winner_span = trace::winner_compress_span(winner.id(), before_nbytes).entered();
        let compressed = self
            .compress_with_scheme(winner, &data, &compress_ctx, exec_ctx)
            .inspect_err(|err| {
                // NB: this is the only way we can tell which scheme panicked / bailed on their
                // data, especially for third-party schemes where the error site may not carry any
                // compressor context.
                trace::scheme_compress_failed(winner.id(), before_nbytes, error_ctx.as_ref(), err);
            })?;

        // The winner was picked on estimates. A trial output is measured, so it wins when smaller.
        let (compressed, applied) = match data.get::<TrialOutput>() {
            Some(trial) if trial.array.nbytes() < compressed.nbytes() => {
                (trial.array.clone(), trial.scheme)
            }
            _ => (compressed, winner.id()),
        };

        let after_nbytes = compressed.nbytes();
        let actual_ratio = (after_nbytes != 0).then(|| before_nbytes as f64 / after_nbytes as f64);

        let accepted = after_nbytes < before_nbytes;

        trace::record_winner_compress_result(
            after_nbytes,
            winner_estimate.trace_ratio(),
            actual_ratio,
            accepted,
        );

        Ok(if accepted {
            observed(
                policy,
                Some(applied),
                &compress_ctx,
                Selection::Compressed(compressed),
            )
        } else {
            observed(policy, None, &compress_ctx, Selection::unchanged(data))
        })
    }

    /// Compresses `data` in full with each candidate scheme, and keeps the output `policy` ranks
    /// lowest, or the canonical array if it ranks lower still.
    ///
    /// The candidates are the eligible schemes in `ids`, or every eligible scheme when `ids` is
    /// empty.
    #[expect(
        clippy::too_many_arguments,
        reason = "internal helper threading selection state"
    )]
    fn compress_best(
        &self,
        policy: &dyn CompressionPolicy,
        eligible_schemes: &[&'static dyn Scheme],
        ids: &[SchemeId],
        data: ArrayAndStats,
        compress_ctx: &CompressorContext,
        before_nbytes: u64,
        exec_ctx: &mut ExecutionCtx,
    ) -> Selection {
        let mut best: Option<(f64, SchemeId, ArrayRef)> = None;
        for &scheme in eligible_schemes {
            if !ids.is_empty() && !ids.contains(&scheme.id()) {
                continue;
            }
            let Some(compressed) =
                self.try_scheme(scheme, &data, compress_ctx, before_nbytes, exec_ctx)
            else {
                continue;
            };
            policy.observe_candidate(&SiteOutcome {
                scheme: Some(scheme.id()),
                array: &compressed,
                ctx: compress_ctx,
            });
            let rank = policy.rank(&compressed, compress_ctx);
            if best
                .as_ref()
                .is_none_or(|(best_rank, ..)| rank < *best_rank)
            {
                best = Some((rank, scheme.id(), compressed));
            }
        }
        match best {
            Some((rank, id, compressed)) if rank < policy.rank(data.array(), compress_ctx) => {
                observed(
                    Some(policy),
                    Some(id),
                    compress_ctx,
                    Selection::Compressed(compressed),
                )
            }
            _ => observed(Some(policy), None, compress_ctx, Selection::unchanged(data)),
        }
    }

    /// Compresses `data` with `scheme` on a policy's request, returning `None` if the scheme fails
    /// or does not make the array smaller than `before_nbytes`.
    fn try_scheme(
        &self,
        scheme: &dyn Scheme,
        data: &ArrayAndStats,
        compress_ctx: &CompressorContext,
        before_nbytes: u64,
        exec_ctx: &mut ExecutionCtx,
    ) -> Option<ArrayRef> {
        match self.compress_with_scheme(scheme, data, compress_ctx, exec_ctx) {
            Ok(compressed) => (compressed.nbytes() < before_nbytes).then_some(compressed),
            Err(err) => {
                trace::candidate_failed(scheme.id(), &err);
                None
            }
        }
    }
}

/// Reports the scheme a site applied to `policy`, if any, and hands `selection` back.
fn observed(
    policy: Option<&dyn CompressionPolicy>,
    scheme: Option<SchemeId>,
    ctx: &CompressorContext,
    selection: Selection,
) -> Selection {
    if let Some(policy) = policy {
        let array = match &selection {
            Selection::Compressed(array) | Selection::Unchanged(array, _) => array,
        };
        policy.observe(&SiteOutcome { scheme, array, ctx });
    }
    selection
}

impl CascadingCompressor {
    /// Compresses `data` with `scheme`: runs its [`encode`](Scheme::encode) and compresses the
    /// children it registers, or calls its [`compress`](Scheme::compress) when `encode` falls back
    /// to it. An estimate that already stored the scheme's complete output as a [`TrialOutput`]
    /// is reused.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding or compressing a child fails.
    pub(crate) fn compress_with_scheme(
        &self,
        scheme: &dyn Scheme,
        data: &ArrayAndStats,
        compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        if let Some(trial) = data.get::<TrialOutput>()
            && trial.scheme == scheme.id()
        {
            return Ok(trial.array.clone());
        }
        let encoded = scheme.encode(data, compress_ctx, exec_ctx)?;
        if encoded.is_uses_compress() {
            // The scheme compresses its children, and any validity it stores, itself.
            return scheme.compress(self, data, compress_ctx.clone(), exec_ctx);
        }
        self.compress_encoded(scheme.id(), encoded, compress_ctx, exec_ctx)
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
