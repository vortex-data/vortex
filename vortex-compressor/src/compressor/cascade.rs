// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Core cascading compression flow.

use vortex_array::ArrayId;
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
use vortex_utils::aliases::hash_set::HashSet;

use super::CascadingCompressor;
use super::constant;
use super::select::WinnerEstimate;
use crate::scheme::CompressorContext;
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
    /// Returns the array, compressed or not.
    fn into_array(self) -> ArrayRef {
        match self {
            Selection::Compressed(array) | Selection::Unchanged(array) => array,
        }
    }
}

/// The outcome of [`CascadingCompressor::choose`].
enum Choice {
    /// The compressor decided the result without a scheme.
    Decided(Selection),
    /// A scheme won the selection, but it did not compress the array yet.
    Scheme(Winner),
}

/// A scheme that won the selection, with the state it needs to compress the array.
struct Winner {
    /// The scheme that won.
    scheme: &'static dyn Scheme,
    /// The estimate with which the scheme won.
    estimate: WinnerEstimate,
    /// The array to compress, with its statistics.
    data: ArrayAndStats,
    /// The context with the merged statistics options of the eligible schemes.
    compress_ctx: CompressorContext,
    /// The bytes the array's rows use, which the compressed output must beat.
    before_nbytes: u64,
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
        match self.choose(canonical, compress_ctx, exec_ctx)? {
            Choice::Decided(selection) => Ok(selection),
            Choice::Scheme(winner) => self.compress_with_winner(winner, exec_ctx),
        }
    }

    /// Returns the encodings that the array [`compress`](Self::compress) returns for `array` can
    /// have.
    ///
    /// For a leaf array, this runs the same scheme selection as `compress`, with the statistics
    /// and the sample compressions that the selection needs. If the winning scheme declares its
    /// [`returned_encodings`](Scheme::returned_encodings), the full compression does not occur.
    /// The result is then the encodings of the winner and the encoding of the uncompressed array,
    /// because `compress` returns the uncompressed array if the output of the winner is not
    /// smaller. If the winner does not declare its encodings, or if `array` is not a leaf array,
    /// this compresses `array` and returns the one encoding of the result.
    ///
    /// # Errors
    ///
    /// Returns an error if canonicalization or compression fails.
    pub fn encodings_for(
        &self,
        array: &ArrayRef,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<HashSet<ArrayId>> {
        let canonical = array.clone().execute::<CanonicalValidity>(exec_ctx)?.0;
        let compact = canonical.compact(exec_ctx)?;

        let is_leaf = matches!(
            compact,
            Canonical::Bool(_)
                | Canonical::Primitive(_)
                | Canonical::Decimal(_)
                | Canonical::VarBinView(_)
        );

        if !is_leaf {
            let compressed = self.compress_canonical(compact, CompressorContext::new(), exec_ctx)?;
            return Ok([compressed.encoding_id()].into_iter().collect());
        }

        let winner = match self.choose(compact, CompressorContext::new(), exec_ctx)? {
            Choice::Decided(selection) => {
                return Ok([selection.into_array().encoding_id()].into_iter().collect());
            }
            Choice::Scheme(winner) => winner,
        };

        let declared = winner.scheme.returned_encodings();
        if declared.is_empty() {
            let compressed = self.compress_with_winner(winner, exec_ctx)?.into_array();
            return Ok([compressed.encoding_id()].into_iter().collect());
        }

        let mut encodings: HashSet<ArrayId> = declared.into_iter().collect();
        encodings.insert(winner.data.array().encoding_id());

        Ok(encodings)
    }

    /// Chooses how to compress a single leaf array, but does not run the winning scheme.
    ///
    /// The compressor decides empty, all-null and constant arrays itself, and arrays for which no
    /// scheme beats the canonical encoding.
    fn choose(
        &self,
        canonical: Canonical,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Choice> {
        let eligible_schemes: Vec<&'static dyn Scheme> = self
            .schemes
            .iter()
            .copied()
            .filter(|s| s.matches(&canonical) && !self.is_excluded(*s, &compress_ctx))
            .collect();

        let array: ArrayRef = canonical.into();

        if array.is_empty() {
            return Ok(Choice::Decided(Selection::Unchanged(array)));
        }

        if array.all_invalid(exec_ctx)? {
            return Ok(Choice::Decided(Selection::Compressed(
                ConstantArray::new(Scalar::null(array.dtype().clone()), array.len()).into_array(),
            )));
        }

        // Constant detection is built into the compressor: a constant leaf always short-circuits
        // scheme selection. Samples are exempt because a constant sample does not imply that the
        // full array is constant.
        //
        // Compare valid values directly so even nullable constants avoid distinct-value counting.
        if !compress_ctx.is_sample() && constant::is_constant_for_compression(&array, exec_ctx)? {
            return constant::compress_as_constant(array, exec_ctx)
                .map(|constant| Choice::Decided(Selection::Compressed(constant)));
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

        if eligible_schemes.is_empty() {
            return Ok(Choice::Decided(Selection::Unchanged(data.into_array())));
        }

        let Some((scheme, estimate)) =
            self.choose_best_scheme(&eligible_schemes, &data, compress_ctx.clone(), exec_ctx)?
        else {
            return Ok(Choice::Decided(Selection::Unchanged(data.into_array())));
        };

        Ok(Choice::Scheme(Winner {
            scheme,
            estimate,
            data,
            compress_ctx,
            before_nbytes,
        }))
    }

    /// Runs the winning scheme's `compress`, and keeps its output only if it is smaller than the
    /// input.
    fn compress_with_winner(
        &self,
        winner: Winner,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Selection> {
        let Winner {
            scheme,
            estimate,
            data,
            compress_ctx,
            before_nbytes,
        } = winner;

        // On failure, emit an ERROR event carrying the scheme name and cascade history before
        // propagating.
        let error_ctx = trace::enabled_error_context(&compress_ctx);
        let _winner_span = trace::winner_compress_span(scheme.id(), before_nbytes).entered();
        let compressed = scheme
            .compress(self, &data, compress_ctx, exec_ctx)
            .inspect_err(|err| {
                // NB: this is the only way we can tell which scheme panicked / bailed on their
                // data, especially for third-party schemes where the error site may not carry any
                // compressor context.
                trace::scheme_compress_failed(scheme.id(), before_nbytes, error_ctx.as_ref(), err);
            })?;

        debug_assert!(
            {
                let declared = scheme.returned_encodings();
                declared.is_empty() || declared.contains(&compressed.encoding_id())
            },
            "scheme {} returned encoding {}, which is not in its returned_encodings",
            scheme.id(),
            compressed.encoding_id(),
        );

        let after_nbytes = compressed.nbytes();
        let actual_ratio = (after_nbytes != 0).then(|| before_nbytes as f64 / after_nbytes as f64);

        let accepted = after_nbytes < before_nbytes;

        trace::record_winner_compress_result(
            after_nbytes,
            estimate.trace_ratio(),
            actual_ratio,
            accepted,
        );

        if accepted {
            Ok(Selection::Compressed(compressed))
        } else {
            Ok(Selection::Unchanged(data.into_array()))
        }
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
