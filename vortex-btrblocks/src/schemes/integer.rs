// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integer compression schemes.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Patched;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::patched::use_experimental_patches;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::scalar::Scalar;
use vortex_compressor::builtins::FloatDictScheme;
use vortex_compressor::builtins::StringDictScheme;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::DeferredEstimate;
use vortex_compressor::estimate::EstimateScore;
use vortex_compressor::estimate::EstimateVerdict;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::DescendantExclusion;
#[cfg(feature = "unstable_encodings")]
use vortex_compressor::scheme::SchemeId;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_fastlanes::BitPacked;
#[cfg(feature = "unstable_encodings")]
use vortex_fastlanes::Delta;
use vortex_fastlanes::FoR;
use vortex_fastlanes::FoRArrayExt;
use vortex_fastlanes::RLE;
use vortex_fastlanes::RLEArrayExt;
use vortex_fastlanes::bitpack_compress::bit_width_histogram;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_fastlanes::bitpack_compress::find_best_bit_width;
use vortex_runend::RunEnd;
use vortex_runend::compress::runend_encode;
use vortex_sequence::sequence_encode;
use vortex_sparse::Sparse;
use vortex_sparse::SparseExt as _;
use vortex_zigzag::ZigZag;
use vortex_zigzag::ZigZagArrayExt;
use vortex_zigzag::zigzag_encode;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::GenerateStatsOptions;
use crate::Scheme;
use crate::SchemeExt;
use crate::compress_patches;
use crate::schemes::rle_ancestor_exclusions;
use crate::schemes::rle_descendant_exclusions;

/// Frame of Reference encoding.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FoRScheme;

/// ZigZag encoding for negative integers.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ZigZagScheme;

/// BitPacking encoding for non-negative integers.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BitPackingScheme;

/// Sparse encoding for single-value-dominated arrays.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SparseScheme;

/// Run-end encoding with end positions.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RunEndScheme;

/// Sequence encoding for sequential patterns.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SequenceScheme;

/// Pco (pcodec) compression for integers.
#[cfg(feature = "pco")]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct PcoScheme;

// Re-export builtin schemes from vortex-compressor.
pub use vortex_compressor::builtins::IntConstantScheme;
pub use vortex_compressor::builtins::IntDictScheme;
pub use vortex_compressor::builtins::is_integer_primitive;
pub use vortex_compressor::stats::IntegerStats;

/// RLE scheme for integer arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntRLEScheme;

/// Threshold for the average run length in an array before we consider run-length encoding.
pub(crate) const RUN_LENGTH_THRESHOLD: u32 = 4;

/// Threshold for the average run length in an array before we consider run-end encoding.
const RUN_END_THRESHOLD: u32 = 4;

impl Scheme for FoRScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.for"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_integer_primitive(canonical)
    }

    /// Dict codes always start at 0, so FoR (which subtracts the min) is a no-op.
    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        vec![
            AncestorExclusion {
                ancestor: IntDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: FloatDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: StringDictScheme.id(),
                children: ChildSelection::One(1),
            },
        ]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // FoR only subtracts the min. Without further compression (e.g. BitPacking), the output is
        // the same size.
        if compress_ctx.finished_cascading() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        let stats = data.integer_stats(exec_ctx);

        // Only apply when the min is not already zero.
        if stats.erased().min_is_zero() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // Difference between max and min.
        let for_bitwidth = match stats.erased().max_minus_min().checked_ilog2() {
            Some(l) => l + 1,
            // If max-min == 0, the we should be compressing this as a constant array.
            None => return CompressionEstimate::Verdict(EstimateVerdict::Skip),
        };

        // If BitPacking can be applied (only non-negative values) and FoR doesn't reduce bit width
        // compared to BitPacking, don't use FoR since it has a small amount of overhead (storing
        // the reference) for effectively no benefits.
        if let Some(max_log) = stats
            .erased()
            .max_ilog2()
            // Only skip FoR when min >= 0, otherwise BitPacking can't be applied without ZigZag.
            .filter(|_| !stats.erased().min_is_negative())
        {
            let bitpack_bitwidth = max_log + 1;
            if for_bitwidth >= bitpack_bitwidth {
                return CompressionEstimate::Verdict(EstimateVerdict::Skip);
            }
        }

        let full_width: u32 = data
            .array_as_primitive()
            .ptype()
            .bit_width()
            .try_into()
            .vortex_expect("bit width must fit in u32");

        CompressionEstimate::Verdict(EstimateVerdict::Ratio(
            full_width as f64 / for_bitwidth as f64,
        ))
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let primitive = data.array().clone().execute::<PrimitiveArray>(exec_ctx)?;
        let for_array = FoR::encode(primitive)?;
        let biased = for_array
            .encoded()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?;

        // Immediately bitpack. If any other scheme was preferable, it would be chosen instead
        // of bitpacking.
        // NOTE: we could delegate in the future if we had another downstream codec that performs
        //  as well.
        let leaf_ctx = compress_ctx.clone().as_leaf();
        let biased_data =
            ArrayAndStats::new(biased.into_array(), compress_ctx.merged_stats_options());
        let compressed = BitPackingScheme.compress(compressor, &biased_data, leaf_ctx, exec_ctx)?;

        // TODO(connor): This should really be `new_unchecked`.
        let for_compressed = FoR::try_new(compressed, for_array.reference_scalar().clone())?;
        for_compressed
            .as_ref()
            .statistics()
            .inherit_from(for_array.as_ref().statistics());

        Ok(for_compressed.into_array())
    }
}

impl Scheme for ZigZagScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.zigzag"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_integer_primitive(canonical)
    }

    /// Children: encoded=0.
    fn num_children(&self) -> usize {
        1
    }

    /// ZigZag is a bijective value transform that preserves cardinality, run patterns, and value
    /// dominance. If Dict, RunEnd, or Sparse lost on the original array, they will lose on ZigZag's
    /// output too, so we skip evaluating them.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![
            DescendantExclusion {
                excluded: IntDictScheme.id(),
                children: ChildSelection::All,
            },
            DescendantExclusion {
                excluded: RunEndScheme.id(),
                children: ChildSelection::All,
            },
            DescendantExclusion {
                excluded: SparseScheme.id(),
                children: ChildSelection::All,
            },
        ]
    }

    /// Dict codes are unsigned integers (0..cardinality). ZigZag only helps negatives.
    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        vec![
            AncestorExclusion {
                ancestor: IntDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: FloatDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: StringDictScheme.id(),
                children: ChildSelection::One(1),
            },
        ]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // ZigZag only transforms negative values to positive. Without further compression,
        // the output is the same size.
        if compress_ctx.finished_cascading() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        let stats = data.integer_stats(exec_ctx);

        // ZigZag is only useful when there are negative values.
        if !stats.erased().min_is_negative() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        // Zigzag encode the values, then recursively compress the inner values.
        let zag = zigzag_encode(data.array_as_primitive())?;
        let encoded = zag.encoded().clone().execute::<PrimitiveArray>(exec_ctx)?;

        let compressed = compressor.compress_child(
            &encoded.into_array(),
            &compress_ctx,
            self.id(),
            0,
            exec_ctx,
        )?;

        Ok(ZigZag::try_new(compressed)?.into_array())
    }
}

impl Scheme for BitPackingScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.bitpacking"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_integer_primitive(canonical)
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let stats = data.integer_stats(exec_ctx);

        // BitPacking only works for non-negative values.
        if stats.erased().min_is_negative() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let primitive_array = data.array_as_primitive();

        let histogram = bit_width_histogram(primitive_array, exec_ctx)?;
        let bw = find_best_bit_width(primitive_array.ptype(), &histogram)?;

        // If best bw is determined to be the current bit-width, return the original array.
        if bw as usize == primitive_array.ptype().bit_width() {
            return Ok(primitive_array.array().clone());
        }

        // Otherwise we can bitpack the array.
        let primitive_array = primitive_array.into_owned();
        let packed = bitpack_encode(&primitive_array, bw, Some(&histogram), exec_ctx)?;

        let packed_stats = packed.statistics().to_owned();
        let ptype = packed.dtype().as_ptype();
        let mut parts = BitPacked::into_parts(packed);

        let array = if use_experimental_patches() {
            let patches = parts.patches.take();
            // Transpose patches into G-ALP style PatchedArray, wrapping an inner BitPackedArray.
            let array = BitPacked::try_new(
                parts.packed,
                ptype,
                parts.validity,
                None,
                parts.bit_width,
                parts.len,
                parts.offset,
            )?
            .into_array();

            match patches {
                None => array,
                Some(p) => Patched::from_array_and_patches(array, &p, exec_ctx)?
                    .with_stats_set(packed_stats)
                    .into_array(),
            }
        } else {
            // Compress patches and place back into BitPackedArray.
            let patches = parts
                .patches
                .take()
                .map(|p| compress_patches(p, exec_ctx))
                .transpose()?;
            parts.patches = patches;
            BitPacked::try_new(
                parts.packed,
                ptype,
                parts.validity,
                parts.patches,
                parts.bit_width,
                parts.len,
                parts.offset,
            )?
            .with_stats_set(packed_stats)
            .into_array()
        };

        Ok(array)
    }
}

impl Scheme for SparseScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.sparse"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_integer_primitive(canonical)
    }

    fn stats_options(&self) -> GenerateStatsOptions {
        GenerateStatsOptions {
            count_distinct_values: true,
        }
    }

    /// Children: values=0, indices=1.
    fn num_children(&self) -> usize {
        2
    }

    /// Sparse indices (child 1) are monotonically increasing positions with all unique values.
    /// Dict, RunEnd, RLE, and Sparse are all pointless on such data.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![
            DescendantExclusion {
                excluded: IntDictScheme.id(),
                children: ChildSelection::One(1),
            },
            DescendantExclusion {
                excluded: RunEndScheme.id(),
                children: ChildSelection::One(1),
            },
            DescendantExclusion {
                excluded: IntRLEScheme.id(),
                children: ChildSelection::One(1),
            },
            DescendantExclusion {
                excluded: SparseScheme.id(),
                children: ChildSelection::One(1),
            },
        ]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let len = data.array_len() as f64;
        let stats = data.integer_stats(exec_ctx);
        let value_count = stats.value_count();

        // All-null arrays should be compressed as constant instead anyways.
        if value_count == 0 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // If the majority (90%) of values is null, this will compress well.
        if stats.null_count() as f64 / len > 0.9 {
            return CompressionEstimate::Verdict(EstimateVerdict::Ratio(len / value_count as f64));
        }

        let (_, most_frequent_count) = stats
            .erased()
            .most_frequent_value_and_count()
            .vortex_expect(
                "this must be present since `SparseScheme` declared that we need distinct values",
            );

        // If the most frequent value is the only value, we should compress as constant instead.
        if most_frequent_count == value_count {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        debug_assert!(value_count > most_frequent_count);

        // See if the most frequent value accounts for >= 90% of the set values.
        let freq = most_frequent_count as f64 / value_count as f64;
        if freq < 0.9 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // We only store the positions of the non-top values.
        CompressionEstimate::Verdict(EstimateVerdict::Ratio(
            value_count as f64 / (value_count - most_frequent_count) as f64,
        ))
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let len = data.array_len();
        let stats = data.integer_stats(exec_ctx);
        let array = data.array();

        let (most_frequent_value, most_frequent_count) = stats
            .erased()
            .most_frequent_value_and_count()
            .vortex_expect(
                "this must be present since `SparseScheme` declared that we need distinct values",
            );

        if most_frequent_count as usize == len {
            // If the most frequent value is the only value, we should compress as constant instead.
            return Ok(ConstantArray::new(
                Scalar::primitive_value(
                    most_frequent_value,
                    most_frequent_value.ptype(),
                    array.dtype().nullability(),
                ),
                len,
            )
            .into_array());
        }

        let sparse_encoded = Sparse::encode(
            array,
            Some(Scalar::primitive_value(
                most_frequent_value,
                most_frequent_value.ptype(),
                array.dtype().nullability(),
            )),
            exec_ctx,
        )?;

        if let Some(sparse) = sparse_encoded.as_opt::<Sparse>() {
            let sparse_values_primitive = sparse
                .patches()
                .values()
                .clone()
                .execute::<PrimitiveArray>(exec_ctx)?;
            let compressed_values = compressor.compress_child(
                &sparse_values_primitive.into_array(),
                &compress_ctx,
                self.id(),
                0,
                exec_ctx,
            )?;

            let indices = sparse
                .patches()
                .indices()
                .clone()
                .execute::<PrimitiveArray>(exec_ctx)?
                .narrow(exec_ctx)?;

            let compressed_indices = compressor.compress_child(
                &indices.into_array(),
                &compress_ctx,
                self.id(),
                1,
                exec_ctx,
            )?;

            Sparse::try_new(
                compressed_indices,
                compressed_values,
                sparse.len(),
                sparse.fill_scalar().clone(),
            )
            .map(|a| a.into_array())
        } else {
            Ok(sparse_encoded)
        }
    }
}

impl Scheme for RunEndScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.runend"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_integer_primitive(canonical)
    }

    /// Children: values=0, ends=1.
    fn num_children(&self) -> usize {
        2
    }

    /// RunEnd ends (child 1) are monotonically increasing positions with all unique values.
    /// Dict, RunEnd, RLE, and Sparse are all pointless on such data.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![
            DescendantExclusion {
                excluded: IntDictScheme.id(),
                children: ChildSelection::One(1),
            },
            DescendantExclusion {
                excluded: RunEndScheme.id(),
                children: ChildSelection::One(1),
            },
            DescendantExclusion {
                excluded: IntRLEScheme.id(),
                children: ChildSelection::One(1),
            },
            DescendantExclusion {
                excluded: SparseScheme.id(),
                children: ChildSelection::One(1),
            },
        ]
    }

    /// Dict values (child 0) are all unique by definition, so run-end encoding them is
    /// pointless. Codes (child 1) can have runs and may benefit from RunEnd.
    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        vec![
            AncestorExclusion {
                ancestor: IntDictScheme.id(),
                children: ChildSelection::One(0),
            },
            AncestorExclusion {
                ancestor: FloatDictScheme.id(),
                children: ChildSelection::One(0),
            },
            AncestorExclusion {
                ancestor: StringDictScheme.id(),
                children: ChildSelection::One(0),
            },
        ]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // If the run length is below the threshold, drop it.
        if data.integer_stats(exec_ctx).average_run_length() < RUN_END_THRESHOLD {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        // Run-end encode the ends.
        let (ends, values) = runend_encode(data.array_as_primitive(), exec_ctx);

        let values_primitive = values.execute::<PrimitiveArray>(exec_ctx)?;
        let compressed_values = compressor.compress_child(
            &values_primitive.into_array(),
            &compress_ctx,
            self.id(),
            0,
            exec_ctx,
        )?;

        let compressed_ends =
            compressor.compress_child(&ends.into_array(), &compress_ctx, self.id(), 1, exec_ctx)?;

        // SAFETY: compression doesn't affect invariants.
        Ok(unsafe {
            RunEnd::new_unchecked(compressed_ends, compressed_values, 0, data.array_len())
                .into_array()
        })
    }
}

impl Scheme for SequenceScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.sequence"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_integer_primitive(canonical)
    }

    /// Sequence encoding on dictionary codes just adds a layer of indirection without compressing
    /// the data. Dict codes are compact integers that benefit from BitPacking or FoR, not from
    /// sequence detection.
    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        vec![
            AncestorExclusion {
                ancestor: IntDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: FloatDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: StringDictScheme.id(),
                children: ChildSelection::One(1),
            },
        ]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // It is pointless checking if a sample is a sequence since it will not correspond to the
        // entire array.
        if compress_ctx.is_sample() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        let stats = data.integer_stats(exec_ctx);

        // `SequenceArray` does not support nulls.
        if stats.null_count() > 0 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // If the distinct_values_count was computed, and not all values are unique, then this
        // cannot be encoded as a sequence array.
        if stats
            .distinct_count()
            .is_some_and(|count| count as usize != data.array_len())
        {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // TODO(connor): `sequence_encode` allocates the encoded array just to confirm feasibility.
        // A cheaper `is_sequence` probe would let us skip the allocation entirely.
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            |_compressor, data, best_so_far, _ctx, exec_ctx| {
                // `SequenceArray` stores exactly two scalars (base and multiplier), so the best
                // achievable compression ratio is `array_len / 2`.
                let compressed_size = 2usize;
                let max_ratio = data.array_len() as f64 / compressed_size as f64;

                // If we cannot beat the best so far, then we do not want to even try sequence
                // encoding the data.
                let threshold = best_so_far.and_then(EstimateScore::finite_ratio);
                if threshold.is_some_and(|t| max_ratio <= t) {
                    return Ok(EstimateVerdict::Skip);
                }

                // TODO(connor): We should pass this array back to the compressor in the case that
                // we do want to sequence encode this so that we do not need to recompress.
                if sequence_encode(data.array_as_primitive(), exec_ctx)?.is_none() {
                    return Ok(EstimateVerdict::Skip);
                }
                // TODO(connor): Should we get the actual ratio here?
                Ok(EstimateVerdict::Ratio(max_ratio))
            },
        )))
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let stats = data.integer_stats(exec_ctx);

        if stats.null_count() > 0 {
            vortex_bail!("sequence encoding does not support nulls");
        }
        sequence_encode(data.array_as_primitive(), exec_ctx)?
            .ok_or_else(|| vortex_err!("cannot sequence encode array"))
    }
}

#[cfg(feature = "pco")]
impl Scheme for PcoScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.pco"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_integer_primitive(canonical)
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        use vortex_array::dtype::PType;

        // Pco does not support I8 or U8.
        if matches!(data.array_as_primitive().ptype(), PType::I8 | PType::U8) {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        Ok(vortex_pco::Pco::from_primitive(
            data.array_as_primitive(),
            pco::DEFAULT_COMPRESSION_LEVEL,
            8192,
            exec_ctx,
        )?
        .into_array())
    }
}

/// Shared compression logic for RLE schemes.
pub(crate) fn rle_compress(
    scheme: &dyn Scheme,
    compressor: &CascadingCompressor,
    data: &ArrayAndStats,
    compress_ctx: CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let rle_array = RLE::encode(data.array_as_primitive(), exec_ctx)?;

    let rle_values_primitive = rle_array
        .values()
        .clone()
        .execute::<PrimitiveArray>(exec_ctx)?;
    let compressed_values = compressor.compress_child(
        &rle_values_primitive.into_array(),
        &compress_ctx,
        scheme.id(),
        0,
        exec_ctx,
    )?;

    // Delta is an unstable encoding, once we deem it stable we can switch over to this always.
    #[cfg(feature = "unstable_encodings")]
    let compressed_indices = {
        let rle_indices_primitive = rle_array
            .indices()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?;
        try_compress_delta(
            compressor,
            &rle_indices_primitive.into_array(),
            &compress_ctx,
            scheme.id(),
            1,
            exec_ctx,
        )?
    };

    #[cfg(not(feature = "unstable_encodings"))]
    let compressed_indices = {
        let rle_indices_primitive = rle_array
            .indices()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?;
        compressor.compress_child(
            &rle_indices_primitive.into_array(),
            &compress_ctx,
            scheme.id(),
            1,
            exec_ctx,
        )?
    };

    let rle_offsets_primitive = rle_array
        .values_idx_offsets()
        .clone()
        .execute::<PrimitiveArray>(exec_ctx)?
        .narrow(exec_ctx)?;
    let compressed_offsets = compressor.compress_child(
        &rle_offsets_primitive.into_array(),
        &compress_ctx,
        scheme.id(),
        2,
        exec_ctx,
    )?;

    // SAFETY: Recursive compression doesn't affect the invariants.
    unsafe {
        Ok(RLE::new_unchecked(
            compressed_values,
            compressed_indices,
            compressed_offsets,
            rle_array.offset(),
            rle_array.len(),
        )
        .into_array())
    }
}

#[cfg(feature = "unstable_encodings")]
fn try_compress_delta(
    compressor: &CascadingCompressor,
    child: &ArrayRef,
    parent_ctx: &CompressorContext,
    parent_id: SchemeId,
    child_index: usize,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let child_primitive = child.clone().execute::<PrimitiveArray>(exec_ctx)?;
    let (bases, deltas) = vortex_fastlanes::delta_compress(&child_primitive, exec_ctx)?;

    let compressed_bases = compressor.compress_child(
        &bases.into_array(),
        parent_ctx,
        parent_id,
        child_index,
        exec_ctx,
    )?;
    let compressed_deltas = compressor.compress_child(
        &deltas.into_array(),
        parent_ctx,
        parent_id,
        child_index,
        exec_ctx,
    )?;

    Delta::try_new(compressed_bases, compressed_deltas, 0, child.len()).map(IntoArray::into_array)
}

impl Scheme for IntRLEScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.rle"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_integer_primitive(canonical)
    }

    /// Children: values=0, indices=1, offsets=2.
    fn num_children(&self) -> usize {
        3
    }

    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        rle_descendant_exclusions()
    }

    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        rle_ancestor_exclusions()
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // RLE is only useful when we cascade it with another encoding.
        if compress_ctx.finished_cascading() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        if data.integer_stats(exec_ctx).average_run_length() < RUN_LENGTH_THRESHOLD {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        rle_compress(self, compressor, data, compress_ctx, exec_ctx)
    }
}

#[cfg(test)]
mod tests {
    use std::iter;
    use std::sync::LazyLock;

    use itertools::Itertools;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Constant;
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::Masked;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::BufferMut;
    use vortex_buffer::buffer;
    use vortex_compressor::CascadingCompressor;
    use vortex_error::VortexResult;
    use vortex_fastlanes::RLE;
    use vortex_sequence::Sequence;
    use vortex_session::VortexSession;

    use crate::BtrBlocksCompressor;
    use crate::schemes::integer::IntRLEScheme;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[test]
    fn test_empty() -> VortexResult<()> {
        // Make sure empty array compression does not fail.
        let btr = BtrBlocksCompressor::default();
        let array = PrimitiveArray::new(Buffer::<i32>::empty(), Validity::NonNullable);
        let result = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;

        assert!(result.is_empty());
        Ok(())
    }

    #[test]
    fn test_dict_encodable() -> VortexResult<()> {
        let mut codes = BufferMut::<i32>::with_capacity(65_535);
        // Write some runs of length 3 of a handful of different values. Interrupted by some
        // one-off values.

        let numbers = [0, 10, 50, 100, 1000, 3000]
            .into_iter()
            .map(|i| 12340 * i) // must be big enough to not prefer fastlanes.bitpacked
            .collect_vec();

        let mut rng = StdRng::seed_from_u64(1u64);
        while codes.len() < 64000 {
            let run_length = rng.next_u32() % 5;
            let value = numbers[rng.next_u32() as usize % numbers.len()];
            for _ in 0..run_length {
                codes.push(value);
            }
        }

        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(
            &codes.freeze().into_array(),
            &mut SESSION.create_execution_ctx(),
        )?;
        assert!(compressed.is::<Dict>());
        Ok(())
    }

    #[test]
    fn constant_mostly_nulls() -> VortexResult<()> {
        let array = PrimitiveArray::new(
            buffer![189u8, 189, 189, 189, 189, 189, 189, 189, 189, 0, 46],
            Validity::from_iter(vec![
                false, false, false, false, false, false, false, false, false, false, true,
            ]),
        );
        let validity = array.validity()?;

        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;

        assert!(compressed.is::<Masked>());
        assert!(compressed.children()[0].is::<Constant>());

        let decoded = compressed;
        let expected =
            PrimitiveArray::new(buffer![0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46], validity).into_array();
        assert_arrays_eq!(decoded, expected);
        Ok(())
    }

    #[test]
    fn nullable_sequence() -> VortexResult<()> {
        let values = (0i32..20).step_by(7).collect_vec();
        let array = PrimitiveArray::from_option_iter(values.clone().into_iter().map(Some));

        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
        assert!(compressed.is::<Sequence>());

        let decoded = compressed;
        let expected = PrimitiveArray::from_option_iter(values.into_iter().map(Some)).into_array();
        assert_arrays_eq!(decoded, expected);
        Ok(())
    }

    #[test]
    fn test_rle_compression() -> VortexResult<()> {
        let mut values = Vec::new();
        values.extend(iter::repeat_n(42i32, 100));
        values.extend(iter::repeat_n(123i32, 200));
        values.extend(iter::repeat_n(987i32, 150));

        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let compressor = CascadingCompressor::new(vec![&IntRLEScheme]);
        let compressed =
            compressor.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
        assert!(compressed.is::<RLE>());

        let expected = Buffer::copy_from(&values).into_array();
        assert_arrays_eq!(compressed, expected);
        Ok(())
    }

    #[test_with::env(CI)]
    #[test_with::no_env(VORTEX_SKIP_SLOW_TESTS)]
    fn compress_large_int() -> VortexResult<()> {
        const NUM_LISTS: usize = 10_000;
        const ELEMENTS_PER_LIST: usize = 5_000;

        let prim = (0..NUM_LISTS)
            .flat_map(|list_idx| {
                (0..ELEMENTS_PER_LIST).map(move |elem_idx| (list_idx * 1000 + elem_idx) as f64)
            })
            .collect::<PrimitiveArray>()
            .into_array();

        let btr = BtrBlocksCompressor::default();
        btr.compress(&prim, &mut SESSION.create_execution_ctx())?;

        Ok(())
    }
}

/// Tests to verify that each integer compression scheme produces the expected encoding.
#[cfg(test)]
mod scheme_selection_tests {
    use std::iter;
    use std::sync::LazyLock;

    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Constant;
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::expr::stats::Precision;
    use vortex_array::expr::stats::Stat;
    use vortex_array::expr::stats::StatsProviderExt;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_fastlanes::BitPacked;
    use vortex_fastlanes::FoR;
    use vortex_runend::RunEnd;
    use vortex_sequence::Sequence;
    use vortex_session::VortexSession;
    use vortex_sparse::Sparse;

    use crate::BtrBlocksCompressor;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[test]
    fn test_constant_compressed() -> VortexResult<()> {
        let values: Vec<i32> = iter::repeat_n(42, 100).collect();
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
        assert!(compressed.is::<Constant>());
        Ok(())
    }

    #[test]
    fn test_for_compressed() -> VortexResult<()> {
        let values: Vec<i32> = (0..1000).map(|i| 1_000_000 + ((i * 37) % 100)).collect();
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
        assert!(compressed.is::<FoR>());
        Ok(())
    }

    #[test]
    fn test_bitpacking_compressed() -> VortexResult<()> {
        let values: Vec<u32> = (0..1000).map(|i| i % 16).collect();
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
        assert!(compressed.is::<BitPacked>());
        assert_eq!(
            compressed.statistics().get_as::<u64>(Stat::NullCount),
            Precision::exact(0u64)
        );
        assert_eq!(
            compressed.statistics().get_as::<u32>(Stat::Min),
            Precision::exact(0u32)
        );
        assert_eq!(
            compressed.statistics().get_as::<u32>(Stat::Max),
            Precision::exact(15u32)
        );
        Ok(())
    }

    #[test]
    fn test_sparse_compressed() -> VortexResult<()> {
        let mut values: Vec<i32> = Vec::new();
        for i in 0..1000 {
            if i % 20 == 0 {
                values.push(2_000_000 + (i * 7) % 1000);
            } else {
                values.push(1_000_000);
            }
        }
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
        assert!(compressed.is::<Sparse>());
        Ok(())
    }

    #[test]
    fn test_dict_compressed() -> VortexResult<()> {
        let mut codes = Vec::with_capacity(65_535);
        let numbers: Vec<i32> = [0, 10, 50, 100, 1000, 3000]
            .into_iter()
            .map(|i| 12340 * i) // must be big enough to not prefer fastlanes.bitpacked
            .collect();

        let mut rng = StdRng::seed_from_u64(1u64);
        while codes.len() < 64000 {
            let run_length = rng.next_u32() % 5;
            let value = numbers[rng.next_u32() as usize % numbers.len()];
            for _ in 0..run_length {
                codes.push(value);
            }
        }

        let array = PrimitiveArray::new(Buffer::copy_from(&codes), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
        assert!(compressed.is::<Dict>());
        Ok(())
    }

    #[test]
    fn test_runend_compressed() -> VortexResult<()> {
        let mut values: Vec<i32> = Vec::new();
        for i in 0..100 {
            values.extend(iter::repeat_n((i32::MAX - 50).wrapping_add(i), 10));
        }
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
        assert!(compressed.is::<RunEnd>());
        Ok(())
    }

    #[test]
    fn test_sequence_compressed() -> VortexResult<()> {
        let values: Vec<i32> = (0..1000).map(|i| i * 7).collect();
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
        assert!(compressed.is::<Sequence>());
        Ok(())
    }

    #[test]
    fn test_rle_compressed() -> VortexResult<()> {
        let mut values: Vec<i32> = Vec::new();
        for i in 0..1024 {
            values.extend(iter::repeat_n(i, 10));
        }
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
        eprintln!("{}", compressed.display_tree());
        assert!(compressed.is::<RunEnd>());
        Ok(())
    }
}
