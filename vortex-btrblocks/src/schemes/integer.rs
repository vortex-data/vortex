// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integer compression schemes.

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::ConstantArray;
use vortex_array::scalar::Scalar;
use vortex_compressor::builtins::FloatDictScheme;
use vortex_compressor::builtins::StringDictScheme;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_fastlanes::FoR;
use vortex_fastlanes::bitpack_compress::bit_width_histogram;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_fastlanes::bitpack_compress::find_best_bit_width;
use vortex_runend::RunEnd;
use vortex_runend::compress::runend_encode;
use vortex_sequence::sequence_encode;
use vortex_sparse::Sparse;
use vortex_zigzag::ZigZag;
use vortex_zigzag::zigzag_encode;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::GenerateStatsOptions;
use crate::Scheme;
use crate::SchemeExt;
use crate::compress_patches;
use crate::estimate_compression_ratio_with_sampling;

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

pub use crate::schemes::rle::RLE_INTEGER_SCHEME;

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
        _compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<f64> {
        // FoR only subtracts the min. Without further compression (e.g. BitPacking), the output is
        // the same size.
        if ctx.finished_cascading() {
            return Ok(0.0);
        }

        let stats = data.integer_stats();

        // All-null cannot be FOR compressed.
        if stats.value_count() == 0 {
            return Ok(0.0);
        }

        // Only apply when the min is not already zero.
        if stats.erased().min_is_zero() {
            return Ok(0.0);
        }

        // Difference between max and min.
        let for_bitwidth = match stats.erased().max_minus_min().checked_ilog2() {
            Some(l) => l + 1,
            // If max-min == 0, the we should compress as a constant array.
            None => return Ok(0.0),
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
                return Ok(0.0);
            }
        }

        let full_width: u32 = stats
            .source()
            .ptype()
            .bit_width()
            .try_into()
            .vortex_expect("bit width must fit in u32");

        Ok(full_width as f64 / for_bitwidth as f64)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let primitive = data.array().to_primitive();
        let for_array = FoR::encode(primitive)?;
        let biased = for_array.encoded().to_primitive();

        // Immediately bitpack. If any other scheme was preferable, it would be chosen instead
        // of bitpacking.
        // NOTE: we could delegate in the future if we had another downstream codec that performs
        //  as well.
        let leaf_ctx = ctx.clone().as_leaf();
        let mut biased_data = ArrayAndStats::new(biased.into_array(), ctx.stats_options());
        let compressed = BitPackingScheme.compress(compressor, &mut biased_data, leaf_ctx)?;

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
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<f64> {
        // ZigZag only transforms negative values to positive. Without further compression,
        // the output is the same size.
        if ctx.finished_cascading() {
            return Ok(0.0);
        }

        let stats = data.integer_stats();

        // Don't try and compress all-null arrays.
        if stats.value_count() == 0 {
            return Ok(0.0);
        }

        // ZigZag is only useful when there are negative values.
        if !stats.erased().min_is_negative() {
            return Ok(0.0);
        }

        // Run compression on a sample to see how it performs.
        estimate_compression_ratio_with_sampling(self, compressor, data.array(), ctx)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let stats = data.integer_stats();

        // Zigzag encode the values, then recursively compress the inner values.
        let zag = zigzag_encode(stats.source().clone())?;
        let encoded = zag.encoded().to_primitive();

        let compressed = compressor.compress_child(&encoded.into_array(), &ctx, self.id(), 0)?;

        tracing::debug!("zigzag output: {}", compressed.encoding_id());

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
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<f64> {
        let stats = data.integer_stats();

        // BitPacking only works for non-negative values.
        if stats.erased().min_is_negative() {
            return Ok(0.0);
        }

        // Don't compress all-null arrays.
        if stats.value_count() == 0 {
            return Ok(0.0);
        }

        estimate_compression_ratio_with_sampling(self, compressor, data.array(), ctx)
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let stats = data.integer_stats();

        let histogram = bit_width_histogram(stats.source())?;
        let bw = find_best_bit_width(stats.source().ptype(), &histogram)?;
        // If best bw is determined to be the current bit-width, return the original array.
        if bw as usize == stats.source().ptype().bit_width() {
            return Ok(stats.source().clone().into_array());
        }
        let packed = bitpack_encode(stats.source(), bw, Some(&histogram))?;
        let mut packed_data = packed.into_data();

        let patches = packed_data.patches().map(compress_patches).transpose()?;
        packed_data.replace_patches(patches);

        Ok(Array::<vortex_fastlanes::BitPacked>::try_from_data(packed_data)?.into_array())
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
                excluded: RLE_INTEGER_SCHEME.id(),
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
        _compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> VortexResult<f64> {
        let stats = data.integer_stats();

        if stats.value_count() == 0 {
            // All nulls should use ConstantScheme.
            return Ok(0.0);
        }

        // If the majority is null, will compress well.
        if stats.null_count() as f64 / stats.source().len() as f64 > 0.9 {
            return Ok(stats.source().len() as f64 / stats.value_count() as f64);
        }

        // See if the top value accounts for >= 90% of the set values.
        let (_, top_count) = stats
            .erased()
            .most_frequent_value_and_count()
            .vortex_expect(
                "this must be present since `SparseScheme` declared that we need distinct values",
            );

        if top_count == stats.value_count() {
            // top_value is the only value, should use ConstantScheme instead.
            return Ok(0.0);
        }

        let freq = top_count as f64 / stats.value_count() as f64;
        if freq >= 0.9 {
            // We only store the positions of the non-top values.
            return Ok(stats.value_count() as f64 / (stats.value_count() - top_count) as f64);
        }

        Ok(0.0)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let stats = data.integer_stats();

        let (top_pvalue, top_count) = stats
            .erased()
            .most_frequent_value_and_count()
            .vortex_expect(
                "this must be present since `SparseScheme` declared that we need distinct values",
            );
        if top_count as usize == stats.source().len() {
            // top_value is the only value, use ConstantScheme.
            return Ok(ConstantArray::new(
                Scalar::primitive_value(
                    top_pvalue,
                    top_pvalue.ptype(),
                    stats.source().dtype().nullability(),
                ),
                stats.source().len(),
            )
            .into_array());
        }

        let sparse_encoded = Sparse::encode(
            &stats.source().clone().into_array(),
            Some(Scalar::primitive_value(
                top_pvalue,
                top_pvalue.ptype(),
                stats.source().dtype().nullability(),
            )),
        )?;

        if let Some(sparse) = sparse_encoded.as_opt::<Sparse>() {
            let compressed_values = compressor.compress_child(
                &sparse.patches().values().to_primitive().into_array(),
                &ctx,
                self.id(),
                0,
            )?;

            let indices = sparse.patches().indices().to_primitive().narrow()?;

            let compressed_indices =
                compressor.compress_child(&indices.into_array(), &ctx, self.id(), 1)?;

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
                excluded: RLE_INTEGER_SCHEME.id(),
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
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<f64> {
        let stats = data.integer_stats();

        // If the run length is below the threshold, drop it.
        if stats.average_run_length() < RUN_END_THRESHOLD {
            return Ok(0.0);
        }

        // Run compression on a sample, see how it performs.
        estimate_compression_ratio_with_sampling(self, compressor, data.array(), ctx)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let stats = data.integer_stats();

        // Run-end encode the ends.
        let (ends, values) = runend_encode(stats.source().as_view());

        let compressed_values =
            compressor.compress_child(&values.to_primitive().into_array(), &ctx, self.id(), 0)?;

        let compressed_ends = compressor.compress_child(&ends.into_array(), &ctx, self.id(), 1)?;

        // SAFETY: compression doesn't affect invariants.
        unsafe {
            Ok(
                RunEnd::new_unchecked(compressed_ends, compressed_values, 0, stats.source().len())
                    .into_array(),
            )
        }
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
        _compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> VortexResult<f64> {
        let stats = data.integer_stats();

        if stats.null_count() > 0 {
            return Ok(0.0);
        }

        // TODO(connor): Why do we sequence encode the whole thing and then throw it away? And then
        // why do we divide the ratio by 2???

        // If the distinct_values_count was computed, and not all values are unique, then this
        // cannot be encoded as a sequence array.
        if stats
            .distinct_count()
            // TODO(connor): Shouldn't this be `is_none_or`??? Why do things fail if not this?
            .is_some_and(|count| count as usize != stats.source().len())
        {
            return Ok(0.0);
        }

        // TODO(connor): Why divide by 2???
        // Since two values are required to store base and multiplier the compression ratio is
        // divided by 2.
        Ok(sequence_encode(stats.source())?
            .map(|_| stats.source().len() as f64 / 2.0)
            .unwrap_or(0.0))
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let stats = data.integer_stats();

        if stats.null_count() > 0 {
            vortex_bail!("sequence encoding does not support nulls");
        }
        sequence_encode(stats.source())?.ok_or_else(|| vortex_err!("cannot sequence encode array"))
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
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<f64> {
        let stats = data.integer_stats();

        // Pco does not support I8 or U8.
        if matches!(
            stats.source().ptype(),
            vortex_array::dtype::PType::I8 | vortex_array::dtype::PType::U8
        ) {
            return Ok(0.0);
        }

        estimate_compression_ratio_with_sampling(self, compressor, data.array(), ctx)
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let stats = data.integer_stats();

        Ok(
            vortex_pco::Pco::from_primitive(stats.source(), pco::DEFAULT_COMPRESSION_LEVEL, 8192)?
                .into_array(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use itertools::Itertools;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use vortex_array::IntoArray;
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::BufferMut;
    use vortex_buffer::buffer;
    use vortex_compressor::CascadingCompressor;
    use vortex_error::VortexResult;
    use vortex_fastlanes::RLE;
    use vortex_sequence::Sequence;
    use vortex_sparse::Sparse;

    use crate::BtrBlocksCompressor;
    use crate::schemes::rle::RLE_INTEGER_SCHEME;

    #[test]
    fn test_empty() -> VortexResult<()> {
        // Make sure empty array compression does not fail.
        let btr = BtrBlocksCompressor::default();
        let array = PrimitiveArray::new(Buffer::<i32>::empty(), Validity::NonNullable);
        let result = btr.compress(&array.into_array())?;

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
        let compressed = btr.compress(&codes.freeze().into_array())?;
        assert!(compressed.is::<Dict>());
        Ok(())
    }

    #[test]
    fn sparse_mostly_nulls() -> VortexResult<()> {
        let array = PrimitiveArray::new(
            buffer![189u8, 189, 189, 189, 189, 189, 189, 189, 189, 0, 46],
            Validity::from_iter(vec![
                false, false, false, false, false, false, false, false, false, false, true,
            ]),
        );
        let validity = array.validity();

        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array.into_array())?;
        assert!(compressed.is::<Sparse>());

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
        let compressed = btr.compress(&array.into_array())?;
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
        let compressor = CascadingCompressor::new(vec![&RLE_INTEGER_SCHEME]);
        let compressed = compressor.compress(&array.into_array())?;
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
        drop(btr.compress(&prim)?);

        Ok(())
    }
}

/// Tests to verify that each integer compression scheme produces the expected encoding.
#[cfg(test)]
mod scheme_selection_tests {
    use std::iter;

    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use vortex_array::IntoArray;
    use vortex_array::arrays::Constant;
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_fastlanes::BitPacked;
    use vortex_fastlanes::FoR;
    use vortex_runend::RunEnd;
    use vortex_sequence::Sequence;
    use vortex_sparse::Sparse;

    use crate::BtrBlocksCompressor;

    #[test]
    fn test_constant_compressed() -> VortexResult<()> {
        let values: Vec<i32> = iter::repeat_n(42, 100).collect();
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array.into_array())?;
        assert!(compressed.is::<Constant>());
        Ok(())
    }

    #[test]
    fn test_for_compressed() -> VortexResult<()> {
        let values: Vec<i32> = (0..1000).map(|i| 1_000_000 + ((i * 37) % 100)).collect();
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array.into_array())?;
        assert!(compressed.is::<FoR>());
        Ok(())
    }

    #[test]
    fn test_bitpacking_compressed() -> VortexResult<()> {
        let values: Vec<u32> = (0..1000).map(|i| i % 16).collect();
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array.into_array())?;
        assert!(compressed.is::<BitPacked>());
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
        let compressed = btr.compress(&array.into_array())?;
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
        let compressed = btr.compress(&array.into_array())?;
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
        let compressed = btr.compress(&array.into_array())?;
        assert!(compressed.is::<RunEnd>());
        Ok(())
    }

    #[test]
    fn test_sequence_compressed() -> VortexResult<()> {
        let values: Vec<i32> = (0..1000).map(|i| i * 7).collect();
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.compress(&array.into_array())?;
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
        let compressed = btr.compress(&array.into_array())?;
        eprintln!("{}", compressed.display_tree());
        assert!(compressed.is::<RunEnd>());
        Ok(())
    }
}
