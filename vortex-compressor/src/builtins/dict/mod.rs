// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Dictionary encoding schemes for integer, float, and string arrays.

pub mod float;
pub mod integer;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::DictArray;
use vortex_array::builders::dict::dict_encode;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::is_float_primitive;
use super::is_integer_primitive;
use super::is_utf8_string;
use crate::CascadingCompressor;
use crate::ctx::CompressorContext;
use crate::scheme::ChildSelection;
use crate::scheme::DescendantExclusion;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::scheme::estimate_compression_ratio_with_sampling;
use crate::stats::ArrayAndStats;
use crate::stats::GenerateStatsOptions;

/// Dictionary encoding for low-cardinality integer values.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IntDictScheme;

impl Scheme for IntDictScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.dict"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_integer_primitive(canonical)
    }

    fn stats_options(&self) -> GenerateStatsOptions {
        GenerateStatsOptions {
            count_distinct_values: true,
        }
    }

    /// Children: values=0, codes=1.
    fn num_children(&self) -> usize {
        2
    }

    fn expected_compression_ratio(
        &self,
        _compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> VortexResult<f64> {
        let stats = data.integer_stats();

        if stats.value_count() == 0 {
            return Ok(0.0);
        }

        let distinct_values_count = stats.distinct_count().vortex_expect(
            "this must be present since `DictScheme` declared that we need distinct values",
        );

        // If > 50% of the values are distinct, skip dict.
        if distinct_values_count > stats.value_count() / 2 {
            return Ok(0.0);
        }

        // Ignore nulls encoding for the estimate. We only focus on values.
        let values_size = stats.source().ptype().bit_width() * distinct_values_count as usize;

        // Assume codes are compressed RLE + BitPacking.
        let codes_bw = usize::BITS - distinct_values_count.leading_zeros();

        let n_runs = (stats.value_count() / stats.average_run_length()) as usize;

        // Assume that codes will either be BitPack or RLE-BitPack.
        let codes_size_bp = (codes_bw * stats.value_count()) as usize;
        let codes_size_rle_bp = usize::checked_mul((codes_bw + 32) as usize, n_runs);

        let codes_size = usize::min(codes_size_bp, codes_size_rle_bp.unwrap_or(usize::MAX));

        let before = stats.value_count() as usize * stats.source().ptype().bit_width();

        Ok(before as f64 / (values_size + codes_size) as f64)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let stats = data.integer_stats();

        let dict = integer::dictionary_encode(stats);

        // Values = child 0.
        let compressed_values = compressor.compress_child(dict.values(), &ctx, self.id(), 0)?;

        // Codes = child 1.
        let compressed_codes = compressor.compress_child(
            &dict.codes().to_primitive().narrow()?.into_array(),
            &ctx,
            self.id(),
            1,
        )?;

        // SAFETY: compressing codes does not change their values.
        unsafe {
            Ok(
                DictArray::new_unchecked(compressed_codes, compressed_values)
                    .set_all_values_referenced(dict.has_all_values_referenced())
                    .into_array(),
            )
        }
    }
}

/// Dictionary encoding for low-cardinality float values.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FloatDictScheme;

impl Scheme for FloatDictScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.dict"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_float_primitive(canonical)
    }

    fn stats_options(&self) -> GenerateStatsOptions {
        GenerateStatsOptions {
            count_distinct_values: true,
        }
    }

    /// Children: values=0, codes=1.
    fn num_children(&self) -> usize {
        2
    }

    /// Float dict codes (child 1) are compact unsigned integers that should not be
    /// dict-encoded again. Float dict values (child 0) flow through ALP into integer-land,
    /// where integer dict encoding is redundant since the values are already deduplicated at
    /// the float level.
    ///
    /// Additional exclusions for codes (IntSequenceScheme, IntRunEndScheme, FoRScheme,
    /// ZigZagScheme, SparseScheme, RLE) are expressed as pull rules on those schemes in
    /// vortex-btrblocks.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![
            DescendantExclusion {
                excluded: IntDictScheme.id(),
                children: ChildSelection::One(1),
            },
            DescendantExclusion {
                excluded: IntDictScheme.id(),
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
        let stats = data.float_stats();

        if stats.value_count() == 0 {
            return Ok(0.0);
        }

        if stats
            .distinct_count()
            .is_some_and(|count| count <= stats.value_count() / 2)
        {
            return estimate_compression_ratio_with_sampling(self, compressor, data.array(), ctx);
        }

        Ok(0.0)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let stats = data.float_stats();

        let dict = float::dictionary_encode(stats);
        let has_all_values_referenced = dict.has_all_values_referenced();
        // let DictArrayParts { codes, values, .. } = dict.into_parts();

        // Values = child 0.
        let compressed_values = compressor.compress_child(dict.values(), &ctx, self.id(), 0)?;

        // Codes = child 1.
        let compressed_codes = compressor.compress_child(
            &dict.codes().to_primitive().narrow()?.into_array(),
            &ctx,
            self.id(),
            1,
        )?;

        // SAFETY: compressing codes or values does not alter the invariants.
        unsafe {
            Ok(
                DictArray::new_unchecked(compressed_codes, compressed_values)
                    .set_all_values_referenced(has_all_values_referenced)
                    .into_array(),
            )
        }
    }
}

/// Dictionary encoding for low-cardinality string values.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct StringDictScheme;

impl Scheme for StringDictScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.string.dict"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_utf8_string(canonical)
    }

    fn stats_options(&self) -> GenerateStatsOptions {
        GenerateStatsOptions {
            count_distinct_values: true,
        }
    }

    /// Children: values=0, codes=1.
    fn num_children(&self) -> usize {
        2
    }

    /// String dict codes (child 1) are compact unsigned integers that should not be dict-encoded
    /// again.
    ///
    /// Additional exclusions for codes (IntSequenceScheme, FoRScheme, ZigZagScheme, SparseScheme,
    /// RunEndScheme, RLE, etc.) are expressed as pull rules on those schemes in `vortex-btrblocks`.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![DescendantExclusion {
            excluded: IntDictScheme.id(),
            children: ChildSelection::One(1),
        }]
    }

    fn expected_compression_ratio(
        &self,
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<f64> {
        let stats = data.string_stats();

        if stats
            .estimated_distinct_count()
            .is_none_or(|c| c > stats.value_count() / 2)
        {
            return Ok(0.0);
        }

        if stats.value_count() == 0 {
            return Ok(0.0);
        }

        estimate_compression_ratio_with_sampling(self, compressor, data.array(), ctx)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let stats = data.string_stats();

        let dict = dict_encode(&stats.source().clone().into_array())?;

        // Values = child 0.
        let compressed_values = compressor.compress_child(dict.values(), &ctx, self.id(), 0)?;

        // Codes = child 1.
        let compressed_codes = compressor.compress_child(
            &dict.codes().to_primitive().narrow()?.into_array(),
            &ctx,
            self.id(),
            1,
        )?;

        // SAFETY: compressing codes or values does not alter the invariants.
        unsafe {
            Ok(
                DictArray::new_unchecked(compressed_codes, compressed_values)
                    .set_all_values_referenced(dict.has_all_values_referenced())
                    .into_array(),
            )
        }
    }
}
