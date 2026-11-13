// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal-specific dictionary encoding implementation.
//!
//! Vortex encoders must always produce unsigned integer codes; signed codes are only accepted for
//! external compatibility.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::decimal::DecimalArrayExt;
use vortex_array::arrays::dict::DictArrayExt;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::builders::dict::dict_encode;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::CascadingCompressor;
use crate::builtins::IntDictScheme;
use crate::scheme::ChildSelection;
use crate::scheme::CompressionEstimate;
use crate::scheme::CompressorContext;
use crate::scheme::DescendantExclusion;
use crate::scheme::EstimateVerdict;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::stats::ArrayAndStats;
use crate::stats::GenerateStatsOptions;

/// Dictionary encoding for low-cardinality decimal values.
///
/// The dictionary holds each distinct decimal once, so a wide decimal that the decimal scheme
/// would split into several parts gets one codes array instead of one per part.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DecimalDictScheme;

impl Scheme for DecimalDictScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.decimal.dict"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches!(canonical, Canonical::Decimal(_))
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Dict.id()]
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

    /// Decimal dict codes (child 1) are compact unsigned integers that should not be
    /// dict-encoded again.
    ///
    /// The values (child 0) keep integer dictionaries: distinct decimals that byte parts split
    /// into several parts can share a most significant part.
    ///
    /// Pull rules in `vortex-btrblocks` also keep `SequenceScheme`, `FoRScheme`, `ZigZagScheme` and
    /// `DeltaScheme` off the codes (child 1), and `RunEndScheme` and the RLE schemes off the values
    /// (child 0).
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![DescendantExclusion {
            excluded: IntDictScheme.id(),
            children: ChildSelection::One(1),
        }]
    }

    /// Estimates the ratio from the exact distinct count, like `IntDictScheme`.
    ///
    /// A sample would under-rate a dictionary of thousands of distinct values: the sample holds
    /// almost as many distinct values as rows, so its dictionary is no smaller than its input.
    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let stats = data.decimal_stats(exec_ctx);

        if stats.value_count() == 0 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        let distinct_values_count = stats.distinct_count().vortex_expect(
            "this must be present since `DecimalDictScheme` declared that we need distinct values",
        );

        // If > 50% of the values are distinct, skip dictionary scheme.
        if distinct_values_count > stats.value_count() / 2 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // Ignore nulls encoding for the estimate. We only focus on values, at their stored width,
        // since the values are compressed again.
        let bit_width = data.array_as_decimal().values_type().byte_width() * 8;
        let values_size = bit_width * distinct_values_count as usize;

        // Assume codes are compressed RLE + BitPacking.
        let codes_bw = u32::BITS - distinct_values_count.leading_zeros();
        let average_run_length = stats
            .average_run_length()
            .vortex_expect("runs are counted with the distinct values");
        let n_runs = (stats.value_count() / average_run_length) as usize;

        // Assume that codes will either be BitPack or RLE-BitPack.
        let codes_size_bp = codes_bw as usize * stats.value_count() as usize;
        let codes_size_rle_bp = usize::checked_mul(codes_bw as usize + 32, n_runs);

        let codes_size = usize::min(codes_size_bp, codes_size_rle_bp.unwrap_or(usize::MAX));

        let before = stats.value_count() as usize * bit_width;

        CompressionEstimate::Verdict(EstimateVerdict::Ratio(
            before as f64 / (values_size + codes_size) as f64,
        ))
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let dict = dict_encode(data.array(), exec_ctx)?;
        let has_all_values_referenced = dict.has_all_values_referenced();

        // Values = child 0.
        let compressed_values =
            compressor.compress_child(dict.values(), &compress_ctx, self.id(), 0, exec_ctx)?;

        // Codes = child 1. The encoder already narrowed them.
        let compressed_codes =
            compressor.compress_child(dict.codes(), &compress_ctx, self.id(), 1, exec_ctx)?;

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

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::Dict;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::DecimalDictScheme;
    use crate::CascadingCompressor;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn decimal(values: impl IntoIterator<Item = i128>, validity: Validity) -> DecimalArray {
        DecimalArray::new(
            values.into_iter().collect::<Buffer<i128>>(),
            DecimalDType::new(38, 2),
            validity,
        )
    }

    #[rstest]
    #[case::non_nullable(Validity::NonNullable)]
    #[case::nullable(Validity::from_iter((0..1024).map(|i| i % 5 != 0)))]
    fn compresses_low_cardinality_decimals_to_dict(#[case] validity: Validity) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let distinct = [1i128 << 100, -(1i128 << 90), 7];
        let array = decimal((0..1024).map(|i| distinct[i % 3]), validity).into_array();

        let compressed =
            CascadingCompressor::new(vec![&DecimalDictScheme]).compress(&array, &mut ctx)?;

        assert!(compressed.is::<Dict>(), "{}", compressed.display_tree());
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }

    #[test]
    fn skips_high_cardinality_decimals() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = decimal((0..1024).map(|i| i * 17), Validity::NonNullable).into_array();

        let compressed =
            CascadingCompressor::new(vec![&DecimalDictScheme]).compress(&array, &mut ctx)?;

        assert!(!compressed.is::<Dict>(), "{}", compressed.display_tree());
        Ok(())
    }
}
