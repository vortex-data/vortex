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
use vortex_array::arrays::dict::DictArrayExt;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::builders::dict::dict_encode;
use vortex_error::VortexResult;

use crate::CascadingCompressor;
use crate::builtins::IntDictScheme;
use crate::scheme::ChildSelection;
use crate::scheme::CompressionEstimate;
use crate::scheme::CompressorContext;
use crate::scheme::DeferredEstimate;
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
    /// dict-encoded again. Decimal dict values (child 0) are split into byte parts, whose
    /// integers are already deduplicated at the decimal level.
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
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // If > 50% of the values are distinct, skip dictionary scheme.
        if low_cardinality_decimal(data, exec_ctx) != Some(true) {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // Let sampling determine the expected ratio.
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
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

/// Returns whether at most half of the valid values are distinct, or `None` when the distinct
/// values were not counted for `data`.
///
/// [`DecimalDictScheme`] requests the count. The decimal byte parts scheme uses this to compete
/// with the dictionary instead of winning outright when a dictionary may be smaller.
pub fn low_cardinality_decimal(data: &ArrayAndStats, exec_ctx: &mut ExecutionCtx) -> Option<bool> {
    let stats = data.decimal_stats(exec_ctx);
    let distinct_count = stats.distinct_count()?;
    Some(stats.value_count() > 0 && distinct_count <= stats.value_count() / 2)
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
