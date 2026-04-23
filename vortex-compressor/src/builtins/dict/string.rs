// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! UTF8-specific dictionary encoding implementation.
//!
//! Vortex encoders must always produce unsigned integer codes; signed codes are only accepted
//! for external compatibility.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::DictArrayExt;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::builders::dict::dict_encode;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::CascadingCompressor;
use crate::builtins::IntDictScheme;
use crate::builtins::StringDictScheme;
use crate::builtins::is_utf8_string;
use crate::ctx::CompressorContext;
use crate::estimate::CompressionEstimate;
use crate::estimate::DeferredEstimate;
use crate::estimate::EstimateVerdict;
use crate::scheme::ChildSelection;
use crate::scheme::DescendantExclusion;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::stats::ArrayAndStats;
use crate::stats::GenerateStatsOptions;

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
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let stats = data.string_stats(exec_ctx);

        if stats.value_count() == 0 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // This gate is intentionally permissive. Dictionary encoding only needs a cheap signal that
        // repeated categories are plausible; sampling decides whether the final dictionary layout
        // is actually better than the alternatives. Using the suffix-aware string distinct count
        // here turned out to be too strict for URL- and path-like columns: those arrays often have
        // high full-value cardinality while still benefiting from dictionary encoding because many
        // rows reuse exact values across the larger compression batch. The coarser
        // `(length, first four bytes)` count keeps those candidates eligible for sampling instead
        // of skipping dictionary encoding up front.
        let estimated_distinct_values_count =
            stats.estimated_prefix_distinct_count().vortex_expect(
                "this must be present since `DictScheme` declared that we need distinct values",
            );

        // If > 50% of the values are distinct, skip dictionary scheme.
        if estimated_distinct_values_count > stats.value_count() / 2 {
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
        let dict = dict_encode(data.array())?;

        // Values = child 0.
        let compressed_values =
            compressor.compress_child(dict.values(), &compress_ctx, self.id(), 0, exec_ctx)?;

        // Codes = child 1.
        let narrowed_codes = dict
            .codes()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow()?
            .into_array();
        let compressed_codes =
            compressor.compress_child(&narrowed_codes, &compress_ctx, self.id(), 1, exec_ctx)?;

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

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;

    use super::*;

    #[test]
    fn string_dict_stays_eligible_for_common_prefix_varying_tail_values() {
        let strings = VarBinViewArray::from_iter(
            [
                Some("https://example.com/events/0000"),
                Some("https://example.com/events/0001"),
                Some("https://example.com/events/0002"),
                Some("https://example.com/events/0003"),
            ],
            DType::Utf8(Nullability::NonNullable),
        );
        let data = ArrayAndStats::new(
            strings.into_array(),
            GenerateStatsOptions {
                count_distinct_values: true,
            },
        );
        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        assert!(matches!(
            StringDictScheme.expected_compression_ratio(&data, CompressorContext::new(), &mut ctx,),
            CompressionEstimate::Deferred(DeferredEstimate::Sample)
        ));
    }

    #[test]
    fn string_dict_skips_when_prefix_cardinality_is_high() {
        let strings = VarBinViewArray::from_iter(
            [Some("aaaa"), Some("bbbb"), Some("cccc"), Some("dddd")],
            DType::Utf8(Nullability::NonNullable),
        );
        let data = ArrayAndStats::new(
            strings.into_array(),
            GenerateStatsOptions {
                count_distinct_values: true,
            },
        );
        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        assert!(matches!(
            StringDictScheme.expected_compression_ratio(&data, CompressorContext::new(), &mut ctx,),
            CompressionEstimate::Verdict(EstimateVerdict::Skip)
        ));
    }
}
