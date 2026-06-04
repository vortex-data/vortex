// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Binary-specific dictionary encoding implementation.
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
use vortex_error::VortexResult;

use crate::CascadingCompressor;
use crate::builtins::IntDictScheme;
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

/// Dictionary encoding for low-cardinality binary values.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BinaryDictScheme;

impl Scheme for BinaryDictScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.binary.dict"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_binary()
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

    /// Binary dict codes (child 1) are compact unsigned integers that should not be dict-encoded
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
        let stats = data.varbinview_stats(exec_ctx);

        if stats.value_count() == 0 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        let max_useful_distinct_count = stats.value_count() / 2;

        // If more than 50% of the values are definitely distinct, skip dictionary encoding.
        // Cardinality is approximate for larger counts, so keep borderline estimates alive long
        // enough for sampling to decide.
        if !stats.estimated_distinct_count_could_be_at_most(max_useful_distinct_count) {
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

        // Values = child 0.
        let compressed_values =
            compressor.compress_child(dict.values(), &compress_ctx, self.id(), 0, exec_ctx)?;

        // Codes = child 1.
        let narrowed_codes = dict
            .codes()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?
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
