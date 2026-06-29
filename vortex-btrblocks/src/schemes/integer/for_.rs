// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Frame of Reference integer encoding.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_compressor::builtins::BinaryDictScheme;
use vortex_compressor::builtins::FloatDictScheme;
use vortex_compressor::builtins::IntDictScheme;
use vortex_compressor::builtins::StringDictScheme;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::EstimateVerdict;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_fastlanes::FoR;
use vortex_fastlanes::FoRArrayExt;

use super::BitPackingScheme;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

/// Frame of Reference encoding.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FoRScheme;

impl Scheme for FoRScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.for"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
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
            AncestorExclusion {
                ancestor: BinaryDictScheme.id(),
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
        let for_array = FoR::encode(primitive, exec_ctx)?;
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
