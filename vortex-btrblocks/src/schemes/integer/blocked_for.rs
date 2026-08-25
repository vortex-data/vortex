// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block-wise Frame of Reference integer encoding.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::PrimitiveArray;
use vortex_compressor::builtins::BinaryDictScheme;
use vortex_compressor::builtins::FloatDictScheme;
use vortex_compressor::builtins::IntDictScheme;
use vortex_compressor::builtins::StringDictScheme;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_fastlanes::BLOCK_SIZE;
use vortex_fastlanes::BlockedFoR;
use vortex_fastlanes::BlockedFoRArraySlotsExt;
use vortex_fastlanes::block_summary;

use super::BitPackingScheme;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

/// Number of bits needed to hold every value in `0..=range`.
fn bits_for(range: u128) -> u32 {
    range.checked_ilog2().map_or(0, |log| log + 1)
}

/// Block-wise Frame of Reference encoding.
///
/// Unlike [`super::FoRScheme`], which subtracts a single minimum from the whole array, this
/// scheme subtracts a separate minimum from every [`BLOCK_SIZE`] values. The bit width that
/// [`BitPackingScheme`] then picks is driven by the widest *block*, not by the spread of the
/// whole array, so data that drifts over the array — timestamps, sorted keys, counters — packs
/// far more tightly. The price is one reference per block.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BlockedFoRScheme;

impl Scheme for BlockedFoRScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.blocked_for"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![BlockedFoR.id()]
    }

    fn num_children(&self) -> usize {
        2
    }

    /// Dict codes always start at 0, so subtracting a minimum is a no-op.
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
        // Like FoR, this only subtracts a reference. Without a downstream codec (BitPacking) the
        // output is the same size as the input, plus the references.
        if compress_ctx.finished_cascading() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        let stats = data.integer_stats(exec_ctx);
        let global_bits = bits_for(u128::from(stats.erased().max_minus_min()));
        // A constant array should be compressed as one, not as a frame of reference.
        if global_bits == 0 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        let primitive = data.array_as_primitive();
        let Some(blocks) = block_summary(primitive, exec_ctx) else {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        };
        let blocked_bits = bits_for(blocks.max_range);

        let full_width: u32 = primitive
            .ptype()
            .bit_width()
            .try_into()
            .vortex_expect("bit width must fit in u32");

        // One reference per block, amortized over the values it covers.
        let len = primitive.len();
        let num_blocks = len.div_ceil(BLOCK_SIZE);
        #[expect(clippy::cast_precision_loss, reason = "estimate only")]
        let reference_bits = (num_blocks * full_width as usize) as f64 / len as f64;

        // Per-block references only pay off when they narrow the residuals. Where they don't,
        // skip and let `FoRScheme` take the array with a single global reference — this scheme
        // must never stand in for it, or an encoding policy that forbids `BlockedFoR` would
        // drop frame of reference altogether. `blocked_bits == 0` means every block is
        // constant, a shape RunEnd and RLE model better than zero-width residuals.
        if blocked_bits == 0 || blocked_bits >= global_bits || blocks.all_minima_zero {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        let effective_bits = f64::from(blocked_bits) + reference_bits;

        if let Some(max_log) = stats
            .erased()
            .max_ilog2()
            // Only skip when min >= 0, otherwise BitPacking can't be applied without ZigZag.
            .filter(|_| !stats.erased().min_is_negative())
        {
            // Plain BitPacking would already do at least this well, without any reference.
            if effective_bits >= f64::from(max_log + 1) {
                return CompressionEstimate::Verdict(EstimateVerdict::Skip);
            }
        }

        CompressionEstimate::Verdict(EstimateVerdict::Ratio(
            f64::from(full_width) / effective_bits,
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
        let blocked = BlockedFoR::encode(primitive, exec_ctx)?;
        let residuals = blocked
            .encoded()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?;

        // Immediately bitpack the residuals. If any other scheme was preferable, it would have
        // been chosen instead of this one.
        let leaf_ctx = compress_ctx.clone().as_leaf();
        let residual_data =
            ArrayAndStats::new(residuals.into_array(), compress_ctx.merged_stats_options());
        let compressed_residuals =
            BitPackingScheme.compress(compressor, &residual_data, leaf_ctx, exec_ctx)?;

        // The references are themselves a small, highly structured integer array — usually
        // sorted or near-sorted — so hand them back to the cascading compressor.
        let compressed_references = compressor.compress_child(
            blocked.references(),
            &compress_ctx,
            self.id(),
            1,
            exec_ctx,
        )?;

        let compressed = BlockedFoR::try_new(compressed_residuals, compressed_references, 0)?;
        compressed
            .as_ref()
            .statistics()
            .inherit_from(blocked.as_ref().statistics());

        Ok(compressed.into_array())
    }
}
