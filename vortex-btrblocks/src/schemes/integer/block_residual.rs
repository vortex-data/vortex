// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integer compression with one block-local reference and packed residuals.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::match_each_integer_ptype;
use vortex_block_residual::BlockResidual;
use vortex_block_residual::BlockResidualEstimate;
use vortex_compressor::builtins::BinaryDictScheme;
use vortex_compressor::builtins::StringDictScheme;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;

use super::ZigZagScheme;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::normalize_null_values;
use crate::schemes::sample_primitive_blocks;

const BLOCK_LEN: usize = 1024;
const ESTIMATE_BLOCKS: usize = 8;
const MIN_COMPRESSION_RATIO: f64 = 1.05;
// The weakest measured 8-bit gain increased file-access latency by 29 to 55 percent.
// Require about 12 percent estimated savings for 8-bit values.
const EIGHT_BIT_ACCESS_COST_FACTOR: f64 = 1.12;
// The 25-percent patch sweep needs about 80 cost bits per patch to reject its slow tree.
// This quadratic slope reaches that cost at 25 percent while it keeps sparse patches cheap.
const PATCH_DENSITY_COST_BITS: u64 = 320;

/// Compress integers with one reference and packed residuals per 1,024-value block.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BlockResidualScheme;

impl Scheme for BlockResidualScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.block_residual"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![BlockResidual.id()]
    }

    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        vec![
            AncestorExclusion {
                ancestor: StringDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: BinaryDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: ZigZagScheme.id(),
                children: ChildSelection::One(0),
            },
        ]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // A single block cannot amortize a block-local reference against FoR.
        if data.array().len() <= BLOCK_LEN
            || compress_ctx.finished_cascading()
            || compress_ctx.is_sample()
        {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            |_compressor, data, _best_so_far, _compress_ctx, exec_ctx| {
                let sample = locality_sample(data.array_as_primitive(), exec_ctx)?;
                let sample = normalize_null_values(sample.as_view(), exec_ctx)?;
                let before_nbytes = sample.nbytes();
                let estimate = BlockResidual::estimate_primitive(sample.as_view())?;
                let after_nbytes = patch_adjusted_estimate_nbytes(estimate, sample.len());
                if after_nbytes == 0 {
                    return Ok(EstimateVerdict::Skip);
                }

                let ratio = before_nbytes as f64 / after_nbytes as f64;
                if ratio < MIN_COMPRESSION_RATIO {
                    return Ok(EstimateVerdict::Skip);
                }
                let adjusted_ratio = if sample.ptype().byte_width() == 1 {
                    ratio / EIGHT_BIT_ACCESS_COST_FACTOR
                } else {
                    ratio
                };
                Ok(EstimateVerdict::Ratio(adjusted_ratio))
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
        let primitive = normalize_null_values(data.array_as_primitive(), exec_ctx)?;
        Ok(BlockResidual::from_primitive(primitive.as_view())?.into_array())
    }
}

pub(crate) fn patch_adjusted_estimate_nbytes(estimate: BlockResidualEstimate, len: usize) -> u64 {
    estimate
        .nbytes()
        .saturating_add(patch_density_cost_bytes(len, estimate.patch_count()))
}

fn patch_density_cost_bytes(len: usize, patch_count: usize) -> u64 {
    if len == 0 {
        return 0;
    }
    let patch_count = u64::try_from(patch_count).unwrap_or(u64::MAX);
    patch_count
        .saturating_mul(patch_count)
        .saturating_mul(PATCH_DENSITY_COST_BITS)
        .div_ceil(u64::try_from(len).unwrap_or(u64::MAX).saturating_mul(8))
}

fn locality_sample(
    primitive: vortex_array::ArrayView<'_, Primitive>,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let validity = primitive
        .validity()?
        .execute_mask(primitive.len(), exec_ctx)?;
    let full_blocks = primitive.len() / BLOCK_LEN;

    if full_blocks <= ESTIMATE_BLOCKS {
        return primitive
            .array()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx);
    }

    let sample_blocks = ESTIMATE_BLOCKS.min(full_blocks);
    Ok(match_each_integer_ptype!(primitive.ptype(), |T| {
        sample_primitive_blocks(
            primitive.as_slice::<T>(),
            validity.all_true(),
            |index| validity.value(index),
            full_blocks,
            sample_blocks,
            BLOCK_LEN,
        )
    }))
}

#[cfg(test)]
mod tests {
    use super::patch_density_cost_bytes;

    #[test]
    fn patch_density_cost_is_nonlinear() {
        assert_eq!(patch_density_cost_bytes(1_024, 0), 0);
        assert_eq!(patch_density_cost_bytes(1_024, 102), 407);
        assert_eq!(patch_density_cost_bytes(1_024, 256), 2_560);
    }
}
