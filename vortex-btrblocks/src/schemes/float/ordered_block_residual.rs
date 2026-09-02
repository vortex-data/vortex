// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Ordered float bits with one block-local reference and packed residuals.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::match_each_float_ptype;
use vortex_block_residual::BlockResidual;
use vortex_block_residual::OrderedFloat;
use vortex_block_residual::OrderedFloatArraySlotsExt;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::normalize_null_values;
use crate::schemes::integer::patch_adjusted_estimate_nbytes;
use crate::schemes::sample_primitive_blocks;

const BLOCK_LEN: usize = 1024;
const ESTIMATE_BLOCKS: usize = 8;
const MIN_COMPRESSION_RATIO: f64 = 1.05;
const DECODE_COST_FACTOR: f64 = 1.02;

/// Compress floats as block-local residuals of ordered IEEE bits.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct OrderedBlockResidualScheme;

impl Scheme for OrderedBlockResidualScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.ordered_block_residual"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_float()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![OrderedFloat.id(), BlockResidual.id()]
    }

    fn num_children(&self) -> usize {
        1
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        if compress_ctx.finished_cascading() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            |_compressor, data, _best_so_far, _compress_ctx, exec_ctx| {
                let sample = locality_sample(data.array_as_primitive(), exec_ctx)?;
                let sample = normalize_null_values(sample.as_view(), exec_ctx)?;
                let before_nbytes = sample.nbytes();
                let estimate = OrderedFloat::estimate_block_residual(sample.as_view())?;
                let after_nbytes = patch_adjusted_estimate_nbytes(estimate, sample.len());
                if after_nbytes == 0 {
                    return Ok(EstimateVerdict::Skip);
                }

                let ratio = before_nbytes as f64 / after_nbytes as f64;
                if ratio < MIN_COMPRESSION_RATIO {
                    return Ok(EstimateVerdict::Skip);
                }
                let adjusted_ratio = ratio / DECODE_COST_FACTOR;
                if adjusted_ratio < MIN_COMPRESSION_RATIO {
                    return Ok(EstimateVerdict::Skip);
                }
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
        let ordered = OrderedFloat::from_primitive(primitive.as_view())?;
        let ordered_values = ordered.encoded().as_::<Primitive>();
        let residuals = BlockResidual::from_primitive(ordered_values)?;
        Ok(OrderedFloat::try_new(residuals.into_array(), primitive.ptype())?.into_array())
    }
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
    Ok(match_each_float_ptype!(primitive.ptype(), |T| {
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
