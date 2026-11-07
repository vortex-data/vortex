// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arithmetic run-end positions with a terminal exception, as produced by sparse list offsets.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateScore;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_sequence::Sequence;
use vortex_sequence::sequence_encode;

use super::RunEndScheme;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

/// Encodes regularly spaced run ends followed by a single terminal exception.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RunEndSequenceScheme;

impl Scheme for RunEndSequenceScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.runend_sequence"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Chunked.id(), Sequence.id(), Constant.id()]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        if compress_ctx.is_sample()
            || data.array_len() < 4
            || !compress_ctx
                .cascade_history()
                .contains(&(RunEndScheme.id(), 1))
            || data.integer_stats(exec_ctx).null_count() != 0
        {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            |_compressor, data, best, _ctx, exec_ctx| {
                let upper_bound = data.array().nbytes() as f64 / 24.0;
                if best
                    .and_then(EstimateScore::finite_ratio)
                    .is_some_and(|ratio| ratio >= upper_bound)
                {
                    return Ok(EstimateVerdict::Skip);
                }
                let Some(encoded) = encode_runend_sequence(data, exec_ctx)? else {
                    return Ok(EstimateVerdict::Skip);
                };
                Ok(EstimateVerdict::Ratio(
                    data.array().nbytes() as f64 / encoded.nbytes() as f64,
                ))
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
        encode_runend_sequence(data, exec_ctx)?
            .ok_or_else(|| vortex_err!("run ends do not have a sequential prefix"))
    }
}

fn encode_runend_sequence(
    data: &ArrayAndStats,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let last = data.array_len() - 1;
    let prefix = data.array().slice(0..last)?.execute::<PrimitiveArray>(ctx)?;
    let Some(prefix) = sequence_encode(prefix.as_view(), ctx)? else {
        return Ok(None);
    };
    let tail = ConstantArray::new(data.array().execute_scalar(last, ctx)?, 1).into_array();
    Ok(Some(
        ChunkedArray::try_new([prefix, tail], data.array().dtype().clone())?.into_array(),
    ))
}
