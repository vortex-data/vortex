// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Constant compression, selected through the same registry as other schemes.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::VTable;
use vortex_array::arrays::Bool;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Masked;
use vortex_error::VortexResult;

use crate::CascadingCompressor;
use crate::compressor::constant::compress_constant;
use crate::compressor::constant::is_constant_for_compression;
use crate::scheme::CompressionEstimate;
use crate::scheme::CompressorContext;
use crate::scheme::DeferredEstimate;
use crate::scheme::EstimateVerdict;
use crate::scheme::Scheme;
use crate::stats::ArrayAndStats;

/// Compress entirely valid or entirely null leaves as constants.
#[derive(Debug)]
pub struct ConstantScheme;

/// Compress partially valid constant leaves with a validity wrapper.
#[derive(Debug)]
pub struct MaskedConstantScheme;

impl Scheme for ConstantScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.compressor.constant"
    }

    fn selection_priority(&self) -> u16 {
        0
    }

    fn matches(&self, _array: &Canonical) -> bool {
        true
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Constant.id()]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        ctx: CompressorContext,
        exec: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        constant_estimate(data, ctx, exec, false)
    }

    fn compress(
        &self,
        _: &CascadingCompressor,
        data: &ArrayAndStats,
        _: CompressorContext,
        exec: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        compress_constant(data.array(), exec)
    }
}

impl Scheme for MaskedConstantScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.compressor.masked_constant"
    }

    fn selection_priority(&self) -> u16 {
        0
    }

    fn matches(&self, _array: &Canonical) -> bool {
        true
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Constant.id(), Masked.id(), Bool.id()]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        ctx: CompressorContext,
        exec: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        constant_estimate(data, ctx, exec, true)
    }

    fn compress(
        &self,
        _: &CascadingCompressor,
        data: &ArrayAndStats,
        _: CompressorContext,
        exec: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        compress_constant(data.array(), exec)
    }
}

/// Keep constant samples from deciding whether a full input is constant.
fn constant_estimate(
    data: &ArrayAndStats,
    ctx: CompressorContext,
    exec: &mut ExecutionCtx,
    masked: bool,
) -> CompressionEstimate {
    let result = (|| -> VortexResult<bool> {
        let validity = data
            .array()
            .validity()?
            .execute_mask(data.array_len(), exec)?;
        if validity.all_false() {
            return Ok(!masked);
        }
        if ctx.is_sample() || masked == validity.all_true() {
            return Ok(false);
        }
        is_constant_for_compression(data, exec)
    })();
    match result {
        Ok(true) => CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse),
        Ok(false) => CompressionEstimate::Verdict(EstimateVerdict::Skip),
        Err(error) => CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            move |_, _, _, _, _| Err(error),
        ))),
    }
}
