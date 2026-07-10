// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Zstd buffer-level binary compression preserving array layout for GPU decompression.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_compressor::scheme::DeferredEvaluation;
use vortex_compressor::scheme::SchemeEvaluation;
use vortex_error::VortexResult;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;

/// Zstd buffer-level compression preserving array layout for GPU decompression.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ZstdBuffersScheme;

impl Scheme for ZstdBuffersScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.binary.zstd_buffers"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_binary()
    }

    fn evaluate(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> SchemeEvaluation {
        SchemeEvaluation::Deferred(DeferredEvaluation::Sample)
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        Ok(vortex_zstd::ZstdBuffers::compress(data.array(), 3, exec_ctx.session())?.into_array())
    }
}
