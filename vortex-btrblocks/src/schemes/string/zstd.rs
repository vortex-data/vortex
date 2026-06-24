// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Zstd string compression without dictionaries (nvCOMP compatible).

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::DeferredEstimate;
use vortex_error::VortexResult;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;

/// Zstd compression without dictionaries (nvCOMP compatible).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ZstdScheme;

impl Scheme for ZstdScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.string.zstd"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_utf8()
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let compacted = data
            .array_as_varbinview()
            .into_owned()
            .compact_buffers(exec_ctx)?;
        Ok(
            vortex_zstd::Zstd::from_var_bin_view_without_dict(&compacted, 3, 8192, exec_ctx)?
                .into_array(),
        )
    }
}
