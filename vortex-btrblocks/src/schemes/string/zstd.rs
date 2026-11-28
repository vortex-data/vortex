// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Zstd string compression without dictionaries (nvCOMP compatible).

use vortex_array::ArrayId;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::VarBinViewArray;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_error::VortexResult;

use crate::ArrayAndStats;
use crate::CompressorContext;
use crate::Encoded;
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

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![vortex_zstd::Zstd.id()]
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    /// Zstd stores the input's validity, which is registered.
    fn encode(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Encoded> {
        // Compact before the validity is registered: compaction rebuilds the array, so compacting
        // the leaf would replace its compressed bitmap with an uncompressed copy.
        let compacted = data
            .array()
            .clone()
            .execute::<VarBinViewArray>(exec_ctx)?
            .compact_buffers(exec_ctx)?;
        Encoded::of_leaf(compacted.into_array(), |input, exec_ctx| {
            let input = input.execute::<VarBinViewArray>(exec_ctx)?;
            Ok(
                vortex_zstd::Zstd::from_var_bin_view_without_dict(&input, 3, 8192, exec_ctx)?
                    .into_array(),
            )
        })
    }
}
