// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Entropy-coded bins for integers.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_error::VortexResult;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;

/// Entropy-coded bins: pco's bins with a SIMD tANS id stream and variable-width offsets, in
/// independently decodable 1024-value blocks. Opt-in: add it with
/// [`with_new_scheme`](crate::BtrBlocksCompressorBuilder::with_new_scheme).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct EntropyBinsScheme;

impl Scheme for EntropyBinsScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.entropy_bins"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![vortex_entropy_bins::EntropyBins.id()]
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
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let primitive = data.array_as_primitive();
        // Bins that do not fit the encoding's limits leave the array as it is.
        match vortex_entropy_bins::EntropyBins::from_primitive(
            primitive,
            pco::DEFAULT_COMPRESSION_LEVEL,
        ) {
            Ok(array) => Ok(array.into_array()),
            Err(_) => Ok(primitive.array().clone()),
        }
    }
}
