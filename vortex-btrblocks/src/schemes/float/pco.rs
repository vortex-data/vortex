// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pco (pcodec) float compression.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Primitive;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Encoded;
use crate::Scheme;

/// Pco (pcodec) compression for floats.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct PcoScheme;

impl Scheme for PcoScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.pco"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_float()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![vortex_pco::Pco.id()]
    }

    fn top_level_encodings(&self) -> Vec<ArrayId> {
        vec![vortex_pco::Pco.id()]
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    /// Pco stores the input's validity, which is registered.
    fn encode(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: &CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Encoded> {
        Encoded::of_leaf(data.array().clone(), |input, exec_ctx| {
            let input = input
                .as_opt::<Primitive>()
                .vortex_expect("compressing the validity keeps a primitive leaf primitive");
            Ok(vortex_pco::Pco::from_primitive(
                input,
                pco::DEFAULT_COMPRESSION_LEVEL,
                8192,
                exec_ctx,
            )?
            .into_array())
        })
    }

    #[expect(clippy::unimplemented, reason = "replaced by `encode`")]
    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unimplemented!("{} compresses through Scheme::encode", self.scheme_name())
    }
}
