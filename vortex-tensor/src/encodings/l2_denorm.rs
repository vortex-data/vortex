// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_compressor::CascadingCompressor;
use vortex_compressor::ctx::CompressorContext;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::EstimateVerdict;
use vortex_compressor::scheme::Scheme;
use vortex_compressor::stats::ArrayAndStats;
use vortex_error::VortexResult;

use crate::matcher::AnyTensor;
use crate::scalar_fns::l2_denorm::normalize_as_l2_denorm;

#[derive(Debug)]
pub struct L2DenormScheme;

impl Scheme for L2DenormScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.tensor.l2_denorm"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches!(
            canonical,
            Canonical::Extension(ext) if ext.ext_dtype().is::<AnyTensor>()
        )
    }

    fn expected_compression_ratio(
        &self,
        _data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> CompressionEstimate {
        CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let l2_denorm =
            normalize_as_l2_denorm(data.array().clone(), &mut compressor.execution_ctx())?;
        Ok(l2_denorm.into_array())
    }
}
