// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ALP (Adaptive Lossless floating-Point) encoding.

use vortex_alp::ALP;
use vortex_alp::ALPArrayExt;
use vortex_alp::ALPArraySlotsExt;
use vortex_alp::alp_encode;
use vortex_array::ArrayId;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Patched;
use vortex_array::arrays::patched::use_experimental_patches;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::PType;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;

use crate::ArrayAndStats;
use crate::Child;
use crate::CompressorContext;
use crate::Encoded;
use crate::Scheme;
use crate::schemes::patches::KeptPatches;

/// ALP (Adaptive Lossless floating-Point) encoding.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ALPScheme;

impl Scheme for ALPScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.alp"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_float()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        // `compress_patches` stores constant patch values as a Constant array.
        let mut encodings = vec![ALP.id(), Constant.id()];
        if use_experimental_patches() {
            encodings.push(Patched.id());
        }
        encodings
    }

    /// Children: encoded_ints=0.
    fn num_children(&self) -> usize {
        1
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // ALP encodes floats as integers. Without integer compression afterward, the encoded ints
        // are the same size.
        if compress_ctx.finished_cascading() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // We don't support ALP for f16.
        if data.array_as_primitive().ptype() == PType::F16 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    /// Children: encoded ints=0, which carry the validity; patches, kept, from 1 on.
    fn encode(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Encoded> {
        let alp_encoded = alp_encode(data.array_as_primitive(), None, exec_ctx)?;
        let alp_stats = alp_encoded.as_array().statistics().to_owned();
        let exponents = alp_encoded.exponents();

        let (patches, keep) =
            KeptPatches::new(self, 1, data.array(), alp_encoded.patches(), exec_ctx)?;
        Ok(Encoded::new(
            vec![Child::new(self, 0, alp_encoded.encoded().clone())],
            keep,
            move |children, _| {
                let [encoded] = children.compressed()?;
                let array = patches.build(children, |patches| {
                    Ok(ALP::new(encoded, exponents, patches).into_array())
                })?;
                array.statistics().set_all(alp_stats);
                Ok(array)
            },
        ))
    }
}
