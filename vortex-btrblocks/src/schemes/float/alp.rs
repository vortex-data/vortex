// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ALP (Adaptive Lossless floating-Point) encoding.

use vortex_alp::ALP;
use vortex_alp::ALPArrayExt;
use vortex_alp::ALPArraySlotsExt;
use vortex_alp::alp_encode;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Patched;
use vortex_array::arrays::patched::use_experimental_patches;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::PType;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::compress_patches;

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
        let mut encodings = vec![ALP.id()];
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

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let alp_encoded = alp_encode(data.array_as_primitive(), None, exec_ctx)?;

        // Compress the ALP ints.
        let compressed_alp_ints = compressor.compress_child(
            alp_encoded.encoded(),
            &compress_ctx,
            self.id(),
            0,
            exec_ctx,
        )?;

        let alp_stats = alp_encoded.as_array().aggregations().snapshot_results();
        let exponents = alp_encoded.exponents();

        let array = if use_experimental_patches() {
            let patches = alp_encoded.patches();

            // Create ALP array without interior patches.
            let alp_array = ALP::new(compressed_alp_ints, exponents, None).into_array();

            match patches {
                None => alp_array,
                Some(p) => Patched::from_array_and_patches(alp_array, &p, exec_ctx)?.into_array(),
            }
        } else {
            let patches = alp_encoded
                .patches()
                .map(|p| compress_patches(p, exec_ctx))
                .transpose()?;

            ALP::new(compressed_alp_ints, exponents, patches).into_array()
        };

        if alp_encoded.as_array().dtype() == array.dtype()
            && alp_encoded.as_array().len() == array.len()
        {
            for (aggregate, result) in alp_stats.iter() {
                if aggregate.is_representation_invariant() {
                    // SAFETY: recompressing ALP integers and patches preserves values and null
                    // positions at the same dtype and length.
                    unsafe {
                        array
                            .aggregations()
                            .seed_result(aggregate.clone(), result.clone())?;
                    }
                }
            }
        }

        Ok(array)
    }
}
