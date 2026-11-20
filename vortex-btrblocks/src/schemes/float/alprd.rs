// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ALPRD (ALP with Real Double) encoding variant.

use vortex_alp::ALPRDArrayExt;
use vortex_alp::ALPRDArrayOwnedExt;
use vortex_alp::RDEncoder;
use vortex_alp::RDEncoderExt;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Patched;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::PType;
use vortex_array::patches::Patches;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::compress_patches;
use crate::schemes::patches::PatchingScheme;
use crate::schemes::patches::WithPatched;
use crate::schemes::patches::patched_over;

/// ALPRD (ALP with Real Double) encoding variant.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ALPRDScheme;

impl Scheme for ALPRDScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.alprd"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_float()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![vortex_alp::ALPRD.id()]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // We don't support ALPRD for f16.
        if data.array_as_primitive().ptype() == PType::F16 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn refine(&self, allowed: &dyn Fn(&ArrayId) -> bool) -> &dyn Scheme {
        if allowed(&Patched.id()) {
            &ALPRD_PATCHED
        } else {
            self
        }
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        self.compress_patching(compressor, data, compress_ctx, exec_ctx, false)
    }
}

/// The variant of [`ALPRDScheme`] that writes its exceptions as a `vortex.patched` parent.
static ALPRD_PATCHED: WithPatched<ALPRDScheme> = WithPatched(ALPRDScheme);

impl PatchingScheme for ALPRDScheme {
    fn compress_patching(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
        patched: bool,
    ) -> VortexResult<ArrayRef> {
        let input = compressor.with_compressed_validity(data.array(), exec_ctx)?;
        let primitive_array = input
            .as_opt::<Primitive>()
            .vortex_expect("compressing the validity keeps a primitive leaf primitive");

        let encoder = match primitive_array.ptype() {
            PType::F32 => RDEncoder::new(primitive_array.as_slice::<f32>()),
            PType::F64 => RDEncoder::new(primitive_array.as_slice::<f64>()),
            ptype => vortex_panic!("cannot ALPRD compress ptype {ptype}"),
        };

        let alp_rd = encoder.encode(primitive_array);
        let dtype = alp_rd.dtype().clone();
        let right_bit_width = alp_rd.right_bit_width();
        let mut parts = ALPRDArrayOwnedExt::into_data_parts(alp_rd);
        // ALP-RD patches the left parts of its exceptions. A patched parent instead patches the
        // decoded floats, so it stores each exception as its original value, which costs the
        // right parts of the exceptions a second time and keeps ALP-RD itself free of patches.
        if patched && let Some(left_patches) = parts.left_parts_patches.take() {
            let exceptions = input.take(left_patches.indices().clone())?;
            let float_patches = Patches::new(
                left_patches.array_len(),
                left_patches.offset(),
                left_patches.indices().clone(),
                exceptions,
                None,
                exec_ctx,
            )?;
            let alp_rd = vortex_alp::ALPRD::try_new(
                dtype,
                parts.left_parts,
                parts.left_parts_dictionary,
                parts.right_parts,
                right_bit_width,
                None,
            )?
            .into_array();
            return patched_over(
                compressor,
                &compress_ctx,
                self.id(),
                alp_rd,
                &float_patches,
                0,
                None,
                exec_ctx,
            );
        }

        parts.left_parts_patches = parts
            .left_parts_patches
            .map(|p| compress_patches(p, exec_ctx))
            .transpose()?;

        Ok(vortex_alp::ALPRD::try_new(
            dtype,
            parts.left_parts,
            parts.left_parts_dictionary,
            parts.right_parts,
            right_bit_width,
            parts.left_parts_patches,
        )?
        .into_array())
    }
}
