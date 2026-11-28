// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ALPRD (ALP with Real Double) encoding variant.

use vortex_alp::ALPRDArrayExt;
use vortex_alp::ALPRDArrayOwnedExt;
use vortex_alp::ALPRDDataParts;
use vortex_alp::RDEncoder;
use vortex_alp::RDEncoderExt;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Constant;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedSlots;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Encoded;
use crate::Scheme;
use crate::schemes::patches::KeptPatches;

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
        // ALP-RD bit-packs both children, and `compress_patches` stores constant patch values as a
        // Constant array.
        vec![vortex_alp::ALPRD.id(), BitPacked.id(), Constant.id()]
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

    /// Patches, kept, from 0 on. The left parts store the input's validity, which is registered.
    fn encode(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Encoded> {
        let primitive_array = data.array_as_primitive();
        let encoder = match primitive_array.ptype() {
            PType::F32 => RDEncoder::new(primitive_array.as_slice::<f32>()),
            PType::F64 => RDEncoder::new(primitive_array.as_slice::<f64>()),
            ptype => vortex_panic!("cannot ALPRD compress ptype {ptype}"),
        };

        let alp_rd = encoder.encode(primitive_array);
        let dtype = alp_rd.dtype().clone();
        let right_bit_width = alp_rd.right_bit_width();
        let ALPRDDataParts {
            left_parts,
            left_parts_patches,
            left_parts_dictionary,
            right_parts,
        } = ALPRDArrayOwnedExt::into_data_parts(alp_rd);
        vortex_ensure!(
            left_parts.is::<BitPacked>(),
            "ALP-RD left parts are {}, not bit-packed",
            left_parts.encoding_id()
        );
        let validity = left_parts.validity()?;

        // The patches are of the left parts, not of the input, so they stay inside ALP-RD.
        let (patches, keep) = KeptPatches::interior(self, 0, left_parts_patches, exec_ctx)?;
        Ok(Encoded::new(Vec::new(), keep, move |children, _| {
            let left_parts = with_compressed_validity(left_parts, children.validity()?)?;
            patches.build(children, |patches| {
                Ok(vortex_alp::ALPRD::try_new(
                    dtype,
                    left_parts,
                    left_parts_dictionary,
                    right_parts,
                    right_bit_width,
                    patches,
                )?
                .into_array())
            })
        })
        .with_validity(validity))
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

/// Puts `validity`, the compressed form of the validity of the bit-packed `left_parts`, in place of
/// their own.
fn with_compressed_validity(left_parts: ArrayRef, validity: Validity) -> VortexResult<ArrayRef> {
    if left_parts.slots()[BitPackedSlots::VALIDITY_CHILD].is_none() {
        // An all-valid or non-nullable validity has no bitmap, so compression leaves it as it is.
        return Ok(left_parts);
    }
    let bitmap = validity.to_array(left_parts.len());
    // SAFETY: the bitmap is a compressed form of the one it replaces, which holds the same values.
    unsafe { left_parts.with_slot(BitPackedSlots::VALIDITY_CHILD, bitmap) }
}
