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
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedDataParts;
use vortex_fastlanes::BitWidths;
use vortex_fastlanes::bitpacked_v1_id;

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
        // ALP-RD bit-packs both children with the v1 encoding, and `compress_patches` stores
        // constant patch values as a Constant array.
        vec![vortex_alp::ALPRD.id(), bitpacked_v1_id(), Constant.id()]
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
        let (left_parts, validity) = take_validity(left_parts)?;

        let mut keep = Vec::new();
        // The patches are of the left parts, not of the input, so they stay inside ALP-RD.
        let patches = KeptPatches::interior(self, 0, left_parts_patches, &mut keep, exec_ctx)?;
        Ok(Encoded::new(Vec::new(), keep, move |children, _| {
            let left_parts = left_parts.with_validity(children.validity()?)?;
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

/// ALP-RD's bit-packed left parts with their validity taken out, to be registered.
struct LeftParts {
    /// The parts of the bit-packed array but its validity.
    parts: BitPackedDataParts,
    /// The type of the packed values.
    ptype: PType,
}

impl LeftParts {
    /// Rebuilds the bit-packed left parts with `validity`.
    fn with_validity(self, validity: Validity) -> VortexResult<ArrayRef> {
        let BitPackedDataParts {
            offset,
            bit_widths,
            len,
            packed,
            patches,
            ..
        } = self.parts;
        let BitWidths::Global(bit_width) = bit_widths else {
            vortex_bail!("ALP-RD packs its left parts at one bit width");
        };
        Ok(BitPacked::try_new(
            packed, self.ptype, validity, patches, bit_width, len, offset,
        )?
        .into_array())
    }
}

/// Splits the validity off ALP-RD's bit-packed left parts.
fn take_validity(left_parts: ArrayRef) -> VortexResult<(LeftParts, Validity)> {
    let packed = left_parts.try_downcast::<BitPacked>().map_err(|array| {
        vortex_err!(
            "ALP-RD left parts are {}, not bit-packed",
            array.encoding_id()
        )
    })?;
    let ptype = packed.dtype().as_ptype();
    let mut parts = BitPacked::into_parts(packed);
    let validity = std::mem::replace(&mut parts.validity, Validity::NonNullable);
    Ok((LeftParts { parts, ptype }, validity))
}
