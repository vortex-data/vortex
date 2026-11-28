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
use vortex_compressor::scheme::Child;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::DescendantAllowlist;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedSlots;
use vortex_fastlanes::bitpacked_v1_id;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Encoded;
use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::integer::PivcoScheme;
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

    /// Children: left_parts=0.
    fn num_children(&self) -> usize {
        1
    }

    /// `left_parts` (child 0) already holds dictionary codes `0..8` bit-packed to `<= 3` bits by
    /// `RDEncoder`. Allow-list `PivcoScheme` as the only integer scheme on child 0 (constant
    /// collapse to `ConstantArray` still applies) so other integer schemes never re-encode
    /// `left_parts`, while `PivcoScheme` can compress skewed exponent codes when enabled.
    fn descendant_allowlists(&self) -> Vec<DescendantAllowlist> {
        vec![DescendantAllowlist {
            allowed: vec![PivcoScheme::default().id()],
            children: ChildSelection::One(0),
        }]
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

    /// Child 0 is `left_parts` (compressed only if a scheme allowed by `descendant_allowlists`
    /// is enabled and beats `RDEncoder`'s `BitPacked` output); patches are kept from child 1 on.
    /// `left_parts` stores the input's validity, which is registered for the fallback path.
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

        let has_left_child = left_parts.nbytes() > 0 && left_parts.len() >= PivcoScheme::MIN_LEN;
        let compress = if has_left_child {
            vec![Child::new(self, 0, left_parts.clone())]
        } else {
            Vec::new()
        };

        // The patches are of the left parts, not of the input, so they stay inside ALP-RD.
        let (patches, keep) = KeptPatches::interior(self, 1, left_parts_patches, exec_ctx)?;
        Ok(Encoded::new(compress, keep, move |children, _| {
            let fallback = with_compressed_validity(left_parts, children.validity()?)?;
            let left_parts = if has_left_child {
                let [compressed_left] = children.compressed::<1>()?;
                if compressed_left.nbytes() < fallback.nbytes()
                    && compressed_left.dtype() == fallback.dtype()
                {
                    compressed_left
                } else {
                    fallback
                }
            } else {
                fallback
            };
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
