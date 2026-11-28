// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ALPRD (ALP with Real Double) encoding variant.

use vortex_alp::ALPRDArrayExt;
use vortex_alp::ALPRDArrayOwnedExt;
use vortex_alp::ALPRDDataParts;
use vortex_alp::alp_rd_encode;
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
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedSlots;
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
        let alp_rd = alp_rd_encode(data.array_as_primitive(), exec_ctx)?;
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

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_alp::ALPRD;
    use vortex_alp::ALPRDArrayExt;
    use vortex_alp::ALPRDFloat;
    use vortex_alp::alp_rd_encode;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::NativePType;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_compressor::CascadingCompressor;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::schemes::float::ALPRDScheme;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        vortex_alp::initialize(&session);
        vortex_fastlanes::initialize(&session);
        session
    });

    fn dirty_values_f32() -> Vec<f32> {
        let mut values = vec![
            f32::NAN,
            f32::from_bits(0xdead_beef),
            1e30_f32,
            -1e30_f32,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        for prefix in 0u32..12 {
            values.push(f32::from_bits((0x2000 + prefix * 0x0333) << 16 | 0x1234));
        }
        values
    }

    fn dirty_values_f64() -> Vec<f64> {
        let mut values = vec![
            f64::NAN,
            f64::from_bits(0xdead_beef_cafe_babe),
            1e300_f64,
            -1e300_f64,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        for prefix in 0u64..12 {
            values.push(f64::from_bits(
                (0x2000 + prefix * 0x0333) << 48 | 0x1234_5678_9abc,
            ));
        }
        values
    }

    #[rstest]
    #[case::f32(
        (0..64).map(|idx| 1.0_f32 + (idx as f32) * 1e-4).collect::<Vec<_>>(),
        dirty_values_f32(),
    )]
    #[case::f64(
        (0..64).map(|idx| 1.0_f64 + (idx as f64) * 1e-6).collect::<Vec<_>>(),
        dirty_values_f64(),
    )]
    fn test_alprd_scheme_ignores_dirty_nulls<T>(
        #[case] clean_values: Vec<T>,
        #[case] dirty_values: Vec<T>,
    ) -> VortexResult<()>
    where
        T: ALPRDFloat + NativePType,
    {
        let mut ctx = SESSION.create_execution_ctx();
        let len = 10_000;
        let mut raw = Vec::with_capacity(len);
        let mut valid_only = Vec::with_capacity(len);
        let mut is_valid = Vec::with_capacity(len);

        for row in 0..len {
            if row % 10 == 0 {
                raw.push(dirty_values[row % dirty_values.len()]);
                is_valid.push(false);
            } else {
                let value = clean_values[row % clean_values.len()];
                raw.push(value);
                valid_only.push(value);
                is_valid.push(true);
            }
        }

        let array = PrimitiveArray::new(Buffer::from(raw), Validity::from_iter(is_valid));
        let clean = PrimitiveArray::new(Buffer::from(valid_only), Validity::NonNullable);
        let expected_clean = alp_rd_encode(clean.as_view(), &mut ctx)?;

        let compressor = CascadingCompressor::new(vec![&ALPRDScheme]);
        let compressed = compressor.compress(&array.clone().into_array(), &mut ctx)?;
        let alprd = compressed
            .as_opt::<ALPRD>()
            .expect("expected ALPRDScheme to compress floats with dirty nulls");
        assert!(alprd.left_parts_patches().is_none());
        assert_eq!(alprd.right_bit_width(), expected_clean.right_bit_width());
        assert_eq!(
            alprd.left_parts_dictionary().as_slice(),
            expected_clean.left_parts_dictionary().as_slice()
        );
        assert_arrays_eq!(compressed, array, &mut ctx);

        Ok(())
    }
}
