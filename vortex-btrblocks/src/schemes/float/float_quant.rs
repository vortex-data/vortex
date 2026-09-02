// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lossless float quantization with a fixed frame-of-reference child.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::PType;
use vortex_array::dtype::half::f16;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_array::vtable::validity_to_child;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::FL_CHUNK_SIZE;
use vortex_fastlanes::FoR;
use vortex_fastlanes::bitpack_compress::bitpack_primitive_map;
use vortex_fastlanes::bitpack_compress::bitpack_primitive_map_pair;
use vortex_float_quant::FloatQuant;
use vortex_float_quant::FloatQuantAnalysis;
use vortex_float_quant::analyze_float_quant;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::normalize_null_values;
use crate::schemes::sample_primitive_one_percent;

// Food needs a factor above 1.078 to prevent its sample from displacing a smaller ALP tree.
const SELECTION_COST_FACTOR: f64 = 1.10;

/// FloatQuant split with a fixed frame-of-reference primary child.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FloatQuantScheme;

impl Scheme for FloatQuantScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.float_quant"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_float()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![FloatQuant.id(), FoR.id(), BitPacked.id()]
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        if compress_ctx.finished_cascading() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            |_compressor, data, _best_so_far, _compress_ctx, exec_ctx| {
                let sample = sample_primitive_one_percent(data.array_as_primitive(), exec_ctx)?;
                estimate_float_quant_sample(&sample)
            },
        )))
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let source = data.array_as_primitive();
        let primitive = normalize_null_values(source, exec_ctx)?;
        let Some(analysis) = analyze_float_quant(primitive.as_view()) else {
            return Ok(source.array().clone());
        };
        encode_float_quant(primitive.as_view(), analysis)
    }
}

fn estimate_float_quant_sample(sample: &PrimitiveArray) -> VortexResult<EstimateVerdict> {
    let Some(analysis) = analyze_float_quant(sample.as_view()) else {
        return Ok(EstimateVerdict::Skip);
    };
    // A constant sample does not prove that the full array is constant.
    if analysis.primary_bit_width == 0 && analysis.secondary_bit_width == 0 {
        return Ok(EstimateVerdict::Skip);
    }

    let before_nbytes = sample.nbytes();
    let after_nbytes = estimate_float_quant_nbytes(sample, analysis)?;
    if after_nbytes == 0 || after_nbytes >= before_nbytes {
        return Ok(EstimateVerdict::Skip);
    }

    let adjusted_ratio = before_nbytes as f64 / after_nbytes as f64 / SELECTION_COST_FACTOR;
    if adjusted_ratio <= 1.0 {
        return Ok(EstimateVerdict::Skip);
    }

    Ok(EstimateVerdict::Ratio(adjusted_ratio))
}

fn estimate_float_quant_nbytes(
    sample: &PrimitiveArray,
    analysis: FloatQuantAnalysis,
) -> VortexResult<u64> {
    let packed_chunks = u64::try_from(sample.len().div_ceil(FL_CHUNK_SIZE))?;
    let bytes_per_bit = u64::try_from(FL_CHUNK_SIZE / 8)?;
    let packed_bit_width =
        u64::from(analysis.primary_bit_width) + u64::from(analysis.secondary_bit_width);
    let packed_nbytes = packed_chunks * bytes_per_bit * packed_bit_width;
    let validity_nbytes = validity_to_child(&sample.validity()?, sample.len())
        .map(|validity| validity.nbytes())
        .unwrap_or(0);
    Ok(packed_nbytes + validity_nbytes)
}

fn encode_float_quant(
    primitive: vortex_array::ArrayView<'_, Primitive>,
    analysis: FloatQuantAnalysis,
) -> VortexResult<ArrayRef> {
    // The ordered transform complements the low bits of negative values. Decode complements the
    // secondary bits again, so both signs store the original low bits.
    let (primary_packed, secondary_packed, latent_ptype, reference) = match primitive.ptype() {
        PType::F16 => {
            let primary_min = u16::try_from(analysis.primary_min)?;
            let values = primitive.as_slice::<f16>();
            let (primary, secondary) = if analysis.secondary_bit_width == 0 {
                (
                    bitpack_primitive_map(values, analysis.primary_bit_width, |value| {
                        (ordered_u16(value.to_bits()) >> analysis.k) - primary_min
                    }),
                    None,
                )
            } else {
                let low_mask = (1_u16 << analysis.k) - 1;
                let (primary, secondary) = bitpack_primitive_map_pair(
                    values,
                    analysis.primary_bit_width,
                    analysis.secondary_bit_width,
                    |value| {
                        let bits = value.to_bits();
                        (
                            (ordered_u16(bits) >> analysis.k) - primary_min,
                            bits & low_mask,
                        )
                    },
                );
                (primary, Some(secondary))
            };
            (
                primary.into_byte_buffer(),
                secondary.map(|packed| packed.into_byte_buffer()),
                PType::U16,
                Scalar::from(primary_min),
            )
        }
        PType::F32 => {
            let primary_min = u32::try_from(analysis.primary_min)?;
            let values = primitive.as_slice::<f32>();
            let (primary, secondary) = if analysis.secondary_bit_width == 0 {
                (
                    bitpack_primitive_map(values, analysis.primary_bit_width, |value| {
                        (ordered_u32(value.to_bits()) >> analysis.k) - primary_min
                    }),
                    None,
                )
            } else {
                let low_mask = (1_u32 << analysis.k) - 1;
                let (primary, secondary) = bitpack_primitive_map_pair(
                    values,
                    analysis.primary_bit_width,
                    analysis.secondary_bit_width,
                    |value| {
                        let bits = value.to_bits();
                        (
                            (ordered_u32(bits) >> analysis.k) - primary_min,
                            bits & low_mask,
                        )
                    },
                );
                (primary, Some(secondary))
            };
            (
                primary.into_byte_buffer(),
                secondary.map(|packed| packed.into_byte_buffer()),
                PType::U32,
                Scalar::from(primary_min),
            )
        }
        PType::F64 => {
            let values = primitive.as_slice::<f64>();
            let (primary, secondary) = if analysis.secondary_bit_width == 0 {
                (
                    bitpack_primitive_map(values, analysis.primary_bit_width, |value| {
                        (ordered_u64(value.to_bits()) >> analysis.k) - analysis.primary_min
                    }),
                    None,
                )
            } else {
                let low_mask = (1_u64 << analysis.k) - 1;
                let (primary, secondary) = bitpack_primitive_map_pair(
                    values,
                    analysis.primary_bit_width,
                    analysis.secondary_bit_width,
                    |value| {
                        let bits = value.to_bits();
                        (
                            (ordered_u64(bits) >> analysis.k) - analysis.primary_min,
                            bits & low_mask,
                        )
                    },
                );
                (primary, Some(secondary))
            };
            (
                primary.into_byte_buffer(),
                secondary.map(|packed| packed.into_byte_buffer()),
                PType::U64,
                Scalar::from(analysis.primary_min),
            )
        }
        _ => unreachable!(),
    };
    let compressed_primary = BitPacked::try_new(
        BufferHandle::new_host(primary_packed),
        latent_ptype,
        primitive.validity()?,
        None,
        analysis.primary_bit_width,
        primitive.len(),
        0,
    )?
    .into_array();
    let compressed_primary = FoR::try_new(compressed_primary, reference)?.into_array();
    let compressed_secondary = secondary_packed
        .map(|packed| {
            BitPacked::try_new(
                BufferHandle::new_host(packed),
                latent_ptype,
                Validity::NonNullable,
                None,
                analysis.secondary_bit_width,
                primitive.len(),
                0,
            )
            .map(IntoArray::into_array)
        })
        .transpose()?;
    Ok(FloatQuant::try_new(
        compressed_primary,
        compressed_secondary,
        primitive.ptype(),
        analysis.k,
    )?
    .into_array())
}

#[inline]
fn ordered_u16(bits: u16) -> u16 {
    if bits & (1_u16 << 15) == 0 {
        bits ^ (1_u16 << 15)
    } else {
        !bits
    }
}

#[inline]
fn ordered_u32(bits: u32) -> u32 {
    if bits & (1_u32 << 31) == 0 {
        bits ^ (1_u32 << 31)
    } else {
        !bits
    }
}

#[inline]
fn ordered_u64(bits: u64) -> u64 {
    if bits & (1_u64 << 63) == 0 {
        bits ^ (1_u64 << 63)
    } else {
        !bits
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::half::f16;
    use vortex_compressor::scheme::EstimateVerdict;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::analyze_float_quant;
    use super::encode_float_quant;
    use super::estimate_float_quant_nbytes;
    use super::estimate_float_quant_sample;

    #[test]
    fn constant_sample_is_not_evidence_for_float_quant() -> VortexResult<()> {
        let sample = PrimitiveArray::from_iter(vec![1.0_f64; 1_024]);

        assert!(matches!(
            estimate_float_quant_sample(&sample)?,
            EstimateVerdict::Skip
        ));
        Ok(())
    }

    #[test]
    fn estimated_nbytes_matches_encoded_tree() -> VortexResult<()> {
        let f16_values = PrimitiveArray::from_iter((0..2_050).map(|index| {
            let high_bits = u16::try_from(index).unwrap_or_default().wrapping_mul(17) & 0x03f0;
            let low_bit = u16::from(index % 10 == 0);
            f16::from_bits(0x3c00 | high_bits | low_bit)
        }));
        let f32_values = PrimitiveArray::from_option_iter((0..2_050).map(|index| {
            let high_bits = (index as u32).wrapping_mul(7_919) & 0x007f_ff00;
            let low_bit = u32::from(index % 10 == 0);
            (index % 17 != 0).then_some(f32::from_bits(0x3f80_0000 | high_bits | low_bit))
        }));
        let f64_values = PrimitiveArray::from_iter((0..2_050).map(|index| {
            let high_bits = ((index as u64).wrapping_mul(7_919) << 29) & 0x000f_ffff_ffff_ff00;
            let low_bit = u64::from(index % 10 == 0);
            f64::from_bits(0x3ff0_0000_0000_0000 | high_bits | low_bit)
        }));

        for values in [f16_values, f32_values, f64_values] {
            let analysis = analyze_float_quant(values.as_view())
                .ok_or_else(|| vortex_err!("FloatQuant test input did not produce an analysis"))?;
            let expected = encode_float_quant(values.as_view(), analysis)?.nbytes();
            assert_eq!(estimate_float_quant_nbytes(&values, analysis)?, expected);
        }
        Ok(())
    }

    #[test]
    fn near_miss_sample_is_rejected() -> VortexResult<()> {
        let f32_values = PrimitiveArray::from_iter((0..2_048).map(|index| {
            let scrambled = (index as u32).wrapping_mul(2_654_435_761);
            let sign = (scrambled & 1) << 31;
            let exponent = ((scrambled >> 1) % 254 + 1) << 23;
            let mantissa = scrambled & 0x007f_fffc;
            f32::from_bits(sign | exponent | mantissa)
        }));
        let f64_values = PrimitiveArray::from_iter((0..2_048).map(|index| {
            let scrambled = (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let sign = (scrambled & 1) << 63;
            let exponent = ((scrambled >> 1) % 2_046 + 1) << 52;
            let mantissa = scrambled & 0x000f_ffff_ffff_fffc;
            f64::from_bits(sign | exponent | mantissa)
        }));

        for values in [f32_values, f64_values] {
            assert!(analyze_float_quant(values.as_view()).is_some());
            assert!(matches!(
                estimate_float_quant_sample(&values)?,
                EstimateVerdict::Skip
            ));
        }
        Ok(())
    }

    #[test]
    fn nonzero_secondary_round_trips_negative_values() -> VortexResult<()> {
        let values = PrimitiveArray::from_iter((0_u32..2_050).flat_map(|index| {
            let high_bits = index.wrapping_mul(7_919) & 0x007f_ff00;
            let low_bits = index % 7;
            let positive = f32::from_bits(0x3f80_0000 | high_bits | low_bits);
            [positive, -positive]
        }));
        let analysis = analyze_float_quant(values.as_view())
            .ok_or_else(|| vortex_err!("FloatQuant test input did not produce an analysis"))?;
        assert!(analysis.secondary_bit_width > 0);
        let encoded = encode_float_quant(values.as_view(), analysis)?;
        assert_arrays_eq!(
            encoded,
            values.into_array(),
            &mut array_session().create_execution_ctx()
        );
        Ok(())
    }
}
