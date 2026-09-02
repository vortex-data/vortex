// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests to verify that each float compression scheme produces the expected encoding.

#[cfg(feature = "unstable_encodings")]
use std::f64::consts::TAU;
use std::sync::LazyLock;

use vortex_alp::ALP;
#[cfg(feature = "unstable_encodings")]
use vortex_array::ArrayEq;
#[cfg(feature = "unstable_encodings")]
use vortex_array::EqMode;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Dict;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::PrimitiveBuilder;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::half::f16;
use vortex_array::validity::Validity;
#[cfg(feature = "unstable_encodings")]
use vortex_block_residual::BlockResidual;
#[cfg(feature = "unstable_encodings")]
use vortex_block_residual::OrderedFloat;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
#[cfg(feature = "unstable_encodings")]
use vortex_fastlanes::BitPacked;
#[cfg(feature = "unstable_encodings")]
use vortex_fastlanes::BitPackedArrayExt;
use vortex_float_quant::FloatQuant;
use vortex_float_quant::FloatQuantArraySlotsExt;
use vortex_session::VortexSession;

use crate::BtrBlocksCompressor;
#[cfg(feature = "unstable_encodings")]
use crate::BtrBlocksCompressorBuilder;
use crate::CascadingCompressor;
#[cfg(feature = "unstable_encodings")]
use crate::SchemeExt;
use crate::schemes::float::FloatQuantScheme;
#[cfg(feature = "unstable_encodings")]
use crate::schemes::integer::DeltaScheme;

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

#[cfg(feature = "unstable_encodings")]
#[test]
fn test_quantized_f16_uses_float_quant() -> VortexResult<()> {
    let values = (0_u16..16_384)
        .map(|index| f16::from_bits(0x3c00 | (index.wrapping_mul(7_919) & 0x03f0)))
        .collect::<Vec<_>>();
    let array = PrimitiveArray::from_iter(values).into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = BtrBlocksCompressor::default().compress(&array, &mut ctx)?;

    assert!(compressed.is::<FloatQuant>());
    assert_arrays_eq!(compressed, array, &mut ctx);
    Ok(())
}

#[test]
fn test_f16_secondary_uses_float_quant() -> VortexResult<()> {
    let values = (0_u16..16_384)
        .map(|index| {
            let high_mantissa = index.wrapping_mul(7_919) & 0x03f0;
            f16::from_bits(0x3c00 | high_mantissa | (index & 1))
        })
        .collect::<Vec<_>>();
    let array = PrimitiveArray::from_iter(values).into_array();
    let compressor = CascadingCompressor::new(vec![&FloatQuantScheme]);
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = compressor.compress(&array, &mut ctx)?;

    assert!(compressed.is::<FloatQuant>());
    assert!(compressed.as_::<FloatQuant>().secondary().is_some());
    assert_arrays_eq!(compressed, array, &mut ctx);
    Ok(())
}

#[test]
fn test_constant_compressed() -> VortexResult<()> {
    let values: Vec<f64> = vec![42.5; 100];
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<Constant>());
    Ok(())
}

#[test]
fn test_alp_compressed() -> VortexResult<()> {
    let values: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.01).collect();
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<ALP>());
    Ok(())
}

#[test]
fn test_dict_compressed() -> VortexResult<()> {
    let distinct_values = [1.1, 2.2, 3.3, 4.4, 5.5];
    let values: Vec<f64> = (0..1000)
        .map(|i| distinct_values[i % distinct_values.len()])
        .collect();
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<ALP>());
    assert!(compressed.children()[0].is::<Dict>());
    Ok(())
}

#[test]
fn test_null_dominated_compressed() -> VortexResult<()> {
    let mut builder = PrimitiveBuilder::<f64>::with_capacity(Nullability::Nullable, 100);
    for i in 0..5 {
        builder.append_value(i as f64);
    }
    builder.append_nulls(95);
    let array = builder.finish_into_primitive();
    let btr = BtrBlocksCompressor::default();
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = btr.compress(&array.clone().into_array(), &mut ctx)?;
    assert_arrays_eq!(compressed, array, &mut ctx);
    Ok(())
}

#[cfg(feature = "unstable_encodings")]
#[test]
fn test_widened_f32_uses_float_quant() -> VortexResult<()> {
    let values = (0u32..16_384)
        .map(|index| {
            let mantissa = index.wrapping_mul(7_919) & 0x007f_ffff;
            f64::from(f32::from_bits(0x3f80_0000 | mantissa))
        })
        .collect::<Vec<_>>();
    let array = PrimitiveArray::from_iter(values).into_array();
    let compressed =
        BtrBlocksCompressor::default().compress(&array, &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<FloatQuant>());
    assert!(compressed.as_::<FloatQuant>().secondary().is_none());
    Ok(())
}

#[cfg(feature = "unstable_encodings")]
#[test]
fn test_nonzero_secondary_uses_float_quant() -> VortexResult<()> {
    let values = (0u32..65_536)
        .map(|index| {
            let mantissa = index.wrapping_mul(7_919) & 0x007f_ffff;
            let value = f64::from(f32::from_bits(0x3f80_0000 | mantissa));
            if index % 10 == 0 {
                f64::from_bits(value.to_bits() | 1)
            } else {
                value
            }
        })
        .collect::<Vec<_>>();
    let array = PrimitiveArray::from_iter(values).into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = BtrBlocksCompressor::default().compress(&array, &mut ctx)?;

    assert!(compressed.is::<FloatQuant>());
    let float_quant = compressed.as_::<FloatQuant>();
    let secondary = float_quant
        .secondary()
        .ok_or_else(|| vortex_error::vortex_err!("missing nonzero FloatQuant secondary"))?
        .as_::<BitPacked>();
    assert_eq!(secondary.bit_width(), 1);
    assert_arrays_eq!(compressed, array, &mut ctx);
    Ok(())
}

#[cfg(feature = "unstable_encodings")]
#[test]
fn test_float_quant_ignores_null_payloads() -> VortexResult<()> {
    let values = (0u32..16_384)
        .map(|index| {
            let mantissa = index.wrapping_mul(7_919) & 0x007f_ffff;
            f64::from(f32::from_bits(0x3f80_0000 | mantissa))
        })
        .collect::<Vec<_>>();
    let validity = Validity::from_iter((0..values.len()).map(|index| index % 17 != 0));
    let mut alternate = values.clone();
    for index in (0..alternate.len()).step_by(17) {
        alternate[index] = f64::from_bits(0x3ff0_0000_0000_0001 + index as u64);
    }
    let first = PrimitiveArray::new(Buffer::copy_from(&values), validity.clone()).into_array();
    let second = PrimitiveArray::new(Buffer::copy_from(&alternate), validity).into_array();
    let compressor = BtrBlocksCompressor::default();
    let first = compressor.compress(&first, &mut SESSION.create_execution_ctx())?;
    let second = compressor.compress(&second, &mut SESSION.create_execution_ctx())?;

    assert!(first.is::<FloatQuant>());
    assert!(second.is::<FloatQuant>());
    assert!(first.array_eq(&second, EqMode::Value));
    Ok(())
}

#[cfg(feature = "unstable_encodings")]
#[test]
fn test_f32_does_not_use_float_quant() -> VortexResult<()> {
    let values = (0u32..16_384)
        .map(|index| {
            let mantissa = index.wrapping_mul(7_919) & 0x007f_ffff;
            f32::from_bits(0x3f80_0000 | mantissa)
        })
        .collect::<Vec<_>>();
    let array = PrimitiveArray::from_iter(values).into_array();
    let compressed =
        BtrBlocksCompressor::default().compress(&array, &mut SESSION.create_execution_ctx())?;
    assert!(!compressed.is::<FloatQuant>());
    Ok(())
}

#[cfg(feature = "unstable_encodings")]
#[test]
fn test_quantized_f32_uses_float_quant() -> VortexResult<()> {
    let values = (0_u32..65_536)
        .map(|index| {
            let mantissa = (index.wrapping_mul(7_919) & 0x7fff) << 8;
            f32::from_bits(0x3f80_0000 | mantissa)
        })
        .collect::<Vec<_>>();
    let array = PrimitiveArray::from_iter(values).into_array();
    let compressed =
        BtrBlocksCompressor::default().compress(&array, &mut SESSION.create_execution_ctx())?;

    assert!(compressed.is::<FloatQuant>());
    assert!(compressed.as_::<FloatQuant>().secondary().is_none());
    Ok(())
}

#[cfg(feature = "unstable_encodings")]
#[test]
fn test_repeated_f64_prefers_existing_scheme() -> VortexResult<()> {
    let values = (0u32..16_384)
        .map(|index| f64::from(index % 8))
        .collect::<Vec<_>>();
    let array = PrimitiveArray::from_iter(values).into_array();
    let compressed =
        BtrBlocksCompressor::default().compress(&array, &mut SESSION.create_execution_ctx())?;
    assert!(!compressed.is::<FloatQuant>());
    Ok(())
}

#[cfg(feature = "unstable_encodings")]
#[test]
fn test_random_walk_uses_ordered_block_residual() -> VortexResult<()> {
    fn uniform(state: &mut u64) -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        ((*state >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    }

    let mut state = 0x4d59_5df4_d0f3_3173_u64;
    let mut value = 0.0_f64;
    let values = (0usize..65_536)
        .map(|_| {
            let radius = (-2.0 * uniform(&mut state).ln()).sqrt();
            let normal = radius * (TAU * uniform(&mut state)).cos();
            value += normal * 0.01;
            value
        })
        .collect::<Vec<_>>();
    let array = PrimitiveArray::from_iter(values).into_array();
    let compressed =
        BtrBlocksCompressor::default().compress(&array, &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<OrderedFloat>());
    assert!(compressed.children()[0].is::<BlockResidual>());
    Ok(())
}

#[cfg(feature = "unstable_encodings")]
#[test]
fn test_f32_random_walk_uses_ordered_block_residual() -> VortexResult<()> {
    let values = (0_u32..65_536)
        .map(|index| {
            let block = index / 1_024;
            let residual = index.wrapping_mul(7_919) % 1_024;
            f32::from_bits(0x3f80_0000 + (block * 0x1_0000) + residual)
        })
        .collect::<Vec<_>>();
    let array = PrimitiveArray::from_iter(values).into_array();
    let compressed =
        BtrBlocksCompressor::default().compress(&array, &mut SESSION.create_execution_ctx())?;

    assert!(
        compressed.is::<OrderedFloat>(),
        "expected OrderedFloat, got tree:\n{}",
        compressed.display_tree()
    );
    assert!(compressed.children()[0].is::<BlockResidual>());
    Ok(())
}

#[cfg(feature = "unstable_encodings")]
#[test]
fn test_block_residual_composes_with_alp() -> VortexResult<()> {
    let values = (0..65_536_usize).map(|index| {
        let block = index / 1_024;
        let residual = index.wrapping_mul(2_654_435_761) % 1_024;
        (block * 1_000_000 + residual) as f64
    });
    let array = PrimitiveArray::from_iter(values).into_array();
    #[cfg(not(feature = "unstable_encodings"))]
    let compressor = BtrBlocksCompressor::default();
    #[cfg(feature = "unstable_encodings")]
    let compressor = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([DeltaScheme::default().id()])
        .build();
    let compressed = compressor.compress(&array, &mut SESSION.create_execution_ctx())?;

    assert!(
        compressed.is::<ALP>(),
        "expected ALP, got tree:\n{}",
        compressed.display_tree()
    );
    assert!(
        compressed
            .children()
            .iter()
            .any(|child| child.is::<BlockResidual>()),
        "expected a BlockResidual child:\n{}",
        compressed.display_tree()
    );
    Ok(())
}
