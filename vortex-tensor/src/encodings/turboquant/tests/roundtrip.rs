// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Extension;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use super::*;
use crate::encodings::turboquant::turboquant_encode_unchecked;
use crate::scalar_fns::l2_denorm::normalize_as_l2_denorm;

#[rstest]
#[case(128, 1)]
#[case(128, 2)]
#[case(128, 3)]
#[case(128, 4)]
#[case(128, 6)]
#[case(128, 8)]
#[case(256, 2)]
fn roundtrip(#[case] dim: usize, #[case] bit_width: u8) -> VortexResult<()> {
    let fsl = make_fsl(10, dim, 42);
    let config = TurboQuantConfig {
        bit_width,
        seed: Some(123),
        num_rounds: 3,
    };
    let (original, decoded) = encode_decode(&fsl, &config)?;
    assert_eq!(decoded.len(), original.len());
    Ok(())
}

#[rstest]
#[case(128, 1)]
#[case(128, 2)]
#[case(128, 3)]
#[case(128, 4)]
#[case(256, 2)]
#[case(256, 4)]
fn mse_within_theoretical_bound(#[case] dim: usize, #[case] bit_width: u8) -> VortexResult<()> {
    let num_rows = 200;
    let fsl = make_fsl(num_rows, dim, 42);
    let config = TurboQuantConfig {
        bit_width,
        seed: Some(123),
        num_rounds: 3,
    };
    let (original, decoded) = encode_decode(&fsl, &config)?;

    let normalized_mse = per_vector_normalized_mse(&original, &decoded, dim, num_rows);
    let bound = theoretical_mse_bound(bit_width);

    assert!(
        normalized_mse < bound,
        "Normalized MSE {normalized_mse:.6} exceeds bound {bound:.6} \
             for dim={dim}, bits={bit_width}",
    );
    Ok(())
}

#[rstest]
#[case(128, 6)]
#[case(128, 8)]
#[case(256, 6)]
#[case(256, 8)]
fn high_bitwidth_mse_is_small(#[case] dim: usize, #[case] bit_width: u8) -> VortexResult<()> {
    let num_rows = 200;
    let fsl = make_fsl(num_rows, dim, 42);

    let config_4bit = TurboQuantConfig {
        bit_width: 4,
        seed: Some(123),
        num_rounds: 3,
    };
    let (original_4, decoded_4) = encode_decode(&fsl, &config_4bit)?;
    let mse_4bit = per_vector_normalized_mse(&original_4, &decoded_4, dim, num_rows);

    let config = TurboQuantConfig {
        bit_width,
        seed: Some(123),
        num_rounds: 3,
    };
    let (original, decoded) = encode_decode(&fsl, &config)?;
    let mse = per_vector_normalized_mse(&original, &decoded, dim, num_rows);

    assert!(
        mse < mse_4bit,
        "{bit_width}-bit MSE ({mse:.6}) should be < 4-bit MSE ({mse_4bit:.6})"
    );
    assert!(mse < 0.01, "{bit_width}-bit MSE ({mse:.6}) should be < 1%");
    Ok(())
}

#[test]
fn mse_decreases_with_bits() -> VortexResult<()> {
    let dim = 128;
    let num_rows = 50;
    let fsl = make_fsl(num_rows, dim, 99);

    let mut prev_mse = f32::MAX;
    for bit_width in 1..=8u8 {
        let config = TurboQuantConfig {
            bit_width,
            seed: Some(123),
            num_rounds: 3,
        };
        let (original, decoded) = encode_decode(&fsl, &config)?;
        let mse = per_vector_normalized_mse(&original, &decoded, dim, num_rows);
        assert!(
            mse <= prev_mse * 1.01,
            "MSE should decrease: {bit_width}-bit={mse:.6} > prev={prev_mse:.6}"
        );
        prev_mse = mse;
    }
    Ok(())
}

#[rstest]
#[case(0)]
#[case(1)]
fn roundtrip_edge_cases(#[case] num_rows: usize) -> VortexResult<()> {
    let fsl = make_fsl(num_rows, 128, 42);
    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width: 2,
        seed: Some(123),
        num_rounds: 3,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext, &config, &mut ctx)?;
    let decoded = encoded.execute::<ExtensionArray>(&mut ctx)?;
    assert_eq!(decoded.len(), num_rows);
    Ok(())
}

#[rstest]
#[case(1)]
#[case(64)]
#[case(127)]
fn rejects_dimension_below_128(#[case] dim: usize) {
    let elements = PrimitiveArray::new::<f32>(
        BufferMut::from_iter((0..dim).map(|i| i as f32 + 1.0)).freeze(),
        Validity::NonNullable,
    );
    let fsl = FixedSizeListArray::try_new(
        elements.into_array(),
        dim.try_into().expect("dim fits u32"),
        Validity::NonNullable,
        1,
    )
    .unwrap();
    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width: 2,
        seed: Some(0),
        num_rounds: 3,
    };
    let mut ctx = SESSION.create_execution_ctx();
    assert!(turboquant_encode(ext, &config, &mut ctx).is_err());
}

#[rstest]
#[case(0)]
#[case(9)]
fn rejects_invalid_bit_width(#[case] bit_width: u8) {
    let fsl = make_fsl(10, 128, 42);
    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width,
        seed: Some(0),
        num_rounds: 3,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let normalized = normalize_as_l2_denorm(ext, &mut ctx)
        .unwrap()
        .child_at(0)
        .clone();
    let normalized_ext = normalized
        .as_opt::<Extension>()
        .expect("normalized child should be Extension");
    assert!(unsafe { turboquant_encode_unchecked(normalized_ext, &config, &mut ctx) }.is_err());
}

#[test]
fn all_zero_vectors_roundtrip() -> VortexResult<()> {
    let num_rows = 10;
    let dim = 128;
    let buf = BufferMut::<f32>::full(0.0f32, num_rows * dim);
    let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable);
    let fsl = FixedSizeListArray::try_new(
        elements.into_array(),
        dim.try_into()
            .expect("somehow got dimension greater than u32::MAX"),
        Validity::NonNullable,
        num_rows,
    )?;

    let config = TurboQuantConfig {
        bit_width: 3,
        seed: Some(42),
        num_rounds: 3,
    };
    let (original, decoded) = encode_decode(&fsl, &config)?;
    for (i, (&o, &d)) in original.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(o, 0.0, "original[{i}] not zero");
        assert_eq!(d, 0.0, "decoded[{i}] not zero for all-zero input");
    }
    Ok(())
}

/// Roundtrip at large embedding dimensions.
#[rstest]
#[case(768, 4)]
#[case(1024, 5)]
fn large_dimension_roundtrip(#[case] dim: usize, #[case] bit_width: u8) -> VortexResult<()> {
    let num_rows = 10;
    let fsl = make_fsl(num_rows, dim, 42);
    let config = TurboQuantConfig {
        bit_width,
        seed: Some(123),
        num_rounds: 3,
    };
    let (original, decoded) = encode_decode(&fsl, &config)?;
    assert_eq!(decoded.len(), original.len());

    let normalized_mse = per_vector_normalized_mse(&original, &decoded, dim, num_rows);
    // 2x slack for the SRHT-vs-Haar gap.
    let bound = 2.0 * theoretical_mse_bound(bit_width);
    assert!(
        normalized_mse < bound,
        "Normalized MSE {normalized_mse:.6} exceeds 2x bound {bound:.6} for dim={dim}, bits={bit_width}",
    );
    Ok(())
}

/// Verify that f64 input is accepted and encoded.
#[test]
fn f64_input_encodes_successfully() -> VortexResult<()> {
    let num_rows = 10;
    let dim = 128;
    let mut rng = StdRng::seed_from_u64(99);
    let normal = Normal::new(0.0f64, 1.0).unwrap();

    let mut buf = BufferMut::<f64>::with_capacity(num_rows * dim);
    for _ in 0..(num_rows * dim) {
        buf.push(normal.sample(&mut rng));
    }
    let elements = PrimitiveArray::new::<f64>(buf.freeze(), Validity::NonNullable);
    let fsl = FixedSizeListArray::try_new(
        elements.into_array(),
        dim.try_into().unwrap(),
        Validity::NonNullable,
        num_rows,
    )?;

    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width: 3,
        seed: Some(42),
        num_rounds: 3,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext, &config, &mut ctx)?;
    let (_sorf_child, norms_child) = unwrap_l2denorm(&encoded);
    assert_eq!(norms_child.len(), num_rows);
    Ok(())
}

/// Verify that f16 input is accepted and encoded.
#[test]
fn f16_input_encodes_successfully() -> VortexResult<()> {
    let num_rows = 10;
    let dim = 128;
    let mut rng = StdRng::seed_from_u64(99);
    let normal = Normal::new(0.0f32, 1.0).unwrap();

    let mut buf = BufferMut::<half::f16>::with_capacity(num_rows * dim);
    for _ in 0..(num_rows * dim) {
        buf.push(half::f16::from_f32(normal.sample(&mut rng)));
    }
    let elements = PrimitiveArray::new::<half::f16>(buf.freeze(), Validity::NonNullable);
    let fsl = FixedSizeListArray::try_new(
        elements.into_array(),
        dim.try_into().unwrap(),
        Validity::NonNullable,
        num_rows,
    )?;

    let ext = make_vector_ext(&fsl);
    let config = TurboQuantConfig {
        bit_width: 3,
        seed: Some(42),
        num_rounds: 3,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = turboquant_encode(ext, &config, &mut ctx)?;
    let (_sorf_child, norms_child) = unwrap_l2denorm(&encoded);
    assert_eq!(norms_child.len(), num_rows);

    let decoded_ext = encoded.execute::<ExtensionArray>(&mut ctx)?;
    let decoded_fsl = decoded_ext
        .storage_array()
        .clone()
        .execute::<FixedSizeListArray>(&mut ctx)?;
    assert_eq!(decoded_fsl.len(), num_rows);
    Ok(())
}
