// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use super::*;

/// Encode a nullable Vector array and verify roundtrip preserves validity and non-null values.
#[test]
fn nullable_vectors_roundtrip() -> VortexResult<()> {
    let validity = Validity::from_iter([
        true, true, false, true, true, false, true, false, true, true,
    ]);
    let fsl = make_fsl_with_validity(10, 128, 42, validity);
    let ext = make_vector_ext(&fsl);

    let config = TurboQuantConfig {
        bit_width: 3,
        seed: Some(123),
        num_rounds: 4,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = normalize_and_encode(&ext, &config, &mut ctx)?;

    assert_eq!(encoded.len(), 10);
    assert!(encoded.dtype().is_nullable());

    let encoded_validity = encoded.validity()?;
    for i in 0..10 {
        let expected = ![2, 5, 7].contains(&i);
        assert_eq!(
            encoded_validity.is_valid(i)?,
            expected,
            "validity mismatch at row {i}"
        );
    }

    let decoded_ext = encoded.execute::<ExtensionArray>(&mut ctx)?;
    assert_eq!(decoded_ext.len(), 10);

    let decoded_fsl = decoded_ext
        .storage_array()
        .clone()
        .execute::<FixedSizeListArray>(&mut ctx)?;
    let decoded_prim = decoded_fsl
        .elements()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    let decoded_f32 = decoded_prim.as_slice::<f32>();

    let orig_prim = fsl.elements().clone().execute::<PrimitiveArray>(&mut ctx)?;
    let orig_f32 = orig_prim.as_slice::<f32>();

    for row in [0, 1, 3, 4, 6, 8, 9] {
        let orig_vec = &orig_f32[row * 128..(row + 1) * 128];
        let dec_vec = &decoded_f32[row * 128..(row + 1) * 128];
        let norm_sq: f32 = orig_vec.iter().map(|&v| v * v).sum();
        let err_sq: f32 = orig_vec
            .iter()
            .zip(dec_vec.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum();
        assert!(
            err_sq / norm_sq < 0.1,
            "non-null row {row} has excessive reconstruction error"
        );
    }
    Ok(())
}

/// Verify that norms carry the validity: null vectors have null norms.
#[test]
fn nullable_norms_match_validity() -> VortexResult<()> {
    let validity = Validity::from_iter([true, false, true, false, true]);
    let fsl = make_fsl_with_validity(5, 128, 42, validity);
    let ext = make_vector_ext(&fsl);

    let config = TurboQuantConfig {
        bit_width: 2,
        seed: Some(123),
        num_rounds: 3,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = normalize_and_encode(&ext, &config, &mut ctx)?;
    let (_sorf_child, norms_child) = unwrap_l2denorm(&encoded);

    let norms_validity = norms_child.validity()?;
    for i in 0..5 {
        let expected = i % 2 == 0;
        assert_eq!(
            norms_validity.is_valid(i)?,
            expected,
            "norms validity mismatch at row {i}"
        );
    }
    Ok(())
}

/// Verify that L2Norm readthrough works correctly on nullable TurboQuant arrays.
#[test]
fn nullable_l2_norm_readthrough() -> VortexResult<()> {
    use crate::scalar_fns::l2_norm::L2Norm;

    let validity = Validity::from_iter([true, false, true, false, true]);
    let fsl = make_fsl_with_validity(5, 128, 42, validity);
    let ext = make_vector_ext(&fsl);

    let config = TurboQuantConfig {
        bit_width: 3,
        seed: Some(123),
        num_rounds: 3,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = normalize_and_encode(&ext, &config, &mut ctx)?;

    let norm_sfn = L2Norm::try_new_array(encoded, 5)?;
    let norms: PrimitiveArray = norm_sfn.into_array().execute(&mut ctx)?;

    let orig_prim = fsl.elements().clone().execute::<PrimitiveArray>(&mut ctx)?;
    let orig_f32 = orig_prim.as_slice::<f32>();
    for row in 0..5 {
        if row % 2 == 0 {
            assert!(norms.is_valid(row, &mut ctx)?, "row {row} should be valid");
            let expected: f32 = orig_f32[row * 128..(row + 1) * 128]
                .iter()
                .map(|&v| v * v)
                .sum::<f32>()
                .sqrt();
            let actual = norms.as_slice::<f32>()[row];
            assert!(
                (actual - expected).abs() < 1e-5,
                "norm mismatch at valid row {row}: actual={actual}, expected={expected}"
            );
        } else {
            assert!(!norms.is_valid(row, &mut ctx)?, "row {row} should be null");
        }
    }
    Ok(())
}

/// Verify that slicing a nullable TurboQuant array preserves validity.
#[test]
fn nullable_slice_preserves_validity() -> VortexResult<()> {
    let validity = Validity::from_iter([
        true, true, false, true, true, false, true, false, true, true,
    ]);
    let fsl = make_fsl_with_validity(10, 128, 42, validity);
    let ext = make_vector_ext(&fsl);

    let config = TurboQuantConfig {
        bit_width: 3,
        seed: Some(123),
        num_rounds: 2,
    };
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = normalize_and_encode(&ext, &config, &mut ctx)?;

    let sliced = encoded.slice(1..6)?;
    assert_eq!(sliced.len(), 5);

    let sliced_validity = sliced.validity()?;
    let expected = [true, false, true, true, false];
    for (i, &exp) in expected.iter().enumerate() {
        assert_eq!(
            sliced_validity.is_valid(i)?,
            exp,
            "sliced validity mismatch at index {i}"
        );
    }
    Ok(())
}
