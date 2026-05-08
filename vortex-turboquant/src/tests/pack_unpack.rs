// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use super::execute_tq_pack;
use super::execute_tq_unpack;
use super::f32_vector_array;
use super::test_session;
use super::turboquant_storage;
use super::vector_array;
use super::vector_element_ptype;
use super::vector_validity;
use super::vector_values_f32;
use crate::TurboQuantConfig;
use crate::centroids::compute_or_get_centroids;
use crate::vector::normalize::tq_normalize_as_l2_denorm;

#[rstest]
#[case::zero_bits(0, 42, 3)]
#[case::too_many_bits(9, 42, 3)]
#[case::zero_rounds(2, 42, 0)]
fn config_rejects_invalid_values(#[case] bit_width: u8, #[case] seed: u64, #[case] num_rounds: u8) {
    assert!(TurboQuantConfig::try_new(bit_width, seed, num_rounds).is_err());
}

#[test]
fn pack_rejects_non_vector_input() {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let input = PrimitiveArray::new::<f32>(Buffer::copy_from([1.0, 2.0]), Validity::NonNullable)
        .into_array();

    assert!(execute_tq_pack(input, &TurboQuantConfig::default(), &mut ctx).is_err());
}

#[test]
fn pack_rejects_small_dimensions() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let input = f32_vector_array(127, 1, 1.0, Validity::NonNullable)?;

    assert!(execute_tq_pack(input, &TurboQuantConfig::default(), &mut ctx).is_err());
    Ok(())
}

#[test]
fn pack_rejects_padded_dimension_overflow() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let input = vector_array::<f32>(2_147_483_649, &[], Validity::NonNullable)?;

    assert!(execute_tq_pack(input, &TurboQuantConfig::default(), &mut ctx).is_err());
    Ok(())
}

#[test]
fn centroid_cache_is_deterministic() -> VortexResult<()> {
    let first = compute_or_get_centroids(128, 3)?;
    let second = compute_or_get_centroids(128, 3)?;

    assert_eq!(first.as_slice(), second.as_slice());
    Ok(())
}

#[test]
fn pack_unpack_empty_vectors() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let input = vector_array::<f32>(128, &[], Validity::NonNullable)?;

    let packed = execute_tq_pack(input, &TurboQuantConfig::default(), &mut ctx)?;
    let unpacked = execute_tq_unpack(packed, &TurboQuantConfig::default(), &mut ctx)?;

    assert!(unpacked.is_empty());
    Ok(())
}

#[test]
fn pack_stores_norms_and_struct_validity() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let validity = Validity::from_iter([true, false, true]);
    let input = f32_vector_array(128, 3, 0.25, validity)?;

    let config = TurboQuantConfig::try_new(3, 1, 2)?;
    let packed = execute_tq_pack(input, &config, &mut ctx)?;
    let storage = turboquant_storage(packed, &mut ctx)?;
    let mask = storage.struct_validity().execute_mask(3, &mut ctx)?;
    let norms: PrimitiveArray = storage
        .unmasked_field_by_name("norms")?
        .clone()
        .execute(&mut ctx)?;
    let codes: FixedSizeListArray = storage
        .unmasked_field_by_name("codes")?
        .clone()
        .execute(&mut ctx)?;

    assert!(mask.value(0));
    assert!(!mask.value(1));
    assert!(mask.value(2));
    assert_eq!(norms.validity()?.nullability(), Nullability::Nullable);
    assert_eq!(codes.validity()?.nullability(), Nullability::Nullable);

    let norms_validity = norms.validity()?.execute_mask(3, &mut ctx)?;
    let codes_validity = codes.validity()?.execute_mask(3, &mut ctx)?;
    assert!(norms_validity.value(0));
    assert!(!norms_validity.value(1));
    assert!(norms_validity.value(2));
    assert!(codes_validity.value(0));
    assert!(!codes_validity.value(1));
    assert!(codes_validity.value(2));

    let codes_values: PrimitiveArray = codes.elements().clone().execute(&mut ctx)?;
    assert!(
        codes_values.as_slice::<u8>()[128..256]
            .iter()
            .all(|&code| code == 0)
    );
    Ok(())
}

#[test]
fn normalize_as_l2_denorm_preserves_child_validity() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let mut values = vec![0.0f32; 3 * 128];
    values[0] = 3.0;
    values[1] = 4.0;
    values[128..256].fill(13.0);
    values[256] = 1.0;
    let input = vector_array(128, &values, Validity::from_iter([true, false, true]))?;

    let l2_denorm = tq_normalize_as_l2_denorm(input, &mut ctx)?;
    let normalized = l2_denorm.child_at(0).clone();
    let norms = l2_denorm.child_at(1).clone();

    let normalized_ext: ExtensionArray = normalized.execute(&mut ctx)?;
    let normalized_fsl: FixedSizeListArray =
        normalized_ext.storage_array().clone().execute(&mut ctx)?;
    let normalized_values: PrimitiveArray = normalized_fsl.elements().clone().execute(&mut ctx)?;
    let norms: PrimitiveArray = norms.execute(&mut ctx)?;
    let normalized_validity = normalized_fsl.validity()?.execute_mask(3, &mut ctx)?;
    let norms_validity = norms.validity()?.execute_mask(3, &mut ctx)?;

    assert!(normalized_validity.value(0));
    assert!(!normalized_validity.value(1));
    assert!(normalized_validity.value(2));
    assert!(norms_validity.value(0));
    assert!(!norms_validity.value(1));
    assert!(norms_validity.value(2));
    assert_eq!(norms.validity()?.nullability(), Nullability::Nullable);
    assert_eq!(norms.as_slice::<f32>()[0], 5.0);
    assert!(
        normalized_values.as_slice::<f32>()[128..256]
            .iter()
            .all(|&value| value == 0.0)
    );
    Ok(())
}

#[test]
fn pack_unpack_preserves_nulls_and_zero_norm_rows() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let mut values = vec![0.0f32; 3 * 128];
    values[0] = 3.0;
    values[1] = 4.0;
    values[256] = 1.0;
    values[257] = -1.0;
    let input = vector_array(128, &values, Validity::from_iter([true, true, false]))?;

    let packed = execute_tq_pack(input, &TurboQuantConfig::default(), &mut ctx)?;
    let unpacked = execute_tq_unpack(packed, &TurboQuantConfig::default(), &mut ctx)?;
    let output = vector_values_f32(unpacked.clone(), &mut ctx)?;
    let validity = vector_validity(unpacked, &mut ctx)?.execute_mask(3, &mut ctx)?;

    assert!(validity.value(0));
    assert!(validity.value(1));
    assert!(!validity.value(2));
    assert!(output[128..256].iter().all(|&v| v == 0.0));
    Ok(())
}

#[rstest]
#[case::f16(PType::F16)]
#[case::f64(PType::F64)]
fn pack_unpack_supports_non_f32_inputs(#[case] ptype: PType) -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let config = TurboQuantConfig::try_new(3, 42, 3)?;

    match ptype {
        PType::F16 => {
            let values = (0..2 * 128)
                .map(|i| half::f16::from_f32(((i % 17) as f32 - 8.0) * 0.25))
                .collect::<Vec<_>>();
            let input = vector_array(128, &values, Validity::NonNullable)?;
            let packed = execute_tq_pack(input, &config, &mut ctx)?;
            let unpacked = execute_tq_unpack(packed, &config, &mut ctx)?;
            let ext: ExtensionArray = unpacked.execute(&mut ctx)?;
            assert_eq!(vector_element_ptype(&ext)?, PType::F16);
        }
        PType::F64 => {
            let values = (0..2 * 128)
                .map(|i| ((i % 17) as f64 - 8.0) * 0.25)
                .collect::<Vec<_>>();
            let input = vector_array(128, &values, Validity::NonNullable)?;
            let packed = execute_tq_pack(input, &config, &mut ctx)?;
            let unpacked = execute_tq_unpack(packed, &config, &mut ctx)?;
            let ext: ExtensionArray = unpacked.execute(&mut ctx)?;
            assert_eq!(vector_element_ptype(&ext)?, PType::F64);
        }
        _ => unreachable!("test only passes f16/f64"),
    }
    Ok(())
}

#[test]
fn unpack_scales_by_stored_norms() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let base = f32_vector_array(128, 1, 0.5, Validity::NonNullable)?;
    let scaled = f32_vector_array(128, 1, 1.0, Validity::NonNullable)?;
    let config = TurboQuantConfig::try_new(2, 99, 3)?;

    let base_values = vector_values_f32(
        execute_tq_unpack(execute_tq_pack(base, &config, &mut ctx)?, &config, &mut ctx)?,
        &mut ctx,
    )?;
    let scaled_values = vector_values_f32(
        execute_tq_unpack(
            execute_tq_pack(scaled, &config, &mut ctx)?,
            &config,
            &mut ctx,
        )?,
        &mut ctx,
    )?;

    for (base, scaled) in base_values.iter().zip(scaled_values.iter()) {
        assert!((*scaled - 2.0 * *base).abs() < 1e-5);
    }
    Ok(())
}
