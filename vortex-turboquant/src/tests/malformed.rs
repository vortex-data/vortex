// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use super::execute_tq_decode_from_metadata;
use super::test_session;
use super::vector_validity;
use crate::TurboQuant;
use crate::TurboQuantMetadata;

#[rstest]
#[case::nullable_norms_under_nonnullable_struct(
    Nullability::NonNullable,
    Nullability::Nullable,
    Nullability::NonNullable
)]
#[case::nullable_codes_under_nonnullable_struct(
    Nullability::NonNullable,
    Nullability::NonNullable,
    Nullability::Nullable
)]
#[case::nonnullable_norms_under_nullable_struct(
    Nullability::Nullable,
    Nullability::NonNullable,
    Nullability::Nullable
)]
#[case::nonnullable_codes_under_nullable_struct(
    Nullability::Nullable,
    Nullability::Nullable,
    Nullability::NonNullable
)]
fn decode_accepts_child_nullability_that_covers_struct_validity(
    #[case] struct_nullability: Nullability,
    #[case] norms_nullability: Nullability,
    #[case] codes_nullability: Nullability,
) -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let metadata = TurboQuantMetadata {
        element_ptype: PType::F32,
        dimensions: 128,
        bit_width: 1,
        seed: 42,
        num_rounds: 3,
    };
    let norms =
        PrimitiveArray::new::<f32>(Buffer::copy_from([1.0]), Validity::from(norms_nullability))
            .into_array();
    let codes = PrimitiveArray::new::<u8>(vec![0u8; 128], Validity::NonNullable);
    let codes = FixedSizeListArray::try_new(
        codes.into_array(),
        128,
        Validity::from(codes_nullability),
        1,
    )
    .unwrap()
    .into_array();
    let storage = StructArray::try_new(
        FieldNames::from(["norms", "codes"]),
        vec![norms, codes],
        1,
        Validity::from(struct_nullability),
    )
    .unwrap();
    let tq = ExtensionArray::try_new_from_vtable(TurboQuant, metadata, storage.into_array())
        .unwrap()
        .into_array();

    execute_tq_decode_from_metadata(tq, &mut ctx)?;
    Ok(())
}

#[test]
fn decode_accepts_struct_mask_with_all_valid_children() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let metadata = TurboQuantMetadata {
        element_ptype: PType::F32,
        dimensions: 128,
        bit_width: 1,
        seed: 42,
        num_rounds: 3,
    };
    let norms =
        PrimitiveArray::new::<f32>(Buffer::copy_from([1.0, 1.0, 1.0]), Validity::NonNullable)
            .into_array();
    let codes = PrimitiveArray::new::<u8>(vec![0u8; 3 * 128], Validity::NonNullable);
    let codes = FixedSizeListArray::try_new(codes.into_array(), 128, Validity::NonNullable, 3)?
        .into_array();
    let storage = StructArray::try_new(
        FieldNames::from(["norms", "codes"]),
        vec![norms, codes],
        3,
        Validity::from_iter([true, false, true]),
    )?;
    let tq = ExtensionArray::try_new_from_vtable(TurboQuant, metadata, storage.into_array())?
        .into_array();

    let decoded = execute_tq_decode_from_metadata(tq, &mut ctx)?;
    let validity = vector_validity(decoded, &mut ctx)?.execute_mask(3, &mut ctx)?;
    assert!(validity.value(0));
    assert!(!validity.value(1));
    assert!(validity.value(2));
    Ok(())
}

#[test]
fn decode_rejects_child_masks_that_disagree_with_struct_validity() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let metadata = TurboQuantMetadata {
        element_ptype: PType::F32,
        dimensions: 128,
        bit_width: 1,
        seed: 42,
        num_rounds: 3,
    };
    let norms = PrimitiveArray::new::<f32>(
        Buffer::copy_from([1.0, 1.0, 1.0]),
        Validity::from_iter([true, true, false]),
    )
    .into_array();
    let codes = PrimitiveArray::new::<u8>(vec![0u8; 3 * 128], Validity::NonNullable);
    let codes = FixedSizeListArray::try_new(
        codes.into_array(),
        128,
        Validity::from_iter([true, false, true]),
        3,
    )?
    .into_array();
    let storage = StructArray::try_new(
        FieldNames::from(["norms", "codes"]),
        vec![norms, codes],
        3,
        Validity::from_iter([true, false, true]),
    )?;
    let tq = ExtensionArray::try_new_from_vtable(TurboQuant, metadata, storage.into_array())?
        .into_array();

    assert!(execute_tq_decode_from_metadata(tq, &mut ctx).is_err());
    Ok(())
}

#[test]
#[should_panic(expected = "TurboQuant code exceeds centroid count")]
fn decode_panics_on_codes_outside_centroid_table() {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let metadata = TurboQuantMetadata {
        element_ptype: PType::F32,
        dimensions: 128,
        bit_width: 1,
        seed: 42,
        num_rounds: 3,
    };
    let norms =
        PrimitiveArray::new::<f32>(Buffer::copy_from([1.0]), Validity::NonNullable).into_array();
    let mut codes = vec![0u8; 128];
    codes[0] = 2;
    let codes = PrimitiveArray::new::<u8>(codes, Validity::NonNullable);
    let codes = FixedSizeListArray::try_new(codes.into_array(), 128, Validity::NonNullable, 1)
        .unwrap()
        .into_array();
    let storage = StructArray::try_new(
        FieldNames::from(["norms", "codes"]),
        vec![norms, codes],
        1,
        Validity::NonNullable,
    )
    .unwrap();
    let tq = ExtensionArray::try_new_from_vtable(TurboQuant, metadata, storage.into_array())
        .unwrap()
        .into_array();

    drop(execute_tq_decode_from_metadata(tq, &mut ctx));
}
