// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::ArrayRef;
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

fn metadata() -> TurboQuantMetadata {
    TurboQuantMetadata {
        element_ptype: PType::F32,
        dimensions: 128,
        bit_width: 1,
        seed: 42,
        num_rounds: 3,
        block_sizes: vec![128],
    }
}

fn build_single_block_tq(
    metadata: TurboQuantMetadata,
    norms: ArrayRef,
    codes: ArrayRef,
    rows: usize,
    outer_validity: Validity,
) -> ArrayRef {
    let inner = StructArray::try_new(
        FieldNames::from(["norms", "codes"]),
        vec![norms, codes],
        rows,
        outer_validity.clone(),
    )
    .unwrap()
    .into_array();
    let outer = StructArray::try_new(
        FieldNames::from(["block_0"]),
        vec![inner],
        rows,
        outer_validity,
    )
    .unwrap();
    ExtensionArray::try_new_from_vtable(TurboQuant, metadata, outer.into_array())
        .unwrap()
        .into_array()
}

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
    let norms =
        PrimitiveArray::new::<f32>(Buffer::copy_from([1.0]), Validity::from(norms_nullability))
            .into_array();
    let codes = PrimitiveArray::new::<u8>(vec![0u8; 128], Validity::NonNullable);
    let codes = FixedSizeListArray::try_new(
        codes.into_array(),
        128,
        Validity::from(codes_nullability),
        1,
    )?
    .into_array();
    let tq = build_single_block_tq(
        metadata(),
        norms,
        codes,
        1,
        Validity::from(struct_nullability),
    );

    execute_tq_decode_from_metadata(tq, &mut ctx)?;
    Ok(())
}

#[test]
fn decode_accepts_struct_mask_with_all_valid_children() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let norms =
        PrimitiveArray::new::<f32>(Buffer::copy_from([1.0, 1.0, 1.0]), Validity::NonNullable)
            .into_array();
    let codes = PrimitiveArray::new::<u8>(vec![0u8; 3 * 128], Validity::NonNullable);
    let codes = FixedSizeListArray::try_new(codes.into_array(), 128, Validity::NonNullable, 3)?
        .into_array();
    let tq = build_single_block_tq(
        metadata(),
        norms,
        codes,
        3,
        Validity::from_iter([true, false, true]),
    );

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
    let tq = build_single_block_tq(
        metadata(),
        norms,
        codes,
        3,
        Validity::from_iter([true, false, true]),
    );

    assert!(execute_tq_decode_from_metadata(tq, &mut ctx).is_err());
    Ok(())
}

#[test]
fn decode_rejects_inner_struct_validity_narrower_than_outer() -> VortexResult<()> {
    // The outer struct marks all three rows valid, but the inner `block_0` struct marks row 1
    // invalid. `parse_storage` must reject this: each inner block's struct validity must
    // *cover* the outer struct's validity, so an inner row marked invalid where the outer is
    // valid is a contract violation.
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let rows = 3;
    let norms =
        PrimitiveArray::new::<f32>(Buffer::copy_from([1.0, 1.0, 1.0]), Validity::NonNullable)
            .into_array();
    let codes = PrimitiveArray::new::<u8>(vec![0u8; rows * 128], Validity::NonNullable);
    let codes = FixedSizeListArray::try_new(codes.into_array(), 128, Validity::NonNullable, rows)?
        .into_array();
    let inner = StructArray::try_new(
        FieldNames::from(["norms", "codes"]),
        vec![norms, codes],
        rows,
        Validity::from_iter([true, false, true]),
    )?
    .into_array();
    let outer = StructArray::try_new(
        FieldNames::from(["block_0"]),
        vec![inner],
        rows,
        Validity::NonNullable,
    )?;
    let tq = ExtensionArray::try_new_from_vtable(TurboQuant, metadata(), outer.into_array())?
        .into_array();

    assert!(execute_tq_decode_from_metadata(tq, &mut ctx).is_err());
    Ok(())
}

#[test]
fn decode_rejects_codes_outside_centroid_table() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let norms =
        PrimitiveArray::new::<f32>(Buffer::copy_from([1.0]), Validity::NonNullable).into_array();
    let mut codes = vec![0u8; 128];
    codes[0] = 2;
    let codes = PrimitiveArray::new::<u8>(codes, Validity::NonNullable);
    let codes = FixedSizeListArray::try_new(codes.into_array(), 128, Validity::NonNullable, 1)?
        .into_array();
    let tq = build_single_block_tq(metadata(), norms, codes, 1, Validity::NonNullable);

    // A code pointing past the centroid table must surface a clean error from the public decode
    // path rather than panicking through `vortex_expect`.
    assert!(execute_tq_decode_from_metadata(tq, &mut ctx).is_err());
    Ok(())
}

#[test]
fn decode_ignores_out_of_range_codes_in_null_rows() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    // Two rows; row 1 is masked out (outer validity false). Its placeholder codes contain an
    // out-of-range value (2, where bit_width=1 gives only 2 centroids). Decode must not validate
    // codes for masked-out rows, so this decodes cleanly with row 1 as a null placeholder.
    let norms = PrimitiveArray::new::<f32>(Buffer::copy_from([1.0, 0.0]), Validity::NonNullable)
        .into_array();
    let mut codes = vec![0u8; 2 * 128];
    codes[128] = 2;
    let codes = PrimitiveArray::new::<u8>(codes, Validity::NonNullable);
    let codes = FixedSizeListArray::try_new(codes.into_array(), 128, Validity::NonNullable, 2)?
        .into_array();
    let tq = build_single_block_tq(
        metadata(),
        norms,
        codes,
        2,
        Validity::from_iter([true, false]),
    );

    let decoded = execute_tq_decode_from_metadata(tq, &mut ctx)?;
    let validity = vector_validity(decoded, &mut ctx)?.execute_mask(2, &mut ctx)?;
    assert!(validity.value(0));
    assert!(!validity.value(1));
    Ok(())
}

#[test]
fn decode_rejects_non_finite_stored_norm() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    // A non-finite stored norm would scale the reconstruction to inf/NaN; decode rejects it.
    let norms =
        PrimitiveArray::new::<f32>(Buffer::copy_from([f32::INFINITY]), Validity::NonNullable)
            .into_array();
    let codes = PrimitiveArray::new::<u8>(vec![0u8; 128], Validity::NonNullable);
    let codes = FixedSizeListArray::try_new(codes.into_array(), 128, Validity::NonNullable, 1)?
        .into_array();
    let tq = build_single_block_tq(metadata(), norms, codes, 1, Validity::NonNullable);

    assert!(execute_tq_decode_from_metadata(tq, &mut ctx).is_err());
    Ok(())
}

#[test]
fn decode_rejects_negative_stored_norm() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    // L2 norms are never negative; a negative stored norm (only from corrupt storage) would
    // sign-flip the reconstruction, so decode rejects it.
    let norms =
        PrimitiveArray::new::<f32>(Buffer::copy_from([-1.0]), Validity::NonNullable).into_array();
    let codes = PrimitiveArray::new::<u8>(vec![0u8; 128], Validity::NonNullable);
    let codes = FixedSizeListArray::try_new(codes.into_array(), 128, Validity::NonNullable, 1)?
        .into_array();
    let tq = build_single_block_tq(metadata(), norms, codes, 1, Validity::NonNullable);

    assert!(execute_tq_decode_from_metadata(tq, &mut ctx).is_err());
    Ok(())
}

/// Malformed storage in a LATER block (not block_0) must be rejected too, exercising the per-block
/// decode loop's validation beyond the first block: here block_1 carries an out-of-range code.
#[test]
fn decode_rejects_out_of_range_code_in_later_block() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let metadata = TurboQuantMetadata {
        element_ptype: PType::F32,
        dimensions: 256,
        bit_width: 1,
        seed: 42,
        num_rounds: 3,
        block_sizes: vec![128, 128],
    };
    // bit_width = 1 gives 2 centroids, so a code of 2 is out of range.
    let make_block = |bad_code: bool| -> VortexResult<ArrayRef> {
        let norms = PrimitiveArray::new::<f32>(Buffer::copy_from([1.0]), Validity::NonNullable)
            .into_array();
        let mut codes = vec![0u8; 128];
        if bad_code {
            codes[0] = 2;
        }
        let codes = PrimitiveArray::new::<u8>(codes, Validity::NonNullable);
        let codes = FixedSizeListArray::try_new(codes.into_array(), 128, Validity::NonNullable, 1)?
            .into_array();
        Ok(StructArray::try_new(
            FieldNames::from(["norms", "codes"]),
            vec![norms, codes],
            1,
            Validity::NonNullable,
        )?
        .into_array())
    };
    let outer = StructArray::try_new(
        FieldNames::from(["block_0", "block_1"]),
        vec![make_block(false)?, make_block(true)?],
        1,
        Validity::NonNullable,
    )?;
    let tq =
        ExtensionArray::try_new_from_vtable(TurboQuant, metadata, outer.into_array())?.into_array();

    assert!(execute_tq_decode_from_metadata(tq, &mut ctx).is_err());
    Ok(())
}
