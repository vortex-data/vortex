// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for TurboQuant-specific session-scoped optimizer kernels.

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_tensor::scalar_fns::l2_norm::L2Norm;

use super::execute_tq_decode;
use super::execute_tq_encode;
use super::f32_vector_array;
use super::test_session;
use super::vector_array;
use crate::TQDecode;
use crate::TurboQuant;
use crate::TurboQuantConfig;
use crate::TurboQuantMetadata;
use crate::vector::storage::parse_storage;

const DIM: u32 = 128;

/// Fast path: `L2Norm(TQDecode(tq_arr))` returns the storage `norms` field bit-for-bit
/// across every supported element ptype, so the kernel's per-ptype buffer-handle plumbing
/// is exercised at `f16`, `f32`, and `f64` rather than only the default `f32`.
///
/// `TQDecode` rescales each decoded direction in flight by the reciprocal of its own L2
/// norm before re-applying the stored row norm, so decoded rows preserve the stored norm
/// exactly. Bit-exact equality with the parsed `norms` child is consistent with the
/// session-registered kernel firing (the canonical-cross-check test below pins the
/// equivalence under arithmetic).
#[rstest]
#[case::f16(PType::F16)]
#[case::f32(PType::F32)]
#[case::f64(PType::F64)]
fn l2_norm_over_tq_decode_returns_stored_norms(#[case] ptype: PType) -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let rows = 4;
    let raw = (0..rows * DIM as usize)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.25)
        .collect::<Vec<_>>();
    let input = match ptype {
        PType::F16 => {
            let values: Vec<half::f16> = raw.iter().copied().map(half::f16::from_f32).collect();
            vector_array(DIM, &values, Validity::NonNullable)?
        }
        PType::F32 => vector_array(DIM, &raw, Validity::NonNullable)?,
        PType::F64 => {
            let values: Vec<f64> = raw.iter().copied().map(f64::from).collect();
            vector_array(DIM, &values, Validity::NonNullable)?
        }
        _ => unreachable!("ptype must be float"),
    };
    let config = TurboQuantConfig::try_new(3, 42, 3)?;

    let encoded = execute_tq_encode(input, &config, &mut ctx)?;
    let expected_norms = parse_storage(encoded.clone(), &mut ctx)?.norms.into_array();

    let decoded = TQDecode::try_new_array(encoded)?.into_array();
    let row_count = decoded.len();
    let result: PrimitiveArray = L2Norm::try_new_array(decoded, row_count)?
        .into_array()
        .execute(&mut ctx)?;

    assert_arrays_eq!(result, expected_norms);
    Ok(())
}

/// Negative: directly wrapping a `Vector` (no `TQDecode`) must hit the canonical `L2Norm`
/// path. Proves the kernel only intercepts the matched `(L2Norm, TQDecode)` pair and does
/// not affect the standard tensor scalar-function flow.
#[test]
fn l2_norm_over_plain_vector_uses_canonical_path() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();

    let input = vector_array(
        3,
        &[
            3.0f32, 4.0, 0.0, // row 0, norm = 5.0
            0.0, 0.0, 0.0, // row 1, norm = 0.0
            1.0, 0.0, 0.0, // row 2, norm = 1.0
        ],
        Validity::NonNullable,
    )?;

    let row_count = input.len();
    let result: PrimitiveArray = L2Norm::try_new_array(input, row_count)?
        .into_array()
        .execute(&mut ctx)?;

    let expected =
        PrimitiveArray::new::<f32>(Buffer::copy_from([5.0f32, 0.0, 1.0]), Validity::NonNullable);
    assert_arrays_eq!(result, expected);
    Ok(())
}

/// Empty input: zero-length TurboQuant array still produces a zero-length norms array of the
/// matching primitive dtype.
#[test]
fn l2_norm_over_empty_tq_decode_is_empty_norms() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let input = vector_array::<f32>(DIM, &[], Validity::NonNullable)?;
    let config = TurboQuantConfig::try_new(3, 42, 3)?;

    let encoded = execute_tq_encode(input, &config, &mut ctx)?;
    let decoded = TQDecode::try_new_array(encoded)?.into_array();
    let result: PrimitiveArray = L2Norm::try_new_array(decoded, 0)?
        .into_array()
        .execute(&mut ctx)?;

    assert_eq!(result.len(), 0);
    assert_eq!(
        result.dtype(),
        &DType::Primitive(PType::F32, Nullability::NonNullable)
    );
    Ok(())
}

/// Null rows: the kernel must preserve the input's row-level validity and produce correct
/// norms for the non-null rows.
#[rstest]
#[case::leading_null(Validity::from_iter([false, true, true]))]
#[case::trailing_null(Validity::from_iter([true, true, false]))]
#[case::interior_null(Validity::from_iter([true, false, true]))]
fn l2_norm_over_tq_decode_preserves_nulls(#[case] validity: Validity) -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let input = f32_vector_array(DIM, 3, 0.25, validity)?;
    let config = TurboQuantConfig::try_new(4, 7, 2)?;

    let encoded = execute_tq_encode(input, &config, &mut ctx)?;
    let expected_norms = parse_storage(encoded.clone(), &mut ctx)?.norms.into_array();

    let decoded = TQDecode::try_new_array(encoded)?.into_array();
    let result: PrimitiveArray = L2Norm::try_new_array(decoded, 3)?
        .into_array()
        .execute(&mut ctx)?;

    assert_arrays_eq!(result, expected_norms);
    Ok(())
}

/// Masked input: generic masks narrow the TurboQuant storage struct validity without
/// rewriting the `norms` child, so the kernel must apply the authoritative struct validity
/// before returning.
#[test]
fn l2_norm_over_masked_tq_decode_uses_storage_validity() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let input = f32_vector_array(DIM, 4, 0.25, Validity::NonNullable)?;
    let config = TurboQuantConfig::try_new(3, 42, 3)?;

    let encoded = execute_tq_encode(input, &config, &mut ctx)?;
    let masked = encoded.mask(BoolArray::from_iter([true, false, true, false]).into_array())?;

    let decoded = TQDecode::try_new_array(masked)?.into_array();
    let result: PrimitiveArray = L2Norm::try_new_array(decoded, 4)?
        .into_array()
        .execute(&mut ctx)?;
    let validity = result.validity()?.execute_mask(4, &mut ctx)?;

    assert!(validity.value(0));
    assert!(!validity.value(1));
    assert!(validity.value(2));
    assert!(!validity.value(3));
    assert_eq!(
        result.dtype(),
        &DType::Primitive(PType::F32, Nullability::Nullable)
    );
    Ok(())
}

/// Regression for the wider-child-nullability shape (`Nullable` `norms` with `AllValid`
/// under a `NonNullable` struct). `parse_storage` accepts this shape (see
/// `malformed::decode_accepts_child_nullability_that_covers_struct_validity`); the kernel
/// must return a `NonNullable` result rather than reusing the wider child validity.
#[test]
fn l2_norm_over_tq_decode_nullable_norms_under_nonnullable_struct() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let metadata = TurboQuantMetadata {
        element_ptype: PType::F32,
        dimensions: DIM,
        bit_width: 1,
        seed: 42,
        num_rounds: 3,
    };

    let norms =
        PrimitiveArray::new::<f32>(Buffer::copy_from([5.0f32]), Validity::AllValid).into_array();
    let codes = PrimitiveArray::new::<u8>(vec![0u8; DIM as usize], Validity::NonNullable);
    let codes =
        FixedSizeListArray::try_new(codes.into_array(), DIM, Validity::AllValid, 1)?.into_array();
    let storage = StructArray::try_new(
        FieldNames::from(["norms", "codes"]),
        vec![norms, codes],
        1,
        Validity::NonNullable,
    )?;
    let tq = ExtensionArray::try_new_from_vtable(TurboQuant, metadata, storage.into_array())?
        .into_array();

    let decoded = TQDecode::try_new_array(tq)?.into_array();
    let result: PrimitiveArray = L2Norm::try_new_array(decoded, 1)?
        .into_array()
        .execute(&mut ctx)?;

    assert_eq!(
        result.dtype(),
        &DType::Primitive(PType::F32, Nullability::NonNullable),
        "kernel result dtype must match parent (NonNullable), not the wider child validity"
    );
    assert_eq!(result.as_slice::<f32>(), &[5.0f32]);
    Ok(())
}

/// Cross-check the kernel result against the canonical `L2Norm(execute(TQDecode))` path.
/// Materializing the decoded vector first breaks the `(L2Norm, TQDecode)` pattern so
/// `L2Norm` runs through the canonical scalar-function flow.
#[rstest]
#[case::dim_128(128_u32)]
#[case::dim_129(129_u32)]
#[case::dim_257(257_u32)]
fn l2_norm_over_tq_decode_matches_canonical(#[case] dim: u32) -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let input = f32_vector_array(dim, 4, 0.25, Validity::NonNullable)?;
    let config = TurboQuantConfig::try_new(3, 42, 3)?;

    let encoded = execute_tq_encode(input, &config, &mut ctx)?;

    let kernel_result: PrimitiveArray =
        L2Norm::try_new_array(TQDecode::try_new_array(encoded.clone())?.into_array(), 4)?
            .into_array()
            .execute(&mut ctx)?;

    // Materialize the decoded vector first so `L2Norm` cannot match `(L2Norm, TQDecode)`. The
    // resulting `L2Norm(Vector)` flows through the canonical scalar-function path.
    let decoded = execute_tq_decode(encoded, &mut ctx)?;
    let canonical_result: PrimitiveArray = L2Norm::try_new_array(decoded, 4)?
        .into_array()
        .execute(&mut ctx)?;

    let kernel = kernel_result.as_slice::<f32>();
    let canonical = canonical_result.as_slice::<f32>();
    for (k, c) in kernel.iter().zip(canonical.iter()) {
        assert!(
            (*k - *c).abs() <= 1e-4 * c.max(1.0),
            "kernel result {k} disagrees with canonical {c} (dim {dim})"
        );
    }
    Ok(())
}

/// Adversarial: a hand-constructed TurboQuant storage with a `-5.0` or `-0.0` stored norm
/// makes the fast path fall back to the canonical `L2Norm(execute(TQDecode))` path so that
/// the result preserves `L2Norm`'s always-non-negative output invariant. The kernel scans
/// the parsed `norms` once and triggers fallback via `is_sign_negative`, which covers both
/// strictly-negative values and `-0.0` (where the literal `< 0` comparison would fail per
/// IEEE 754).
#[rstest]
#[case::strict_negative(-5.0_f32)]
#[case::negative_zero(-0.0_f32)]
fn l2_norm_over_tq_decode_with_negative_stored_norm_falls_back(
    #[case] stored: f32,
) -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let metadata = TurboQuantMetadata {
        element_ptype: PType::F32,
        dimensions: DIM,
        bit_width: 1,
        seed: 42,
        num_rounds: 3,
    };

    let norms =
        PrimitiveArray::new::<f32>(Buffer::copy_from([stored]), Validity::NonNullable).into_array();
    let codes = PrimitiveArray::new::<u8>(vec![0u8; DIM as usize], Validity::NonNullable);
    let codes = FixedSizeListArray::try_new(codes.into_array(), DIM, Validity::NonNullable, 1)?
        .into_array();
    let storage = StructArray::try_new(
        FieldNames::from(["norms", "codes"]),
        vec![norms, codes],
        1,
        Validity::NonNullable,
    )?;
    let tq = ExtensionArray::try_new_from_vtable(TurboQuant, metadata, storage.into_array())?
        .into_array();

    let decoded = TQDecode::try_new_array(tq)?.into_array();
    let result: PrimitiveArray = L2Norm::try_new_array(decoded, 1)?
        .into_array()
        .execute(&mut ctx)?;

    // Whatever path runs, the result is an `L2Norm` output and must be non-negative; in
    // particular the kernel must NOT return the stored sign-negative value verbatim. The
    // exact magnitude depends on which centroid the all-zero codes decode to; we only
    // assert the sign and finiteness, which is what `L2Norm`'s contract pins.
    assert_eq!(result.as_slice::<f32>().len(), 1);
    let value = result.as_slice::<f32>()[0];
    assert!(
        value.is_finite() && !value.is_sign_negative(),
        "L2Norm result must be non-negative and finite (got {value})"
    );
    Ok(())
}

/// Adversarial: a hand-constructed TurboQuant storage whose `codes` child has row validity
/// narrower than the outer struct's must fail the fast path the same way it fails the
/// canonical decode path (see `malformed::decode_rejects_child_masks_that_disagree_with_struct_validity`).
/// `parse_storage_norms_only` executes the `codes` FSL wrapper specifically to enforce this
/// invariant.
#[test]
fn l2_norm_over_tq_decode_rejects_codes_validity_narrower_than_struct() -> VortexResult<()> {
    let session = test_session();
    let mut ctx = session.create_execution_ctx();
    let metadata = TurboQuantMetadata {
        element_ptype: PType::F32,
        dimensions: DIM,
        bit_width: 1,
        seed: 42,
        num_rounds: 3,
    };

    let norms =
        PrimitiveArray::new::<f32>(Buffer::copy_from([1.0f32, 1.0, 1.0]), Validity::NonNullable)
            .into_array();
    let codes = PrimitiveArray::new::<u8>(vec![0u8; 3 * DIM as usize], Validity::NonNullable);
    let codes = FixedSizeListArray::try_new(
        codes.into_array(),
        DIM,
        Validity::from_iter([true, false, true]),
        3,
    )?
    .into_array();
    let storage = StructArray::try_new(
        FieldNames::from(["norms", "codes"]),
        vec![norms, codes],
        3,
        Validity::NonNullable,
    )?;
    let tq = ExtensionArray::try_new_from_vtable(TurboQuant, metadata, storage.into_array())?
        .into_array();

    let decoded = TQDecode::try_new_array(tq)?.into_array();
    let result: VortexResult<PrimitiveArray> = L2Norm::try_new_array(decoded, 3)?
        .into_array()
        .execute(&mut ctx);
    assert!(
        result.is_err(),
        "kernel must reject codes-validity narrower than struct-validity"
    );
    Ok(())
}
