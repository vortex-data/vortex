// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for TurboQuant-specific session-scoped optimizer kernels.

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_tensor::scalar_fns::l2_norm::L2Norm;

use super::execute_tq_encode;
use super::f32_vector_array;
use super::tensor_test_session;
use super::vector_array;
use crate::TQDecode;
use crate::TurboQuantConfig;
use crate::vector::storage::parse_storage;

const DIM: u32 = 128;

/// Fast path: `L2Norm(TQDecode(tq_arr))` returns the storage `norms` field bit-for-bit.
///
/// The slow path would recompute norms from lossily decoded vectors, which only approximately
/// match the stored norms. Bit-exact equality is the strongest invariant that confirms the
/// session-registered kernel fired.
#[test]
fn l2_norm_over_tq_decode_returns_stored_norms() -> VortexResult<()> {
    let session = tensor_test_session();
    let mut ctx = session.create_execution_ctx();
    let input = f32_vector_array(DIM, 4, 0.25, Validity::NonNullable)?;
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

/// Negative: directly wrapping a `Vector` (no `TQDecode`) must hit the canonical `L2Norm` path.
///
/// Proves the kernel only intercepts the matched `(L2Norm, TQDecode)` pair and does not affect
/// the standard tensor scalar-function flow.
#[test]
fn l2_norm_over_plain_vector_uses_canonical_path() -> VortexResult<()> {
    let session = tensor_test_session();
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
    let session = tensor_test_session();
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

/// Null rows: the kernel must preserve the input's row-level validity and produce correct norms
/// for the non-null rows.
#[rstest]
#[case::leading_null(Validity::from_iter([false, true, true]))]
#[case::trailing_null(Validity::from_iter([true, true, false]))]
#[case::interior_null(Validity::from_iter([true, false, true]))]
fn l2_norm_over_tq_decode_preserves_nulls(#[case] validity: Validity) -> VortexResult<()> {
    let session = tensor_test_session();
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

/// Masked input: generic masks narrow the TurboQuant storage struct validity without rewriting the
/// `norms` child, so the kernel must apply the authoritative struct validity before returning.
#[test]
fn l2_norm_over_masked_tq_decode_uses_storage_validity() -> VortexResult<()> {
    let session = tensor_test_session();
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
