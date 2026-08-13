// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArrayVTable;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Extension;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::extension::datetime::Date;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::scalar::Scalar;
use vortex_array::session::ArraySessionExt;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_compressor::scheme::Scheme;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::encodings::normalized::Normalized;
use crate::encodings::normalized::NormalizedArraySlotsExt;
use crate::encodings::normalized::NormalizedScheme;
use crate::encodings::normalized::NormalizedSlots;
use crate::encodings::normalized::normalize;
use crate::encodings::normalized::validate_normalized_rows;
use crate::tests::SESSION;
use crate::types::vector::Vector;
use crate::utils::test_helpers::assert_close;
use crate::utils::test_helpers::constant_tensor_array;
use crate::utils::test_helpers::tensor_array;
use crate::utils::test_helpers::vector_array;

fn eval_normalized(
    normalized: ArrayRef,
    norms: ArrayRef,
    validity: Validity,
) -> VortexResult<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    let normalized_array = Normalized::try_new(normalized, norms, validity, &mut ctx)?;

    normalized_array.into_array().execute(&mut ctx)
}

/// Captures dtype, row validity, and visible values without depending on the physical encoding.
fn tensor_snapshot(array: ArrayRef) -> VortexResult<(DType, Vec<bool>, Vec<Option<f64>>)> {
    let mut ctx = SESSION.create_execution_ctx();
    let ext: ExtensionArray = array.execute(&mut ctx)?;
    let validity = (0..ext.len())
        .map(|i| ext.is_valid(i, &mut ctx))
        .collect::<VortexResult<Vec<_>>>()?;
    let storage: FixedSizeListArray = ext.storage_array().clone().execute(&mut ctx)?;
    let elements: PrimitiveArray = storage.elements().clone().execute(&mut ctx)?;

    let list_size = storage.list_size() as usize;
    // Ignore physical values from null rows because their contents are unspecified.
    let values = elements
        .as_slice::<f64>()
        .iter()
        .enumerate()
        .map(|(i, &value)| validity[i / list_size].then_some(value))
        .collect();

    Ok((ext.dtype().clone(), validity, values))
}

#[track_caller]
fn assert_tensor_arrays_eq(actual: ArrayRef, expected: ArrayRef) -> VortexResult<()> {
    let (actual_dtype, actual_validity, actual_values) = tensor_snapshot(actual)?;
    let (expected_dtype, expected_validity, expected_values) = tensor_snapshot(expected)?;

    assert_eq!(actual_dtype, expected_dtype);
    assert_eq!(actual_validity, expected_validity);
    assert_eq!(actual_values.len(), expected_values.len());

    for (i, (actual, expected)) in actual_values.iter().zip(&expected_values).enumerate() {
        match (actual, expected) {
            (None, None) => {}
            (Some(actual), Some(expected)) => assert_close(&[*actual], &[*expected]),
            _ => panic!("element {i}: got {actual:?}, expected {expected:?}"),
        }
    }

    Ok(())
}

fn non_tensor_extension_array() -> VortexResult<ArrayRef> {
    let storage = PrimitiveArray::from_iter([1i32, 2]).into_array();
    let ext_dtype = ExtDType::<Date>::try_new(TimeUnit::Days, storage.dtype().clone())?.erased();

    Ok(ExtensionArray::new(ext_dtype, storage).into_array())
}

fn constant_f64_norms(value: f64, len: usize) -> ArrayRef {
    ConstantArray::new(Scalar::primitive(value, Nullability::NonNullable), len).into_array()
}

fn nullable_vector_input() -> VortexResult<ArrayRef> {
    let vectors = vector_array(2, &[3.0, 4.0, 1.0, 0.0, 0.0, 2.0])?;

    Ok(MaskedArray::try_new(vectors, Validity::from_iter([true, false, true]))?.into_array())
}

// =============================================================================
// Decoding
// =============================================================================

#[test]
fn decodes_vectors() -> VortexResult<()> {
    let normalized = vector_array(3, &[0.6, 0.8, 0.0, 0.0, 0.0, 0.0])?;
    let norms = PrimitiveArray::from_iter([5.0f64, 0.0]).into_array();

    let actual = eval_normalized(normalized, norms, Validity::NonNullable)?;
    let expected = vector_array(3, &[3.0, 4.0, 0.0, 0.0, 0.0, 0.0])?;

    assert_tensor_arrays_eq(actual, expected)
}

#[test]
fn decodes_fixed_shape_tensors() -> VortexResult<()> {
    let normalized = tensor_array(&[2, 2], &[0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0])?;
    let norms = PrimitiveArray::from_iter([4.0f64, 2.0]).into_array();

    let actual = eval_normalized(normalized, norms, Validity::NonNullable)?;
    let expected = tensor_array(&[2, 2], &[2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0])?;

    assert_tensor_arrays_eq(actual, expected)
}

#[test]
fn decodes_null_rows_from_the_stored_validity() -> VortexResult<()> {
    let normalized = vector_array(
        2,
        &[
            0.6, 0.8, // Row 0 decodes to [3.0, 4.0].
            0.0, 0.0, // Row 1 is null and zeroed.
            1.0, 0.0, // Row 2 decodes to [2.0, 0.0].
        ],
    )?;
    let norms = PrimitiveArray::from_iter([5.0f64, 0.0, 2.0]).into_array();

    let mut ctx = SESSION.create_execution_ctx();
    let validity = Validity::from_iter([true, false, true]);
    let actual: ExtensionArray = eval_normalized(normalized, norms, validity)?.execute(&mut ctx)?;
    let storage: FixedSizeListArray = actual.storage_array().clone().execute(&mut ctx)?;
    let elements: PrimitiveArray = storage.elements().clone().execute(&mut ctx)?;

    assert!(actual.is_valid(0, &mut ctx)?);
    assert!(!actual.is_valid(1, &mut ctx)?);
    assert!(actual.is_valid(2, &mut ctx)?);
    assert_close(&elements.as_slice::<f64>()[..2], &[3.0, 4.0]);
    assert_close(&elements.as_slice::<f64>()[4..], &[2.0, 0.0]);

    Ok(())
}

#[test]
fn validity_comes_from_the_stored_null_map() -> VortexResult<()> {
    let normalized = vector_array(
        2,
        &[
            1.0, 0.0, // Row 0.
            1.0, 0.0, // Row 1.
            1.0, 0.0, // Row 2.
        ],
    )?;
    let norms = PrimitiveArray::from_iter([1.0f64, 1.0, 1.0]).into_array();

    let mut ctx = SESSION.create_execution_ctx();
    let validity = Validity::from_iter([true, false, false]);
    let normalized_array = Normalized::try_new(normalized, norms, validity, &mut ctx)?;

    assert!(normalized_array.dtype().is_nullable());
    assert!(!normalized_array.normalized().dtype().is_nullable());
    assert!(!normalized_array.norms().dtype().is_nullable());

    let mask = normalized_array
        .as_ref()
        .validity()?
        .execute_mask(3, &mut ctx)?;
    assert!(mask.value(0));
    assert!(!mask.value(1));
    assert!(!mask.value(2));

    Ok(())
}

// =============================================================================
// Constant fast paths
// =============================================================================

#[test]
fn constant_unit_norms_decode_to_the_normalized_child() -> VortexResult<()> {
    // Every stored norm is exactly 1.0, so the fast path must short-circuit and return the
    // normalized child unchanged.
    let normalized = vector_array(3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])?;
    let norms = constant_f64_norms(1.0, 2);

    let actual = eval_normalized(normalized.clone(), norms, Validity::NonNullable)?;

    assert_tensor_arrays_eq(actual, normalized)
}

#[test]
fn constant_near_unit_norms_are_still_multiplied() -> VortexResult<()> {
    // Only an exact 1.0 is the identity. A norm that merely differs from 1.0 by less than the
    // unit-norm tolerance must still be applied, so that a per-row `scalar_at` cannot answer
    // differently than a bulk decode of the same column.
    let near_unit = 1.0f64 + 2.0 * f64::EPSILON;
    let normalized = vector_array(3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])?;
    let norms = constant_f64_norms(near_unit, 2);

    let mut ctx = SESSION.create_execution_ctx();
    let decoded = eval_normalized(normalized, norms, Validity::NonNullable)?;
    let ext: ExtensionArray = decoded.execute(&mut ctx)?;
    let storage: FixedSizeListArray = ext.storage_array().clone().execute(&mut ctx)?;
    let elements: PrimitiveArray = storage.elements().clone().execute(&mut ctx)?;

    assert_eq!(elements.as_slice::<f64>()[0], near_unit);

    Ok(())
}

#[test]
fn constant_nonunit_norms_scale_vectors() -> VortexResult<()> {
    let normalized = vector_array(3, &[0.6, 0.8, 0.0, 1.0, 0.0, 0.0])?;
    let norms = constant_f64_norms(5.0, 2);

    let actual = eval_normalized(normalized, norms, Validity::NonNullable)?;
    let expected = vector_array(3, &[3.0, 4.0, 0.0, 5.0, 0.0, 0.0])?;

    assert_tensor_arrays_eq(actual, expected)
}

#[test]
fn constant_nonunit_norms_scale_fixed_shape_tensors() -> VortexResult<()> {
    // The constant-scaling fast path must also cover multi-dimensional tensors, where the backing
    // elements buffer spans more than one slot per row.
    let normalized = tensor_array(&[2, 2], &[0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0])?;
    let norms = constant_f64_norms(4.0, 2);

    let actual = eval_normalized(normalized, norms, Validity::NonNullable)?;
    let expected = tensor_array(&[2, 2], &[2.0, 2.0, 2.0, 2.0, 4.0, 0.0, 0.0, 0.0])?;

    assert_tensor_arrays_eq(actual, expected)
}

#[rstest]
#[case::unit_norm(1.0)]
#[case::non_unit_norm(5.0)]
fn nullable_constant_norms_decode_to_the_nullable_dtype(#[case] norm: f64) -> VortexResult<()> {
    // The identity and bulk-multiply paths restore validity through different code paths.
    let normalized = vector_array(2, &[0.6f64, 0.8, 1.0, 0.0])?;
    let norms = constant_f64_norms(norm, 2);

    let mut ctx = SESSION.create_execution_ctx();
    let validity = Validity::from_iter([true, false]);
    let normalized_array = Normalized::try_new(normalized, norms, validity, &mut ctx)?;
    let dtype = normalized_array.dtype().clone();
    let decoded: ArrayRef = normalized_array.into_array().execute(&mut ctx)?;

    assert!(dtype.is_nullable());
    assert_eq!(decoded.dtype(), &dtype);
    assert!(decoded.is_valid(0, &mut ctx)?);
    assert!(!decoded.is_valid(1, &mut ctx)?);

    Ok(())
}

// =============================================================================
// Construction and validation
// =============================================================================

#[rstest]
#[case::non_extension_normalized(
    PrimitiveArray::from_iter([1.0f64, 2.0]).into_array(),
    PrimitiveArray::from_iter([1.0f64, 1.0]).into_array(),
)]
#[case::non_tensor_extension_normalized(
    non_tensor_extension_array().expect("valid date array"),
    PrimitiveArray::from_iter([1.0f64, 1.0]).into_array(),
)]
#[case::integer_tensor_normalized(
    tensor_array(&[2], &[1i32, 2, 3, 4]).expect("valid tensor array"),
    PrimitiveArray::from_iter([1.0f64, 1.0]).into_array(),
)]
#[case::mismatched_norms_ptype(
    vector_array(2, &[1.0f64, 0.0, 0.0, 1.0]).expect("valid vector array"),
    PrimitiveArray::from_iter([1.0f32, 1.0]).into_array(),
)]
#[case::non_primitive_norms(
    vector_array(2, &[1.0f64, 0.0, 0.0, 1.0]).expect("valid vector array"),
    vector_array(1, &[1.0f64, 1.0]).expect("valid vector array"),
)]
#[case::mismatched_child_lengths(
    vector_array(2, &[1.0f64, 0.0, 0.0, 1.0]).expect("valid vector array"),
    PrimitiveArray::from_iter([1.0f64]).into_array(),
)]
#[case::nullable_normalized(
    nullable_unit_vectors().expect("valid masked array"),
    PrimitiveArray::from_iter([1.0f64, 1.0]).into_array(),
)]
#[case::nullable_norms(
    vector_array(2, &[1.0f64, 0.0, 0.0, 1.0]).expect("valid vector array"),
    PrimitiveArray::from_option_iter([Some(1.0f64), None]).into_array(),
)]
fn rejects_structurally_invalid_children(
    #[case] normalized: ArrayRef,
    #[case] norms: ArrayRef,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    assert!(Normalized::try_new(normalized, norms, Validity::NonNullable, &mut ctx).is_err());

    Ok(())
}

fn nullable_unit_vectors() -> VortexResult<ArrayRef> {
    let vectors = vector_array(2, &[1.0f64, 0.0, 0.0, 1.0])?;

    Ok(MaskedArray::try_new(vectors, Validity::AllValid)?.into_array())
}

#[test]
fn rejects_a_validity_of_the_wrong_length() -> VortexResult<()> {
    let normalized = vector_array(2, &[1.0f64, 0.0, 0.0, 1.0])?;
    let norms = PrimitiveArray::from_iter([1.0f64, 1.0]).into_array();

    let mut ctx = SESSION.create_execution_ctx();
    let validity = Validity::from_iter([true, false, true]);

    assert!(Normalized::try_new(normalized, norms, validity, &mut ctx).is_err());

    Ok(())
}

#[rstest]
#[case::unnormalized_child(
    vector_array(2, &[3.0f64, 4.0, 1.0, 0.0]).expect("valid vector array"),
    PrimitiveArray::from_iter([5.0f64, 1.0]).into_array(),
)]
#[case::negative_norm(
    vector_array(2, &[1.0f64, 0.0, 0.0, 1.0]).expect("valid vector array"),
    PrimitiveArray::from_iter([1.0f64, -1.0]).into_array(),
)]
#[case::nonzero_row_with_zero_norm(
    vector_array(2, &[1.0f64, 0.0, 0.0, 0.0]).expect("valid vector array"),
    PrimitiveArray::from_iter([0.0f64, 0.0]).into_array(),
)]
// The mirror image of the case above: it decodes to `[0.0, 0.0]` while `L2Norm` reads the stored
// `5.0` straight back, so the split is not lossless.
#[case::zero_row_with_nonzero_norm(
    vector_array(2, &[0.0f64, 0.0]).expect("valid vector array"),
    PrimitiveArray::from_iter([5.0f64]).into_array(),
)]
fn checked_construction_rejects_semantic_violations(
    #[case] normalized: ArrayRef,
    #[case] norms: ArrayRef,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    assert!(Normalized::try_new(normalized, norms, Validity::NonNullable, &mut ctx).is_err());

    Ok(())
}

#[test]
fn accepts_zero_vectors_paired_with_zero_norms() -> VortexResult<()> {
    let normalized = vector_array(2, &[0.0, 0.0, 1.0, 0.0])?;
    let norms = PrimitiveArray::from_iter([0.0f64, 3.0]).into_array();

    let actual = eval_normalized(normalized, norms, Validity::NonNullable)?;
    let expected = vector_array(2, &[0.0, 0.0, 3.0, 0.0])?;

    assert_tensor_arrays_eq(actual, expected)
}

#[test]
fn validate_accepts_normalized_f16_rows() -> VortexResult<()> {
    let input = vector_array(2, &[3.0f32, 4.0, 0.0, 0.0].map(half::f16::from_f32))?;
    let mut ctx = SESSION.create_execution_ctx();

    let normalized_array = normalize(input, &mut ctx)?;
    validate_normalized_rows(&normalized_array.normalized().clone(), None, &mut ctx)
}

#[test]
fn checked_construction_accepts_dense_normalized_f16_row() -> VortexResult<()> {
    // Every coordinate is smaller than the unit-norm tolerance. Exact zero detection must not
    // misclassify this row as the zero vector.
    let element = half::f16::from_f32(1.0 / 128.0_f32.sqrt());
    let normalized = vector_array(128, &[element; 128])?;
    let norms = PrimitiveArray::from_iter([half::f16::from_f32(1.0)]).into_array();
    let mut ctx = SESSION.create_execution_ctx();

    Normalized::try_new(normalized, norms, Validity::NonNullable, &mut ctx)?;

    Ok(())
}

#[test]
fn validate_rejects_unnormalized_rows() -> VortexResult<()> {
    let input = vector_array(2, &[3.0, 4.0, 1.0, 0.0])?;
    let mut ctx = SESSION.create_execution_ctx();

    assert!(validate_normalized_rows(&input, None, &mut ctx).is_err());

    Ok(())
}

// =============================================================================
// Normalization
// =============================================================================

#[rstest]
#[case::vector(vector_array(3, &[3.0, 4.0, 0.0, 0.0, 0.0, 0.0]).expect("valid vector array"))]
#[case::fixed_shape_tensor(
    tensor_array(&[2, 2], &[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]).expect("valid tensor array")
)]
#[case::constant_tensor(constant_tensor_array(&[2], &[3.0, 4.0], 3).expect("valid tensor array"))]
#[case::constant_vector(Vector::constant_array(&[3.0, 4.0], 2).expect("valid vector array"))]
#[case::nullable_vector(nullable_vector_input().expect("valid vector array"))]
fn normalize_round_trips(#[case] input: ArrayRef) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let normalized_array = normalize(input.clone(), &mut ctx)?;
    let actual = normalized_array.into_array().execute(&mut ctx)?;

    assert_tensor_arrays_eq(actual, input)
}

#[test]
fn normalize_keeps_constant_input_children_constant() -> VortexResult<()> {
    // The constant fast path must leave both children constant, which is what lets cosine
    // similarity and inner product short-circuit against a literal query vector.
    let input = Vector::constant_array(&[3.0, 4.0], 16)?;
    let mut ctx = SESSION.create_execution_ctx();
    let normalized_array = normalize(input, &mut ctx)?;

    let normalized = normalized_array
        .normalized()
        .as_opt::<Extension>()
        .expect("normalized child should be an Extension array");
    assert!(
        normalized.storage_array().as_opt::<Constant>().is_some(),
        "normalized storage should stay constant after the fast path"
    );

    let norms = normalized_array
        .norms()
        .as_opt::<Constant>()
        .expect("norms child should be a ConstantArray");
    assert_close(
        &[norms
            .scalar()
            .as_primitive()
            .typed_value::<f64>()
            .expect("norms scalar")],
        &[5.0],
    );

    Ok(())
}

#[test]
fn normalize_zeroes_rows_with_zero_norms() -> VortexResult<()> {
    let input = vector_array(2, &[0.0, 0.0, 3.0, 4.0])?;
    let mut ctx = SESSION.create_execution_ctx();
    let normalized_array = normalize(input.clone(), &mut ctx)?;

    let normalized: ExtensionArray = normalized_array.normalized().clone().execute(&mut ctx)?;
    let storage: FixedSizeListArray = normalized.storage_array().clone().execute(&mut ctx)?;
    let elements: PrimitiveArray = storage.elements().clone().execute(&mut ctx)?;
    assert_close(&elements.as_slice::<f64>()[..2], &[0.0, 0.0]);

    let actual = normalized_array.into_array().execute(&mut ctx)?;

    assert_tensor_arrays_eq(actual, input)
}

#[test]
fn normalize_moves_input_nulls_onto_the_array() -> VortexResult<()> {
    // Row 1 is masked out but physically holds the unit vector `[1.0, 0.0]`, so a norm of 1.0 would
    // survive into the norms child if the null were not applied.
    let input = vector_array(
        2,
        &[
            3.0, 4.0, // Row 0 has norm 5.0.
            1.0, 0.0, // Row 1 is masked out despite being unit-norm.
            0.0, 1.0, // Row 2 has norm 1.0.
        ],
    )?;
    let input = MaskedArray::try_new(input, Validity::from_iter([true, false, true]))?.into_array();

    let mut ctx = SESSION.create_execution_ctx();
    let normalized_array = normalize(input, &mut ctx)?;

    assert!(normalized_array.dtype().is_nullable());
    assert!(!normalized_array.normalized().dtype().is_nullable());
    assert!(!normalized_array.norms().dtype().is_nullable());

    let mask = normalized_array
        .as_ref()
        .validity()?
        .execute_mask(3, &mut ctx)?;
    assert!(mask.value(0));
    assert!(!mask.value(1));
    assert!(mask.value(2));

    // Both children are zeroed at the null row rather than carrying whatever the masked-out storage
    // happened to hold, so no garbage reaches a downstream lossy encoding.
    let norms: PrimitiveArray = normalized_array.norms().clone().execute(&mut ctx)?;
    assert_close(&norms.as_slice::<f64>()[1..2], &[0.0]);

    let normalized: ExtensionArray = normalized_array.normalized().clone().execute(&mut ctx)?;
    let storage: FixedSizeListArray = normalized.storage_array().clone().execute(&mut ctx)?;
    let elements: PrimitiveArray = storage.elements().clone().execute(&mut ctx)?;
    assert_close(&elements.as_slice::<f64>()[2..4], &[0.0, 0.0]);

    Ok(())
}

// =============================================================================
// Row operations
// =============================================================================

#[test]
fn slice_stays_encoded_and_decodes_correctly() -> VortexResult<()> {
    let input = vector_array(2, &[3.0, 4.0, 1.0, 0.0, 0.0, 2.0, 5.0, 12.0])?;
    let mut ctx = SESSION.create_execution_ctx();
    let normalized_array = normalize(input, &mut ctx)?.into_array();

    let sliced = normalized_array
        .slice(1..3)?
        .execute_until::<Normalized>(&mut ctx)?;
    assert!(
        sliced.is::<Normalized>(),
        "slicing must push down into both children instead of decoding the column"
    );

    let expected = vector_array(2, &[1.0, 0.0, 0.0, 2.0])?;

    assert_tensor_arrays_eq(sliced, expected)
}

#[test]
fn filter_stays_encoded_and_decodes_correctly() -> VortexResult<()> {
    let input = vector_array(2, &[3.0, 4.0, 1.0, 0.0, 0.0, 2.0, 5.0, 12.0])?;
    let mut ctx = SESSION.create_execution_ctx();
    let normalized_array = normalize(input, &mut ctx)?.into_array();

    let mask = Mask::from_iter([true, false, true, false]);
    let filtered = normalized_array
        .filter(mask)?
        .execute_until::<Normalized>(&mut ctx)?;
    assert!(
        filtered.is::<Normalized>(),
        "filtering must push down into both children instead of decoding the column"
    );

    let expected = vector_array(2, &[3.0, 4.0, 0.0, 2.0])?;

    assert_tensor_arrays_eq(filtered, expected)
}

#[test]
fn slice_and_filter_carry_the_validity() -> VortexResult<()> {
    let input = vector_array(2, &[3.0, 4.0, 1.0, 0.0, 0.0, 2.0, 5.0, 12.0])?;
    let input =
        MaskedArray::try_new(input, Validity::from_iter([true, false, true, false]))?.into_array();

    let mut ctx = SESSION.create_execution_ctx();
    let normalized_array = normalize(input, &mut ctx)?.into_array();

    let sliced = normalized_array
        .slice(1..3)?
        .execute_until::<Normalized>(&mut ctx)?;
    assert!(sliced.is::<Normalized>());
    assert!(!sliced.is_valid(0, &mut ctx)?);
    assert!(sliced.is_valid(1, &mut ctx)?);

    let filtered = normalized_array
        .filter(Mask::from_iter([true, true, false, false]))?
        .execute_until::<Normalized>(&mut ctx)?;
    assert!(filtered.is::<Normalized>());
    assert!(filtered.is_valid(0, &mut ctx)?);
    assert!(!filtered.is_valid(1, &mut ctx)?);

    Ok(())
}

#[test]
fn take_decodes_correctly() -> VortexResult<()> {
    let input = vector_array(2, &[3.0, 4.0, 1.0, 0.0, 0.0, 2.0, 5.0, 12.0])?;
    let mut ctx = SESSION.create_execution_ctx();
    let normalized_array = normalize(input, &mut ctx)?.into_array();

    let indices = PrimitiveArray::from_iter([3u64, 0, 3]).into_array();
    let taken = normalized_array
        .take(indices)?
        .execute::<Canonical>(&mut ctx)?;
    let expected = vector_array(2, &[5.0, 12.0, 3.0, 4.0, 5.0, 12.0])?;

    assert_tensor_arrays_eq(taken.into_array(), expected)
}

#[test]
fn scalar_at_reads_a_single_denormalized_row() -> VortexResult<()> {
    let input = vector_array(2, &[3.0, 4.0, 5.0, 12.0])?;
    let mut ctx = SESSION.create_execution_ctx();
    let normalized_array = normalize(input.clone(), &mut ctx)?.into_array();

    for i in 0..input.len() {
        assert_eq!(
            normalized_array.execute_scalar(i, &mut ctx)?,
            input.execute_scalar(i, &mut ctx)?,
        );
    }

    Ok(())
}

#[test]
fn scalar_at_reads_a_nullable_column() -> VortexResult<()> {
    let input = nullable_vector_input()?;
    let mut ctx = SESSION.create_execution_ctx();
    let normalized_array = normalize(input.clone(), &mut ctx)?.into_array();

    for i in 0..input.len() {
        assert_eq!(
            normalized_array.execute_scalar(i, &mut ctx)?,
            input.execute_scalar(i, &mut ctx)?,
        );
    }

    Ok(())
}

// =============================================================================
// Serialization
// =============================================================================

#[rstest]
#[case::vector(vector_array(3, &[3.0, 4.0, 0.0, 0.0, 0.0, 0.0]).expect("valid vector array"))]
#[case::fixed_shape_tensor(
    tensor_array(&[2, 2], &[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]).expect("valid tensor array")
)]
#[case::nullable_vector(nullable_vector_input().expect("valid vector array"))]
fn serde_round_trip(#[case] input: ArrayRef) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let original = normalize(input, &mut ctx)?.into_array();
    let children: Vec<ArrayRef> = original.children();

    let metadata = SESSION
        .array_serialize(&original)?
        .expect("Normalized must serialize");
    let recovered = ArrayPlugin::deserialize(
        &Normalized,
        original.dtype(),
        original.len(),
        &metadata,
        &[],
        &children,
        &SESSION,
    )?;

    assert_eq!(recovered.encoding_id(), ArrayVTable::id(&Normalized));
    assert_eq!(recovered.dtype(), original.dtype());
    assert_eq!(recovered.len(), original.len());
    assert_tensor_arrays_eq(recovered, original)
}

#[test]
fn serialization_carries_no_metadata() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let nullable = normalize(nullable_vector_input()?, &mut ctx)?.into_array();
    let non_nullable = normalize(vector_array(2, &[3.0, 4.0, 1.0, 0.0])?, &mut ctx)?.into_array();

    for array in [&nullable, &non_nullable] {
        let bytes = SESSION
            .array_serialize(array)?
            .expect("Normalized must serialize");
        assert!(bytes.is_empty(), "Normalized must not serialize metadata");
    }

    assert_eq!(nullable.nchildren(), NormalizedSlots::COUNT);
    assert_eq!(non_nullable.nchildren(), NormalizedSlots::COUNT - 1);

    Ok(())
}

#[test]
fn serde_round_trip_of_a_nullable_column_with_no_null_rows() -> VortexResult<()> {
    // AllValid omits the validity child, so deserialization must recover nullability from the
    // parent dtype.
    let normalized = vector_array(2, &[0.6f64, 0.8, 1.0, 0.0])?;
    let norms = PrimitiveArray::from_iter([5.0f64, 1.0]).into_array();

    let mut ctx = SESSION.create_execution_ctx();
    let original =
        Normalized::try_new(normalized, norms, Validity::AllValid, &mut ctx)?.into_array();
    let children: Vec<ArrayRef> = original.children();

    assert!(original.dtype().is_nullable());
    assert_eq!(children.len(), NormalizedSlots::COUNT - 1);

    let metadata = SESSION
        .array_serialize(&original)?
        .expect("Normalized must serialize");
    let recovered = ArrayPlugin::deserialize(
        &Normalized,
        original.dtype(),
        original.len(),
        &metadata,
        &[],
        &children,
        &SESSION,
    )?;

    assert_eq!(recovered.dtype(), original.dtype());
    assert!(matches!(recovered.validity()?, Validity::AllValid));

    Ok(())
}

#[test]
fn deserialize_rejects_validity_child_for_non_nullable_dtype() -> VortexResult<()> {
    let normalized = vector_array(2, &[0.6f64, 0.8, 1.0, 0.0])?;
    let dtype = normalized.dtype().clone();
    let children = vec![
        normalized,
        PrimitiveArray::from_iter([5.0f64, 1.0]).into_array(),
        BoolArray::from_iter([true, false]).into_array(),
    ];

    let error = ArrayPlugin::deserialize(&Normalized, &dtype, 2, &[], &[], &children, &SESSION)
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("Normalized validity child requires a nullable dtype")
    );

    Ok(())
}

#[test]
fn encoding_is_registered_under_the_normalized_id() {
    let id = ArrayVTable::id(&Normalized);

    assert_eq!(id.as_ref(), "vortex.tensor.normalized");
    assert!(SESSION.arrays().registry().contains_key(&id));
}

// =============================================================================
// Compression
// =============================================================================

fn collinear_vectors(rows: usize) -> VortexResult<ArrayRef> {
    let elements: Vec<f64> = (0..rows)
        .flat_map(|i| {
            let scale = 1.0 + i as f64;
            [3.0 * scale, 4.0 * scale, 12.0 * scale, 0.0]
        })
        .collect();

    vector_array(4, &elements)
}

#[rstest]
#[case::vector(vector_array(2, &[3.0, 4.0, 5.0, 12.0]).expect("valid vector array"))]
#[case::fixed_shape_tensor(
    tensor_array(&[2, 2], &[1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 0.0, 0.0]).expect("valid tensor array")
)]
fn scheme_matches_tensor_columns(#[case] input: ArrayRef) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let canonical: Canonical = input.execute(&mut ctx)?;

    assert!(NormalizedScheme.matches(&canonical));
    assert_eq!(
        NormalizedScheme.produced_encodings(),
        vec![ArrayVTable::id(&Normalized)]
    );

    Ok(())
}

#[rstest]
#[case::integer_tensor(tensor_array(&[2], &[1i32, 2, 3, 4]).expect("valid tensor array"))]
#[case::non_tensor_extension(non_tensor_extension_array().expect("valid date array"))]
fn scheme_does_not_match_non_float_tensors(#[case] input: ArrayRef) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let canonical: Canonical = input.clone().execute(&mut ctx)?;

    assert!(!NormalizedScheme.matches(&canonical));

    let compressor = BtrBlocksCompressorBuilder::default()
        .with_new_scheme(&NormalizedScheme)
        .build();
    let compressed = compressor.compress(&input, &mut ctx)?;

    assert_ne!(compressed.encoding_id(), ArrayVTable::id(&Normalized));

    Ok(())
}

#[rstest]
#[case::non_nullable(collinear_vectors(1024).expect("valid vector array"))]
#[case::nullable(nullable_collinear_vectors(1024).expect("valid vector array"))]
fn compressor_emits_the_dedicated_encoding(#[case] input: ArrayRef) -> VortexResult<()> {
    let compressor = BtrBlocksCompressorBuilder::default()
        .with_new_scheme(&NormalizedScheme)
        .build();

    let mut ctx = SESSION.create_execution_ctx();
    let compressed = compressor.compress(&input, &mut ctx)?;

    assert_eq!(compressed.encoding_id(), ArrayVTable::id(&Normalized));
    assert!(compressed.nbytes() < input.nbytes());
    assert_tensor_arrays_eq(compressed, input)
}

fn nullable_collinear_vectors(rows: usize) -> VortexResult<ArrayRef> {
    let vectors = collinear_vectors(rows)?;
    let validity = Validity::from_iter((0..rows).map(|i| i % 8 != 0));

    Ok(MaskedArray::try_new(vectors, validity)?.into_array())
}
