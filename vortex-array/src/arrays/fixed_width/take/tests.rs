// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use super::records::take_byte_records;
use super::slices::take_slices;
use super::slices::take_slices_constant_length;
use super::take_values;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::PiecewiseSequenceArray;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::compute::conformance::take::test_take_conformance;
use crate::dtype::DecimalDType;
use crate::dtype::i256;
use crate::validity::Validity;

#[test]
fn take_four_byte_records() {
    let values = [[1u8, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]];
    let taken = take_values(&values, &[2u32, 0]);
    assert_eq!(taken.as_slice(), &[[9, 10, 11, 12], [1, 2, 3, 4]]);
}

#[test]
fn take_eight_byte_values() {
    let taken = take_values(&[10i64, 20, 30], &[1u16, 2, 0]);
    assert_eq!(taken.as_slice(), &[20, 30, 10]);
}

#[rstest]
#[case(1)]
#[case(2)]
#[case(4)]
#[case(8)]
#[case(16)]
#[case(32)]
#[case::fallback(3)]
#[case::fallback_wide(12)]
fn take_runtime_width_records(#[case] byte_width: usize) -> VortexResult<()> {
    let values = Buffer::from_iter((0u8..).take(3 * byte_width));
    let expected = values[2 * byte_width..3 * byte_width]
        .iter()
        .chain(&values[..byte_width])
        .copied()
        .collect::<Vec<_>>();
    let taken = take_byte_records(&values.into_byte_buffer(), byte_width, 3, &[2u32, 0])?;
    assert_eq!(taken.as_slice(), expected);
    Ok(())
}

#[test]
#[should_panic(expected = "take index 3 out of bounds for length 3")]
fn fallback_take_rejects_out_of_bounds_index() {
    let values = Buffer::from_iter((0u8..).take(9)).into_byte_buffer();
    drop(take_byte_records(&values, 3, 3, &[3u32]));
}

#[test]
fn take_variable_length_slices() -> VortexResult<()> {
    let values = buffer![10u8, 11, 12, 13, 14].into_byte_buffer();
    let taken = take_slices(&values, 1, 5, &[1u32, 3], &[2u32, 1], 3)?;
    assert_eq!(taken.as_slice(), &[11, 12, 13]);
    Ok(())
}

#[test]
fn variable_length_slices_validate_output_length() {
    let values = buffer![10u8, 11, 12, 13].into_byte_buffer();
    assert!(take_slices(&values, 1, 4, &[0u32, 2], &[1u32, 1], 3).is_err());
}

#[test]
fn take_constant_length_slices() -> VortexResult<()> {
    let values = buffer![10u8, 11, 12, 13, 14].into_byte_buffer();
    let taken = take_slices_constant_length(&values, 1, 5, &[0u32, 3], 2, 4)?;
    assert_eq!(taken.as_slice(), &[10, 11, 13, 14]);
    Ok(())
}

#[test]
fn null_index_skips_out_of_bounds_primitive_value() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let values = PrimitiveArray::from_iter([10i32, 20, 30]);
    let indices = PrimitiveArray::new(
        buffer![1u64, 3],
        Validity::Array(BoolArray::from_iter([true, false]).into_array()),
    );

    let taken = values.take(indices.into_array())?;

    assert_arrays_eq!(
        taken,
        PrimitiveArray::from_option_iter([Some(20i32), None]).into_array(),
        &mut ctx
    );
    Ok(())
}

#[test]
fn null_index_skips_out_of_bounds_decimal_value() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let decimal_dtype = DecimalDType::new(19, 1);
    let values = DecimalArray::new(
        buffer![10i128, 20, 30],
        decimal_dtype,
        Validity::NonNullable,
    );
    let indices = PrimitiveArray::new(
        buffer![1u64, 3],
        Validity::Array(BoolArray::from_iter([true, false]).into_array()),
    );

    let taken = values.take(indices.into_array())?;

    assert_arrays_eq!(
        taken,
        DecimalArray::from_option_iter([Some(20i128), None], decimal_dtype).into_array(),
        &mut ctx
    );
    Ok(())
}

#[test]
fn decimal_i256_take_consumes_piecewise_indices() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let decimal_dtype = DecimalDType::new(76, 2);
    let values = DecimalArray::new(
        buffer![
            i256::from_i128(100),
            i256::from_i128(200),
            i256::from_i128(300),
            i256::from_i128(400),
            i256::from_i128(500),
        ],
        decimal_dtype,
        Validity::NonNullable,
    );
    let starts = PrimitiveArray::from_iter([1u64, 3]).into_array();
    let lengths = PrimitiveArray::from_iter([2u64, 1]).into_array();
    let multipliers = ConstantArray::new(1u64, 2).into_array();
    let indices = PiecewiseSequenceArray::try_new(starts, lengths, multipliers, 3)?.into_array();

    let taken = values.take(indices)?;

    let expected = DecimalArray::new(
        buffer![
            i256::from_i128(200),
            i256::from_i128(300),
            i256::from_i128(400),
        ],
        decimal_dtype,
        Validity::NonNullable,
    );
    assert_arrays_eq!(taken, expected, &mut ctx);
    Ok(())
}

#[rstest]
#[case::primitive(PrimitiveArray::new(
    buffer![0i32, 1, 2, 3, 4],
    Validity::NonNullable,
).into_array())]
#[case::primitive_nullable(PrimitiveArray::from_option_iter(
    [Some(1i64), None, Some(3), Some(4), None],
).into_array())]
#[case::decimal_i32(DecimalArray::new(
    buffer![1i32, 2, 3, 4, 5],
    DecimalDType::new(5, 0),
    Validity::NonNullable,
).into_array())]
#[case::decimal_i64(DecimalArray::new(
    buffer![10i64, 20, 30, 40, 50],
    DecimalDType::new(10, 1),
    Validity::NonNullable,
).into_array())]
#[case::decimal_i128(DecimalArray::new(
    buffer![100i128, 200, 300, 400, 500],
    DecimalDType::new(19, 2),
    Validity::from_iter([true, false, true, true, false]),
).into_array())]
#[case::decimal_i256(DecimalArray::new(
    buffer![
        i256::from_i128(100),
        i256::from_i128(200),
        i256::from_i128(300),
        i256::from_i128(400),
        i256::from_i128(500),
    ],
    DecimalDType::new(76, 2),
    Validity::NonNullable,
).into_array())]
fn fixed_width_take_conformance(#[case] array: ArrayRef) {
    test_take_conformance(&array, &mut array_session().create_execution_ctx());
}
