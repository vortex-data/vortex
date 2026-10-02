// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use rstest::rstest;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use super::slices::take_slices;
use super::slices::take_slices_constant_length;
use super::take_values;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array::Array;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::PiecewiseSequenceArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::fixed_width::FixedWidthArray;
use crate::arrays::fixed_width::with_values;
use crate::assert_arrays_eq;
use crate::compute::conformance::take::test_take_conformance;
use crate::dtype::DecimalDType;
use crate::dtype::half::f16;
use crate::dtype::i256;
use crate::validity::Validity;

fn allocator() -> BufferAllocatorRef {
    BufferAllocatorRef::statically_allocated()
}

#[test]
fn take_four_byte_records() {
    let values = [[1u8, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]];
    let taken = take_values(&values, &[2u32, 0], &allocator());
    assert_eq!(taken.as_slice(), &[[9, 10, 11, 12], [1, 2, 3, 4]]);
}

#[test]
fn take_eight_byte_values() {
    let taken = take_values(&[10i64, 20, 30], &[1u16, 2, 0], &allocator());
    assert_eq!(taken.as_slice(), &[20, 30, 10]);
}

#[rstest]
#[case::u8(PrimitiveArray::from_iter([10u8, 20, 30]))]
#[case::u16(PrimitiveArray::from_iter([10u16, 20, 30]))]
#[case::u32(PrimitiveArray::from_iter([10u32, 20, 30]))]
#[case::u64(PrimitiveArray::from_iter([10u64, 20, 30]))]
#[case::u128(DecimalArray::new(
    buffer![10i128, -20, -30],
    DecimalDType::new(19, 0),
    Validity::NonNullable,
))]
#[case::i256(DecimalArray::new(
    Buffer::from_iter([10, -20, -30].map(|value| i256::from_parts(1, value))),
    DecimalDType::new(76, 0),
    Validity::NonNullable,
))]
fn take_typed_records<V: FixedWidthArray>(
    #[case] array: Array<V>,
    #[values(
        PrimitiveArray::from_iter([2u8, 0]),
        PrimitiveArray::from_iter([2u16, 0]),
        PrimitiveArray::from_iter([2u32, 0]),
        PrimitiveArray::from_iter([2u64, 0])
    )]
    indices: PrimitiveArray,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let byte_width = V::byte_width(array.as_view());
    let values = V::values::<u8>(array.as_view()).aligned(Alignment::new(64));
    let expected = values[2 * byte_width..3 * byte_width]
        .iter()
        .chain(&values[..byte_width])
        .copied()
        .collect::<Vec<_>>();
    let array = with_values(array.as_view(), values, array.len(), Validity::NonNullable)?;
    let alignment = V::values::<u8>(array.as_view()).alignment();
    let taken = super::take(array.as_view(), &indices.into_array(), &mut ctx)?
        .expect("native-width take must be supported");
    let buffers = taken.buffers();
    assert_eq!(buffers[0].as_slice(), expected);
    assert_eq!(buffers[0].alignment(), alignment);
    Ok(())
}

#[test]
#[should_panic(expected = "out of bounds")]
fn typed_take_rejects_out_of_bounds_index() {
    let mut ctx = array_session().create_execution_ctx();
    let array = PrimitiveArray::from_iter([10u32, 20, 30]);
    let indices = PrimitiveArray::from_iter([3u32]).into_array();
    drop(super::take(array.as_view(), &indices, &mut ctx));
}

#[rstest]
#[case::u8(buffer![10u8, 11, 12, 13, 14])]
#[case::u16(buffer![10u16, 11, 12, 13, 14])]
#[case::u32(buffer![10u32, 11, 12, 13, 14])]
#[case::u64(buffer![10u64, 11, 12, 13, 14])]
#[case::u128(buffer![10u128, 11, 12, 13, 14])]
#[case::i256(Buffer::from_iter((10..15).map(i256::from_i128)))]
#[case::three_bytes(buffer![[10u8; 3], [11; 3], [12; 3], [13; 3], [14; 3]])]
#[case::twelve_bytes(buffer![[10u8; 12], [11; 12], [12; 12], [13; 12], [14; 12]])]
fn take_typed_slices<T: Copy + Debug + Eq>(#[case] values: Buffer<T>) -> VortexResult<()> {
    let values = values.aligned(Alignment::new(64));
    let taken = take_slices(&values, &[1u32, 3], &[2u32, 1], 3, &allocator())?;
    assert_eq!(taken.as_slice(), &values[1..4]);
    assert_eq!(taken.alignment(), values.alignment());

    let taken = take_slices_constant_length(&values, &[0u32, 3], 2, 4, &allocator())?;
    let expected = [&values[..2], &values[3..]].concat();
    assert_eq!(taken.as_slice(), expected);
    assert_eq!(taken.alignment(), values.alignment());
    Ok(())
}

#[test]
fn variable_length_slices_validate_output_length() {
    let values = buffer![10u8, 11, 12, 13];
    assert!(take_slices(&values, &[0u32, 2], &[1u32, 1], 3, &allocator()).is_err());
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

#[rstest]
#[case::u8(PrimitiveArray::from_iter([10u8, 20, 30, 40, 50]).into_array())]
#[case::f16(PrimitiveArray::from_iter([10.0, -20.0, 30.0, -40.0, 50.0].map(f16::from_f32)).into_array())]
#[case::f32(PrimitiveArray::from_iter([10f32, -20.0, 30.0, -40.0, 50.0]).into_array())]
#[case::f64(PrimitiveArray::from_iter([10f64, -20.0, 30.0, -40.0, 50.0]).into_array())]
#[case::decimal_i128(DecimalArray::new(
    buffer![100i128, -200, 300, -400, 500],
    DecimalDType::new(19, 2),
    Validity::NonNullable,
).into_array())]
#[case::decimal_i256(DecimalArray::new(
    Buffer::from_iter([100, -200, 300, -400, 500].map(i256::from_i128)),
    DecimalDType::new(76, 2),
    Validity::NonNullable,
).into_array())]
fn fixed_width_take_consumes_piecewise_indices(
    #[case] values: ArrayRef,
    #[values(false, true)] constant_length: bool,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let starts = PrimitiveArray::from_iter([1u64, 3]).into_array();
    let (lengths, output_len) = if constant_length {
        (ConstantArray::new(2u64, 2).into_array(), 4)
    } else {
        (PrimitiveArray::from_iter([2u64, 1]).into_array(), 3)
    };
    let multipliers = ConstantArray::new(1u64, 2).into_array();
    let indices =
        PiecewiseSequenceArray::try_new(starts, lengths, multipliers, output_len)?.into_array();

    let taken = values.take(indices)?;

    assert_arrays_eq!(taken, values.slice(1..1 + output_len)?, &mut ctx);
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
