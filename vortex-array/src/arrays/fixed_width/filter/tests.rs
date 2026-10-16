// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::atomic::Ordering;

use rstest::rstest;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array::Array;
use crate::array_session;
use crate::arrays::DecimalArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::fixed_width::FixedWidthArray;
use crate::arrays::fixed_width::with_values;
use crate::compute::conformance::filter::LARGE_SIZE;
use crate::compute::conformance::filter::MEDIUM_SIZE;
use crate::compute::conformance::filter::test_filter_conformance;
use crate::dtype::DecimalDType;
use crate::dtype::i256;
use crate::memory::test_allocator::counting_allocator;
use crate::validity::Validity;

#[rstest]
#[case::u8(PrimitiveArray::from_iter([10u8, 20, 30, 40]))]
#[case::u16(PrimitiveArray::from_iter([10u16, 20, 30, 40]))]
#[case::u32(PrimitiveArray::from_iter([10u32, 20, 30, 40]))]
#[case::u64(PrimitiveArray::from_iter([10u64, 20, 30, 40]))]
#[case::u128(DecimalArray::new(
    buffer![10i128, -20, -30, -40],
    DecimalDType::new(19, 0),
    Validity::NonNullable,
))]
#[case::i256(DecimalArray::new(
    Buffer::from_iter([10, -20, -30, -40].map(|value| i256::from_parts(1, value))),
    DecimalDType::new(76, 0),
    Validity::NonNullable,
))]
fn filter_typed_records<V: FixedWidthArray>(#[case] array: Array<V>) -> VortexResult<()> {
    let Mask::Values(mask) = Mask::from_iter([true, false, true, false]) else {
        panic!("a mixed mask must have mask values");
    };
    let byte_width = V::byte_width(array.as_view());
    let values = V::values::<u8>(array.as_view()).aligned(Alignment::new(64));
    let expected = values[..byte_width]
        .iter()
        .chain(&values[2 * byte_width..3 * byte_width])
        .copied()
        .collect::<Vec<_>>();
    let array = with_values(array.as_view(), values, array.len(), Validity::NonNullable)?;
    let alignment = V::values::<u8>(array.as_view()).alignment();
    let filtered = super::filter(&array, &mask, BufferAllocatorRef::static_ref()).into_array();
    let buffers = filtered.buffers();
    assert_eq!(buffers[0].as_slice(), expected);
    assert_eq!(buffers[0].alignment(), alignment);
    Ok(())
}

#[test]
fn filter_records_use_execution_allocator() {
    let Mask::Values(mask) = Mask::from_iter([true, false, true, false]) else {
        panic!("a mixed mask must have mask values");
    };
    let (allocator, allocations) = counting_allocator();
    let array = PrimitiveArray::from_iter([0u32, 1, 2, 3]);
    drop(super::filter(&array, &mask, &allocator));
    let array = DecimalArray::new(
        Buffer::from_iter([0, 1, 2, 3].map(i256::from_i128)),
        DecimalDType::new(76, 0),
        Validity::NonNullable,
    );
    drop(super::filter(&array, &mask, &allocator));
    assert_eq!(allocations.load(Ordering::Relaxed), 2);
}

#[rstest]
#[case::primitive_i8(PrimitiveArray::from_iter([-2i8, -1, 0, 1, 2]).into_array())]
#[case::primitive_u16(PrimitiveArray::from_iter([1u16, 2, 3, 4, 5]).into_array())]
#[case::primitive_i32(PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]).into_array())]
#[case::primitive_f32(PrimitiveArray::from_iter([0.1f32, 0.2, 0.3, 0.4, 0.5]).into_array())]
#[case::primitive_nullable(PrimitiveArray::from_option_iter(
    [Some(1i64), None, Some(3), Some(4), None],
).into_array())]
#[case::primitive_large(PrimitiveArray::from_iter(0..LARGE_SIZE as u32).into_array())]
#[case::primitive_medium(PrimitiveArray::from_iter(0..MEDIUM_SIZE as i64).into_array())]
#[case::decimal_i8(DecimalArray::new(
    buffer![1i8, 2, 3, 4, 5],
    DecimalDType::new(2, 0),
    Validity::NonNullable,
).into_array())]
#[case::decimal_i32(DecimalArray::new(
    buffer![123i32, 456, -123, 0, 999],
    DecimalDType::new(8, 2),
    Validity::NonNullable,
).into_array())]
#[case::decimal_i64(DecimalArray::new(
    buffer![12345i64, 67890, -12345, 0, 99999],
    DecimalDType::new(18, 2),
    Validity::NonNullable,
).into_array())]
#[case::decimal_i128(DecimalArray::new(
    buffer![12345i128, 67890, -12345, 0, 99999],
    DecimalDType::new(38, 4),
    Validity::from_iter([true, false, true, true, false]),
).into_array())]
#[case::decimal_i256(DecimalArray::new(
    buffer![
        i256::from_i128(12345),
        i256::from_i128(67890),
        i256::from_i128(-12345),
        i256::ZERO,
        i256::from_i128(99999),
    ],
    DecimalDType::new(76, 4),
    Validity::NonNullable,
).into_array())]
fn fixed_width_filter_conformance(#[case] array: ArrayRef) {
    test_filter_conformance(&array, &mut array_session().create_execution_ctx());
}
