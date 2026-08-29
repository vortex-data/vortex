// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_mask::Mask;

use super::filter_records;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::DecimalArray;
use crate::arrays::PrimitiveArray;
use crate::compute::conformance::filter::LARGE_SIZE;
use crate::compute::conformance::filter::MEDIUM_SIZE;
use crate::compute::conformance::filter::test_filter_conformance;
use crate::dtype::DecimalDType;
use crate::dtype::i256;
use crate::validity::Validity;

#[test]
fn filter_fallback_width_records() {
    let Mask::Values(mask) = Mask::from_iter([true, false, true, false]) else {
        panic!("a mixed mask must have mask values");
    };
    let expected = [0u8, 1, 2, 6, 7, 8];

    // A uniquely owned buffer takes the in-place `copy_within` path.
    let owned = Buffer::from_iter(0u8..12);
    let filtered = filter_records(owned, 3, &mask);
    assert_eq!(filtered.as_slice(), &expected);

    // Retaining a second reference forces the copying path instead.
    let shared = Buffer::from_iter(0u8..12);
    let _retained = shared.clone();
    let filtered = filter_records(shared, 3, &mask);
    assert_eq!(filtered.as_slice(), &expected);
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
