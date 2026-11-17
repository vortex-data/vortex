// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayParts;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::ConstantArray;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::assert_arrays_eq;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::Nullability;
use crate::match_each_decimal_value_type;
use crate::validity::Validity;

#[rstest]
#[case::i8(DecimalType::I8, 3)]
#[case::i16(DecimalType::I16, 2)]
#[case::i32(DecimalType::I32, 2)]
#[case::i64(DecimalType::I64, 2)]
#[case::i128(DecimalType::I128, 2)]
#[case::i256(DecimalType::I256, 2)]
fn typed_buffer_must_have_the_canonical_width(
    #[case] values_type: DecimalType,
    #[case] precision: u8,
) {
    let dtype = DecimalDType::new(precision, 0);
    let result = match_each_decimal_value_type!(values_type, |D| {
        DecimalArray::try_new(Buffer::<D>::empty(), dtype, Validity::NonNullable)
    });
    let expected = dtype.canonical_values_type();
    let error = result.unwrap_err().to_string();

    assert!(
        error.contains(&format!(
            "{dtype} requires {expected} values, got {values_type}"
        )),
        "{error}"
    );
}

#[test]
#[should_panic(expected = "decimal(2,0) requires i8 values, got i32")]
fn unchecked_construction_checks_width() {
    // SAFETY: The typed buffer contains one value and needs no validity child. A width mismatch
    // is a checked programmer error.
    unsafe {
        DecimalArray::new_unchecked(
            buffer![1i32],
            DecimalDType::new(2, 0),
            Validity::NonNullable,
        )
    };
}

#[test]
fn iterators_convert_to_the_canonical_width() -> VortexResult<()> {
    let dtype = DecimalDType::new(5, 2);
    let widened = DecimalArray::from_iter([-17i8, 0, 99], dtype);
    assert_eq!(widened.buffer::<i32>(), buffer![-17i32, 0, 99]);
    assert_eq!(
        widened.dtype(),
        &DType::Decimal(dtype, Nullability::NonNullable)
    );

    let narrowed = DecimalArray::from_option_iter([Some(-17i128), None, Some(99)], dtype);
    assert_eq!(narrowed.buffer::<i32>(), buffer![-17i32, 0, 99]);
    assert_arrays_eq!(
        narrowed,
        DecimalArray::try_new(
            buffer![-17i32, 0, 99],
            dtype,
            Validity::from_iter([true, false, true])
        )?,
        &mut VortexSession::empty().create_execution_ctx()
    );

    Ok(())
}

#[test]
#[should_panic(expected = "Decimal value must fit in i8, got 128")]
fn iterators_reject_values_outside_the_canonical_integer_type() {
    DecimalArray::from_iter([128i16], DecimalDType::new(2, 0));
}

/// Parts with one valid value, for tests that replace one part with an invalid one.
fn decimal_parts() -> ArrayParts<Decimal> {
    DecimalArray::new(buffer![1i32], DecimalDType::new(5, 2), Validity::AllValid)
        .try_into_parts()
        .unwrap()
}

#[rstest]
#[case::precision(6, 2)]
#[case::scale(5, 1)]
fn parts_require_matching_decimal_dtype(#[case] precision: u8, #[case] scale: i8) {
    let mut parts = decimal_parts();
    parts.dtype = DType::Decimal(DecimalDType::new(precision, scale), Nullability::Nullable);

    let error = DecimalArray::try_from_parts(parts).unwrap_err().to_string();
    assert!(error.contains("precision and scale"), "{error}");
}

#[test]
fn parts_reject_a_partial_integer_after_the_last_row() {
    let mut parts = decimal_parts();
    let bytes = buffer![1i32, 2].into_byte_buffer().slice_unaligned(..5);
    parts.data.values = BufferHandle::new_host(bytes);

    let error = DecimalArray::try_from_parts(parts).unwrap_err().to_string();
    assert!(error.contains("not divisible"), "{error}");
}

#[test]
fn buffer_replacement_requires_alignment() {
    let array = DecimalArray::new(
        buffer![0i32],
        DecimalDType::new(5, 2),
        Validity::NonNullable,
    );
    let unaligned = buffer![0i32, 0].into_byte_buffer().slice_unaligned(1..5);
    let buffers = [BufferHandle::new_host(unaligned)];

    // SAFETY: The replacement contains the same four zero bytes as the original buffer.
    let result = unsafe { array.into_array().with_buffers(buffers) };
    let error = result.unwrap_err().to_string();
    assert!(error.contains("alignment"), "{error}");
}

#[rstest]
#[case::wrong_length(ConstantArray::new(false, 2).into_array(), Nullability::Nullable, "one boolean per row")]
#[case::wrong_dtype(ConstantArray::new(1i32, 1).into_array(), Nullability::Nullable, "non-nullable booleans")]
#[case::non_nullable(ConstantArray::new(true, 1).into_array(), Nullability::NonNullable, "requires a nullable dtype")]
fn parts_reject_invalid_validity(
    #[case] validity: ArrayRef,
    #[case] nullability: Nullability,
    #[case] message: &str,
) {
    let mut parts = decimal_parts();
    parts.dtype = DType::Decimal(parts.data.decimal_dtype, nullability);
    parts.slots[0] = Some(validity);

    let error = DecimalArray::try_from_parts(parts).unwrap_err().to_string();
    assert!(error.contains(message), "{error}");
}

#[test]
fn parts_require_the_validity_slot() {
    let mut parts = decimal_parts();
    parts.slots.clear();

    let error = DecimalArray::try_from_parts(parts).unwrap_err().to_string();
    assert!(error.contains("one validity slot"), "{error}");
}
