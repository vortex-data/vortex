// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::ArrayParts;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::decimal::DecimalData;
use crate::assert_arrays_eq;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::Nullability;
use crate::match_each_decimal_value_type;
use crate::validity::Validity;

#[rstest]
fn construction_bounds_storage_by_precision(
    #[values(1, 2, 3, 4, 5, 9, 10, 18, 19, 38, 39, 76)] precision: u8,
    #[values(
        DecimalType::I8, DecimalType::I16, DecimalType::I32,
        DecimalType::I64, DecimalType::I128, DecimalType::I256
    )]
    storage: DecimalType,
) -> VortexResult<()> {
    let dtype = DecimalDType::new(precision, 0);
    let maximum = DecimalType::smallest_decimal_value_type(&dtype);
    let bytes = match_each_decimal_value_type!(storage, |D| {
        [-9i8, 9]
            .into_iter()
            .map(|v| v.as_())
            .collect::<Buffer<D>>()
            .into_byte_buffer()
    });
    let ptr = bytes.as_ptr();
    let array = DecimalArray::try_new_handle(
        BufferHandle::new_host(bytes),
        storage,
        dtype,
        Validity::NonNullable,
    )?;
    assert_eq!(array.values_type(), storage.min(maximum));
    if storage <= maximum {
        assert_eq!(array.buffer_handle().as_host().as_ptr(), ptr);
    }
    assert_arrays_eq!(
        array,
        DecimalArray::new(buffer![-9i8, 9], dtype, Validity::NonNullable),
        &mut array_session().create_execution_ctx()
    );
    Ok(())
}

#[test]
fn narrowing_preserves_validity_and_ignores_null_payloads() -> VortexResult<()> {
    let dtype = DecimalDType::new(2, 0);
    let array = DecimalArray::try_new(
        buffer![99i128, i128::MIN, -99],
        dtype,
        Validity::from_iter([true, false, true]),
    )?;
    assert_eq!(array.values_type(), DecimalType::I8);
    assert_arrays_eq!(
        array,
        DecimalArray::from_option_iter([Some(99i8), None, Some(-99)], dtype),
        &mut array_session().create_execution_ctx()
    );
    Ok(())
}

#[test]
fn empty_oversized_storage_is_narrowed() {
    let data = DecimalData::new(Buffer::<i128>::empty(), DecimalDType::new(2, 0));
    assert!(data.is_empty());
    assert_eq!(data.values_type(), DecimalType::I8);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "decimal storage i16 exceeds")]
fn unchecked_construction_asserts_storage_bound() {
    // Deliberately violate the width contract to exercise its debug assertion.
    let _ = unsafe { DecimalData::new_unchecked(buffer![1i16], DecimalDType::new(2, 0)) };
}

#[test]
fn parts_cannot_override_the_precision_bound() {
    let data = DecimalData::new(buffer![1i64], DecimalDType::new(18, 0));
    let result = DecimalArray::try_from_parts(
        ArrayParts::new(
            Decimal,
            DType::Decimal(DecimalDType::new(2, 0), Nullability::NonNullable),
            1,
            data,
        )
        .with_slots(DecimalData::make_slots(&Validity::NonNullable, 1)),
    );
    assert!(result.is_err());
}
