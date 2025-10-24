// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::{
    BoolArray, ConstantArray, DecimalArray, PrimitiveArray, VarBinViewArray,
};
use vortex_array::compute::fill_null;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityHelper;
use vortex_array::{ArrayRef, Canonical, IntoArray, ToCanonical};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, Nullability, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, VortexUnwrap};
use vortex_scalar::{Scalar, match_each_decimal_value_type};

/// Apply fill_null on the canonical form of the array to get a consistent baseline.
/// This implementation manually fills null values for each canonical type
/// without using the fill_null method, to serve as an independent baseline for testing.
pub fn fill_null_canonical_array(
    canonical: Canonical,
    fill_value: &Scalar,
) -> VortexResult<ArrayRef> {
    let result_nullability = fill_value.dtype().nullability();

    Ok(match canonical {
        Canonical::Null(array) => ConstantArray::new(fill_value.clone(), array.len()).into_array(),
        Canonical::Bool(array) => fill_bool_array(&array, fill_value, result_nullability),
        Canonical::Primitive(array) => fill_primitive_array(&array, fill_value, result_nullability),
        Canonical::Decimal(array) => fill_decimal_array(&array, fill_value, result_nullability),
        Canonical::VarBinView(array) => {
            fill_varbinview_array(&array, fill_value, result_nullability)
        }
        Canonical::Struct(_)
        | Canonical::List(_)
        | Canonical::FixedSizeList(_)
        | Canonical::Extension(_) => fill_null(canonical.as_ref(), fill_value)?,
    })
}

fn fill_bool_array(
    array: &BoolArray,
    fill_value: &Scalar,
    result_nullability: Nullability,
) -> ArrayRef {
    let fill_bool = fill_value
        .as_bool()
        .value()
        .vortex_expect("cannot have null fill value");

    match array.validity() {
        Validity::NonNullable | Validity::AllValid => {
            BoolArray::from_bit_buffer(array.bit_buffer().clone(), result_nullability.into())
                .into_array()
        }
        Validity::AllInvalid => ConstantArray::new(fill_value.clone(), array.len()).into_array(),
        Validity::Array(validity_array) => {
            let validity_bool_array = validity_array.to_bool();
            let validity_bits = validity_bool_array.bit_buffer();
            let data_bits = array.bit_buffer();

            let mut new_bits = data_bits.clone().into_mut();

            (!validity_bits)
                .set_indices()
                .for_each(|i| new_bits.set_to(i, fill_bool));

            BoolArray::from_bit_buffer(new_bits.freeze(), result_nullability.into()).into_array()
        }
    }
}

#[allow(clippy::cognitive_complexity)]
fn fill_primitive_array(
    array: &PrimitiveArray,
    fill_value: &Scalar,
    result_nullability: Nullability,
) -> ArrayRef {
    match_each_native_ptype!(array.ptype(), |T| {
        let fill_val = T::try_from(fill_value).vortex_unwrap();

        match array.validity() {
            Validity::NonNullable | Validity::AllValid => PrimitiveArray::from_byte_buffer(
                array.byte_buffer().clone(),
                array.ptype(),
                result_nullability.into(),
            )
            .into_array(),
            Validity::AllInvalid => {
                ConstantArray::new(fill_value.clone(), array.len()).into_array()
            }
            Validity::Array(validity_array) => {
                let validity_bool_array = validity_array.to_bool();
                let validity_bits = validity_bool_array.bit_buffer();
                let data_slice = array.as_slice::<T>();

                let mut new_data = Vec::with_capacity(array.len());
                for i in 0..array.len() {
                    if validity_bits.value(i) {
                        new_data.push(data_slice[i]);
                    } else {
                        new_data.push(fill_val);
                    }
                }

                PrimitiveArray::new::<T>(Buffer::from(new_data), result_nullability.into())
                    .into_array()
            }
        }
    })
}

#[allow(clippy::cognitive_complexity, clippy::unwrap_used)]
fn fill_decimal_array(
    array: &DecimalArray,
    fill_value: &Scalar,
    result_nullability: Nullability,
) -> ArrayRef {
    let decimal_dtype = array.decimal_dtype();
    let decimal_scalar = fill_value.as_decimal();

    match_each_decimal_value_type!(array.values_type(), |D| {
        let fill_val = D::try_from(decimal_scalar).vortex_unwrap();

        match array.validity() {
            Validity::NonNullable | Validity::AllValid => DecimalArray::new(
                array.buffer::<D>(),
                decimal_dtype,
                result_nullability.into(),
            )
            .into_array(),
            Validity::AllInvalid => {
                ConstantArray::new(fill_value.clone(), array.len()).into_array()
            }
            Validity::Array(validity_array) => {
                let validity_bool_array = validity_array.to_bool();
                let validity_bits = validity_bool_array.bit_buffer();
                let data_buffer = array.buffer::<D>();

                let mut new_data = Vec::with_capacity(array.len());
                for i in 0..array.len() {
                    if validity_bits.value(i) {
                        new_data.push(data_buffer[i]);
                    } else {
                        new_data.push(fill_val);
                    }
                }

                DecimalArray::from_option_iter(new_data.into_iter().map(Some), decimal_dtype)
                    .into_array()
            }
        }
    })
}

fn fill_varbinview_array(
    array: &VarBinViewArray,
    fill_value: &Scalar,
    result_nullability: Nullability,
) -> ArrayRef {
    match array.validity() {
        Validity::NonNullable | Validity::AllValid => array.clone().into_array(),
        Validity::AllInvalid => ConstantArray::new(fill_value.clone(), array.len()).into_array(),
        Validity::Array(validity_array) => {
            let validity_bool_array = validity_array.to_bool();
            let validity_bits = validity_bool_array.bit_buffer();

            match array.dtype() {
                DType::Utf8(_) => {
                    let fill_str = fill_value
                        .as_utf8()
                        .value()
                        .vortex_expect("cannot have null fill value");
                    let strings: Vec<String> = (0..array.len())
                        .map(|i| {
                            if validity_bits.value(i) {
                                array
                                    .scalar_at(i)
                                    .as_utf8()
                                    .value()
                                    .vortex_expect("cannot have null valid value")
                                    .to_string()
                            } else {
                                fill_str.to_string()
                            }
                        })
                        .collect();
                    let string_refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();
                    let result = VarBinViewArray::from_iter_str(string_refs).into_array();
                    if result_nullability == Nullability::Nullable {
                        VarBinViewArray::new(
                            result.to_varbinview().views().clone(),
                            result.to_varbinview().buffers().clone(),
                            result.dtype().as_nullable(),
                            result_nullability.into(),
                        )
                        .into_array()
                    } else {
                        result
                    }
                }
                DType::Binary(_) => {
                    let fill_bytes = fill_value.as_binary().value().unwrap();
                    let binaries: Vec<Vec<u8>> = (0..array.len())
                        .map(|i| {
                            if validity_bits.value(i) {
                                array.scalar_at(i).as_binary().value().unwrap().to_vec()
                            } else {
                                fill_bytes.to_vec()
                            }
                        })
                        .collect();
                    let binary_refs: Vec<&[u8]> = binaries.iter().map(|b| b.as_slice()).collect();
                    let result = VarBinViewArray::from_iter_bin(binary_refs).into_array();
                    // If result_nullability is nullable, cast it
                    if result_nullability == Nullability::Nullable {
                        VarBinViewArray::new(
                            result.to_varbinview().views().clone(),
                            result.to_varbinview().buffers().clone(),
                            result.dtype().as_nullable(),
                            result_nullability.into(),
                        )
                        .into_array()
                    } else {
                        result
                    }
                }
                _ => unreachable!(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::ToCanonical;
    use vortex_array::arrays::{BoolArray, DecimalArray, PrimitiveArray, VarBinViewArray};
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_dtype::{DType, DecimalDType, Nullability, PType};
    use vortex_scalar::{DecimalValue, Scalar};

    use super::fill_null_canonical_array;

    #[test]
    fn test_fill_null_primitive() {
        let array = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5)]);
        let fill_value = Scalar::from(42i32);

        let result = fill_null_canonical_array(array.to_canonical(), &fill_value).unwrap();

        assert_eq!(result.len(), 5);
        assert_eq!(result.scalar_at(0), Scalar::from(1));
        assert_eq!(result.scalar_at(1), Scalar::from(42));
        assert_eq!(result.scalar_at(2), Scalar::from(3));
        assert_eq!(result.scalar_at(3), Scalar::from(42));
        assert_eq!(result.scalar_at(4), Scalar::from(5));
    }

    #[test]
    fn test_fill_null_bool() {
        // Create a bool array with some nulls manually
        let data_buffer = BitBuffer::from(vec![true, false, false, false]);
        let validity_buffer = BitBuffer::from(vec![true, false, true, false]);
        let array = BoolArray::from_bit_buffer(data_buffer, Validity::from(validity_buffer));
        let fill_value = Scalar::from(true);

        let result = fill_null_canonical_array(array.to_canonical(), &fill_value).unwrap();

        assert_eq!(result.len(), 4);
        assert_eq!(result.scalar_at(0), true.into());
        assert_eq!(result.scalar_at(1), true.into());
        assert_eq!(result.scalar_at(2), false.into());
        assert_eq!(result.scalar_at(3), true.into());
    }

    #[test]
    fn test_fill_null_string() {
        let array = VarBinViewArray::from_iter(
            [Some("hello"), None, Some("world")].iter().copied(),
            DType::Utf8(Nullability::Nullable),
        );
        let fill_value = Scalar::utf8("default", Nullability::NonNullable);

        let result = fill_null_canonical_array(array.to_canonical(), &fill_value).unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(
            result.scalar_at(0),
            Scalar::utf8("hello", Nullability::NonNullable)
        );
        assert_eq!(
            result.scalar_at(1),
            Scalar::utf8("default", Nullability::NonNullable)
        );
        assert_eq!(
            result.scalar_at(2),
            Scalar::utf8("world", Nullability::NonNullable)
        );
    }

    #[test]
    fn test_fill_null_all_invalid() {
        let array = PrimitiveArray::from_option_iter([None::<i32>, None, None]);
        let fill_value = Scalar::from(100i32);

        let result = fill_null_canonical_array(array.to_canonical(), &fill_value).unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result.scalar_at(0), Scalar::from(100));
        assert_eq!(result.scalar_at(1), Scalar::from(100));
        assert_eq!(result.scalar_at(2), Scalar::from(100));
    }

    #[test]
    fn test_fill_null_no_nulls() {
        let array = PrimitiveArray::from_iter([1i32, 2, 3]);
        let fill_value = Scalar::from(42i32);

        let result = fill_null_canonical_array(array.to_canonical(), &fill_value).unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result.scalar_at(0), Scalar::from(1));
        assert_eq!(result.scalar_at(1), Scalar::from(2));
        assert_eq!(result.scalar_at(2), Scalar::from(3));
    }

    #[test]
    #[should_panic]
    fn test_fill_null_with_null_value_errors() {
        let array = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]);
        let fill_value = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));

        let result = fill_null_canonical_array(array.to_canonical(), &fill_value);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Cannot fill_null with a null value")
        );
    }

    #[test]
    fn test_fill_null_decimal_i32() {
        let array = DecimalArray::from_option_iter(
            [Some(100i32), None, Some(300i32), None, Some(500i32)],
            DecimalDType::new(10, 2),
        );
        let fill_value = Scalar::decimal(
            DecimalValue::I32(999i32),
            DecimalDType::new(10, 2),
            Nullability::NonNullable,
        );

        let result = fill_null_canonical_array(array.to_canonical(), &fill_value).unwrap();

        assert_eq!(result.len(), 5);
        // Check that the values are filled correctly
        let result_decimal = result.to_decimal();
        assert!(result.is_valid(0));
        assert!(result.is_valid(1)); // was null, now filled
        assert!(result.is_valid(2));
        assert!(result.is_valid(3)); // was null, now filled
        assert!(result.is_valid(4));

        // Verify specific values
        assert_eq!(result_decimal.buffer::<i32>()[0], 100i32);
        assert_eq!(result_decimal.buffer::<i32>()[1], 999i32);
        assert_eq!(result_decimal.buffer::<i32>()[2], 300i32);
        assert_eq!(result_decimal.buffer::<i32>()[3], 999i32);
        assert_eq!(result_decimal.buffer::<i32>()[4], 500i32);
    }

    #[test]
    fn test_fill_null_decimal_i64() {
        let array = DecimalArray::from_option_iter(
            [Some(1000i64), None, Some(3000i64)],
            DecimalDType::new(15, 3),
        );
        let fill_value = Scalar::decimal(
            DecimalValue::I64(9999i64),
            DecimalDType::new(15, 3),
            Nullability::NonNullable,
        );

        let result = fill_null_canonical_array(array.to_canonical(), &fill_value).unwrap();

        assert_eq!(result.len(), 3);
        let result_decimal = result.to_decimal();
        assert!(result.is_valid(0));
        assert!(result.is_valid(1));
        assert!(result.is_valid(2));

        assert_eq!(result_decimal.buffer::<i64>()[0], 1000i64);
        assert_eq!(result_decimal.buffer::<i64>()[1], 9999i64);
        assert_eq!(result_decimal.buffer::<i64>()[2], 3000i64);
    }

    #[test]
    fn test_fill_null_decimal_i128() {
        let array = DecimalArray::from_option_iter(
            [Some(10000i128), None, Some(30000i128), None],
            DecimalDType::new(20, 4),
        );
        let fill_value = Scalar::decimal(
            DecimalValue::I128(99999i128),
            DecimalDType::new(20, 4),
            Nullability::NonNullable,
        );

        let result = fill_null_canonical_array(array.to_canonical(), &fill_value).unwrap();

        assert_eq!(result.len(), 4);
        let result_decimal = result.to_decimal();
        assert!(result.is_valid(0));
        assert!(result.is_valid(1));
        assert!(result.is_valid(2));
        assert!(result.is_valid(3));

        assert_eq!(result_decimal.buffer::<i128>()[0], 10000i128);
        assert_eq!(result_decimal.buffer::<i128>()[1], 99999i128);
        assert_eq!(result_decimal.buffer::<i128>()[2], 30000i128);
        assert_eq!(result_decimal.buffer::<i128>()[3], 99999i128);
    }

    #[test]
    fn test_fill_null_decimal_all_invalid() {
        let array =
            DecimalArray::from_option_iter([None::<i64>, None, None], DecimalDType::new(10, 2));
        let fill_value = Scalar::decimal(
            DecimalValue::I64(777i64),
            DecimalDType::new(10, 2),
            Nullability::NonNullable,
        );

        let result = fill_null_canonical_array(array.to_canonical(), &fill_value).unwrap();

        assert_eq!(result.len(), 3);
        // All values should be valid now with the fill value
        assert!(result.is_valid(0));
        assert!(result.is_valid(1));
        assert!(result.is_valid(2));
    }

    #[test]
    fn test_fill_null_decimal_no_nulls() {
        let array = DecimalArray::from_option_iter(
            [Some(100i32), Some(200i32), Some(300i32)],
            DecimalDType::new(10, 2),
        );
        let fill_value = Scalar::decimal(
            DecimalValue::I32(999i32),
            DecimalDType::new(10, 2),
            Nullability::NonNullable,
        );

        let result = fill_null_canonical_array(array.to_canonical(), &fill_value).unwrap();

        assert_eq!(result.len(), 3);
        let result_decimal = result.to_decimal();
        // All should remain as original values
        assert_eq!(result_decimal.buffer::<i32>()[0], 100i32);
        assert_eq!(result_decimal.buffer::<i32>()[1], 200i32);
        assert_eq!(result_decimal.buffer::<i32>()[2], 300i32);
    }
}
