// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::ToCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_decimal_value_type;
use vortex_array::match_each_native_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

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
        Canonical::Bool(array) => fill_bool_array(array, fill_value, result_nullability),
        Canonical::Primitive(array) => fill_primitive_array(array, fill_value, result_nullability),
        Canonical::Decimal(array) => fill_decimal_array(array, fill_value, result_nullability),
        Canonical::VarBinView(array) => {
            fill_varbinview_array(array, fill_value, result_nullability)
        }
        Canonical::Struct(_)
        | Canonical::List(_)
        | Canonical::FixedSizeList(_)
        | Canonical::Extension(_) => canonical.into_array().fill_null(fill_value.clone())?,
        Canonical::Variant(_) => unreachable!("Variant arrays are not fuzzed"),
    })
}

fn fill_bool_array(
    array: BoolArray,
    fill_value: &Scalar,
    result_nullability: Nullability,
) -> ArrayRef {
    let fill_bool = fill_value
        .as_bool()
        .value()
        .vortex_expect("cannot have null fill value");

    match array
        .validity()
        .vortex_expect("bool validity should be derivable in fuzz baseline")
    {
        Validity::NonNullable | Validity::AllValid => {
            BoolArray::new(array.into_bit_buffer(), result_nullability.into()).into_array()
        }
        Validity::AllInvalid => ConstantArray::new(fill_value.clone(), array.len()).into_array(),
        Validity::Array(validity_array) => {
            let validity_bits = validity_array.to_bool().into_bit_buffer();
            let data_bits = array.into_bit_buffer();

            let new_bits = match data_bits.try_into_mut() {
                Ok(mut buf) => {
                    (!validity_bits)
                        .set_indices()
                        .for_each(|i| buf.set_to(i, fill_bool));
                    buf.freeze()
                }
                Err(data_bits) => {
                    if fill_bool {
                        data_bits | !validity_bits
                    } else {
                        data_bits & validity_bits
                    }
                }
            };

            BoolArray::new(new_bits, result_nullability.into()).into_array()
        }
    }
}

#[expect(clippy::cognitive_complexity)]
fn fill_primitive_array(
    array: PrimitiveArray,
    fill_value: &Scalar,
    result_nullability: Nullability,
) -> ArrayRef {
    match_each_native_ptype!(array.ptype(), |T| {
        let fill_val = T::try_from(fill_value)
            .vortex_expect("fill value conversion should succeed in fuzz test");

        match array
            .validity()
            .vortex_expect("primitive validity should be derivable in fuzz baseline")
        {
            Validity::NonNullable | Validity::AllValid => {
                PrimitiveArray::new(array.to_buffer::<T>(), result_nullability.into()).into_array()
            }
            Validity::AllInvalid => {
                ConstantArray::new(fill_value.clone(), array.len()).into_array()
            }
            Validity::Array(validity_array) => {
                let validity_bool_array = validity_array.to_bool();
                let validity_bits = validity_bool_array.to_bit_buffer();
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

fn fill_decimal_array(
    array: DecimalArray,
    fill_value: &Scalar,
    result_nullability: Nullability,
) -> ArrayRef {
    let decimal_dtype = array.decimal_dtype();
    let decimal_scalar = fill_value.as_decimal();

    match_each_decimal_value_type!(array.values_type(), |D| {
        let fill_val = D::try_from(decimal_scalar)
            .vortex_expect("decimal fill value conversion should succeed in fuzz test");

        match array
            .validity()
            .vortex_expect("decimal validity should be derivable in fuzz baseline")
        {
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
                let validity_bits = validity_bool_array.to_bit_buffer();
                let data_buffer = array.buffer::<D>();

                let mut new_data = BufferMut::with_capacity(array.len());
                for i in 0..array.len() {
                    if validity_bits.value(i) {
                        new_data.push(data_buffer[i]);
                    } else {
                        new_data.push(fill_val);
                    }
                }

                DecimalArray::try_new(new_data.freeze(), decimal_dtype, result_nullability.into())
                    .vortex_expect("DecimalArray creation should succeed in fuzz test")
                    .into_array()
            }
        }
    })
}

fn fill_varbinview_array(
    array: VarBinViewArray,
    fill_value: &Scalar,
    result_nullability: Nullability,
) -> ArrayRef {
    let array_ref = array.clone().into_array();
    match array
        .validity()
        .vortex_expect("varbinview validity should be derivable in fuzz baseline")
    {
        Validity::NonNullable | Validity::AllValid => array.into_array(),
        Validity::AllInvalid => ConstantArray::new(fill_value.clone(), array.len()).into_array(),
        Validity::Array(validity_array) => {
            let validity_bool_array = validity_array.to_bool();
            let validity_bits = validity_bool_array.to_bit_buffer();

            match array.dtype() {
                DType::Utf8(_) => {
                    let fill_str = fill_value
                        .as_utf8()
                        .value()
                        .vortex_expect("cannot have null fill value");
                    let strings: Vec<String> = (0..array.len())
                        .map(|i| {
                            if validity_bits.value(i) {
                                array_ref
                                    .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                                    .vortex_expect("scalar_at")
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
                        VarBinViewArray::new_handle(
                            result.to_varbinview().views_handle().clone(),
                            Arc::clone(result.to_varbinview().data_buffers()),
                            result.dtype().as_nullable(),
                            result_nullability.into(),
                        )
                        .into_array()
                    } else {
                        result
                    }
                }
                DType::Binary(_) => {
                    let fill_bytes = fill_value
                        .as_binary()
                        .value()
                        .vortex_expect("cannot have null fill value");
                    let binaries: Vec<Vec<u8>> = (0..array.len())
                        .map(|i| {
                            if validity_bits.value(i) {
                                array_ref
                                    .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
                                    .vortex_expect("scalar_at")
                                    .as_binary()
                                    .value()
                                    .vortex_expect("cannot have null valid value")
                                    .to_vec()
                            } else {
                                fill_bytes.to_vec()
                            }
                        })
                        .collect();
                    let binary_refs: Vec<&[u8]> = binaries.iter().map(|b| b.as_slice()).collect();
                    let result = VarBinViewArray::from_iter_bin(binary_refs).into_array();
                    if result_nullability == Nullability::Nullable {
                        VarBinViewArray::new_handle(
                            result.to_varbinview().views_handle().clone(),
                            Arc::clone(result.to_varbinview().data_buffers()),
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
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::DecimalValue;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;

    use super::fill_null_canonical_array;

    fn canonical(array: impl IntoArray) -> Canonical {
        array
            .into_array()
            .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    }

    #[test]
    fn test_fill_null_primitive() {
        let array = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5)]);
        let fill_value = Scalar::from(42i32);

        let result = fill_null_canonical_array(canonical(array), &fill_value).unwrap();

        let expected = PrimitiveArray::from_iter([1i32, 42, 3, 42, 5]);
        assert_arrays_eq!(expected, result);
    }

    #[test]
    fn test_fill_null_bool() {
        let data_buffer = BitBuffer::from(vec![true, false, false, false]);
        let validity_buffer = BitBuffer::from(vec![true, false, true, false]);
        let array = BoolArray::new(data_buffer, Validity::from(validity_buffer));
        let fill_value = Scalar::from(true);

        let result = fill_null_canonical_array(canonical(array), &fill_value).unwrap();

        let expected = BoolArray::from(BitBuffer::from(vec![true, true, false, true]));
        assert_arrays_eq!(expected, result);
    }

    #[test]
    fn test_fill_null_string() {
        let array = VarBinViewArray::from_iter(
            [Some("hello"), None, Some("world")].iter().copied(),
            DType::Utf8(Nullability::Nullable),
        );
        let fill_value = Scalar::utf8("default", Nullability::NonNullable);

        let result = fill_null_canonical_array(canonical(array), &fill_value).unwrap();

        let expected = VarBinViewArray::from_iter_str(["hello", "default", "world"]);
        assert_arrays_eq!(expected, result);
    }

    #[test]
    fn test_fill_null_all_invalid() {
        let array = PrimitiveArray::from_option_iter([None::<i32>, None, None]);
        let fill_value = Scalar::from(100i32);

        let result = fill_null_canonical_array(canonical(array), &fill_value).unwrap();

        let expected = PrimitiveArray::from_iter([100i32, 100, 100]);
        assert_arrays_eq!(expected, result);
    }

    #[test]
    fn test_fill_null_no_nulls() {
        let array = PrimitiveArray::from_iter([1i32, 2, 3]);
        let fill_value = Scalar::from(42i32);

        let result = fill_null_canonical_array(canonical(array), &fill_value).unwrap();

        let expected = PrimitiveArray::from_iter([1i32, 2, 3]);
        assert_arrays_eq!(expected, result);
    }

    #[test]
    #[should_panic]
    fn test_fill_null_with_null_value_errors() {
        let array = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]);
        let fill_value = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));

        let result = fill_null_canonical_array(canonical(array), &fill_value);

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

        let result = fill_null_canonical_array(canonical(array), &fill_value).unwrap();

        let expected = DecimalArray::from_iter(
            [100i32, 999i32, 300i32, 999i32, 500i32],
            DecimalDType::new(10, 2),
        );
        assert_arrays_eq!(expected, result);
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

        let result = fill_null_canonical_array(canonical(array), &fill_value).unwrap();

        let expected =
            DecimalArray::from_iter([1000i64, 9999i64, 3000i64], DecimalDType::new(15, 3));
        assert_arrays_eq!(expected, result);
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

        let result = fill_null_canonical_array(canonical(array), &fill_value).unwrap();

        let expected = DecimalArray::from_iter(
            [10000i128, 99999i128, 30000i128, 99999i128],
            DecimalDType::new(20, 4),
        );
        assert_arrays_eq!(expected, result);
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

        let result = fill_null_canonical_array(canonical(array), &fill_value).unwrap();

        let expected = DecimalArray::from_option_iter(
            [Some(777i64), Some(777i64), Some(777i64)],
            DecimalDType::new(10, 2),
        )
        .into_array()
        .cast(result.dtype().clone())
        .unwrap();
        assert_arrays_eq!(expected, result);
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

        let result = fill_null_canonical_array(canonical(array), &fill_value).unwrap();

        let expected = DecimalArray::from_option_iter(
            [Some(100i32), Some(200i32), Some(300i32)],
            DecimalDType::new(10, 2),
        )
        .into_array()
        .cast(result.dtype().clone())
        .unwrap();
        assert_arrays_eq!(expected, result);
    }
}
