// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use num_traits::ToBytes;
use vortex_buffer::{BufferString, ByteBuffer};
use vortex_dtype::DType;
use vortex_dtype::half::f16;
use vortex_error::{VortexError, vortex_err};
use vortex_proto::scalar as pb;
use vortex_proto::scalar::ListValue;
use vortex_proto::scalar::scalar_value::Kind;

use crate::pvalue::PValue;
use crate::{DecimalValue, InnerScalarValue, Scalar, ScalarValue};

impl From<&Scalar> for pb::Scalar {
    fn from(value: &Scalar) -> Self {
        pb::Scalar {
            dtype: Some((value.dtype()).into()),
            value: Some((value.value()).into()),
        }
    }
}

impl From<&ScalarValue> for pb::ScalarValue {
    fn from(value: &ScalarValue) -> Self {
        match value {
            ScalarValue(InnerScalarValue::Null) => pb::ScalarValue {
                kind: Some(Kind::NullValue(0)),
            },
            ScalarValue(InnerScalarValue::Bool(v)) => pb::ScalarValue {
                kind: Some(Kind::BoolValue(*v)),
            },
            ScalarValue(InnerScalarValue::Primitive(v)) => v.into(),
            ScalarValue(InnerScalarValue::Decimal(v)) => {
                let inner_value = match v {
                    DecimalValue::I8(v) => v.to_le_bytes().to_vec(),
                    DecimalValue::I16(v) => v.to_le_bytes().to_vec(),
                    DecimalValue::I32(v) => v.to_le_bytes().to_vec(),
                    DecimalValue::I64(v) => v.to_le_bytes().to_vec(),
                    DecimalValue::I128(v128) => v128.to_le_bytes().to_vec(),
                    DecimalValue::I256(v256) => v256.to_le_bytes().to_vec(),
                };

                pb::ScalarValue {
                    kind: Some(Kind::BytesValue(inner_value)),
                }
            }
            ScalarValue(InnerScalarValue::Buffer(v)) => pb::ScalarValue {
                kind: Some(Kind::BytesValue(v.as_slice().to_vec())),
            },
            ScalarValue(InnerScalarValue::BufferString(v)) => pb::ScalarValue {
                kind: Some(Kind::StringValue(v.as_str().to_string())),
            },
            ScalarValue(InnerScalarValue::List(v)) => {
                let mut values = Vec::with_capacity(v.len());
                for elem in v.iter() {
                    values.push(pb::ScalarValue::from(elem));
                }
                pb::ScalarValue {
                    kind: Some(Kind::ListValue(ListValue { values })),
                }
            }
        }
    }
}

impl From<&PValue> for pb::ScalarValue {
    fn from(value: &PValue) -> Self {
        match value {
            PValue::I8(v) => pb::ScalarValue {
                kind: Some(Kind::Int64Value(*v as i64)),
            },
            PValue::I16(v) => pb::ScalarValue {
                kind: Some(Kind::Int64Value(*v as i64)),
            },
            PValue::I32(v) => pb::ScalarValue {
                kind: Some(Kind::Int64Value(*v as i64)),
            },
            PValue::I64(v) => pb::ScalarValue {
                kind: Some(Kind::Int64Value(*v)),
            },
            PValue::U8(v) => pb::ScalarValue {
                kind: Some(Kind::Uint64Value(*v as u64)),
            },
            PValue::U16(v) => pb::ScalarValue {
                kind: Some(Kind::Uint64Value(*v as u64)),
            },
            PValue::U32(v) => pb::ScalarValue {
                kind: Some(Kind::Uint64Value(*v as u64)),
            },
            PValue::U64(v) => pb::ScalarValue {
                kind: Some(Kind::Uint64Value(*v)),
            },
            PValue::F16(v) => pb::ScalarValue {
                kind: Some(Kind::F16Value(v.to_bits() as u64)),
            },
            PValue::F32(v) => pb::ScalarValue {
                kind: Some(Kind::F32Value(*v)),
            },
            PValue::F64(v) => pb::ScalarValue {
                kind: Some(Kind::F64Value(*v)),
            },
        }
    }
}

impl TryFrom<&pb::Scalar> for Scalar {
    type Error = VortexError;

    fn try_from(value: &pb::Scalar) -> Result<Self, Self::Error> {
        let dtype = DType::try_from(
            value
                .dtype
                .as_ref()
                .ok_or_else(|| vortex_err!(InvalidSerde: "Scalar missing dtype"))?,
        )?;

        let value = ScalarValue::try_from(
            value
                .value
                .as_ref()
                .ok_or_else(|| vortex_err!(InvalidSerde: "Scalar missing value"))?,
        )?;

        Ok(Scalar::new(dtype, value))
    }
}

impl TryFrom<&pb::ScalarValue> for ScalarValue {
    type Error = VortexError;

    fn try_from(value: &pb::ScalarValue) -> Result<Self, Self::Error> {
        let kind = value
            .kind
            .as_ref()
            .ok_or_else(|| vortex_err!(InvalidSerde: "ScalarValue missing kind"))?;

        match kind {
            Kind::NullValue(_) => Ok(ScalarValue(InnerScalarValue::Null)),
            Kind::BoolValue(v) => Ok(ScalarValue(InnerScalarValue::Bool(*v))),
            Kind::Int64Value(v) => Ok(ScalarValue(InnerScalarValue::Primitive(PValue::I64(*v)))),
            Kind::Uint64Value(v) => Ok(ScalarValue(InnerScalarValue::Primitive(PValue::U64(*v)))),
            Kind::F16Value(v) => Ok(ScalarValue(InnerScalarValue::Primitive(PValue::F16(
                f16::from_bits(u16::try_from(*v).map_err(|_| {
                    vortex_err!("f16 bitwise representation has more than 16 bits: {}", v)
                })?),
            )))),
            Kind::F32Value(v) => Ok(ScalarValue(InnerScalarValue::Primitive(PValue::F32(*v)))),
            Kind::F64Value(v) => Ok(ScalarValue(InnerScalarValue::Primitive(PValue::F64(*v)))),
            Kind::StringValue(v) => Ok(ScalarValue(InnerScalarValue::BufferString(Arc::new(
                BufferString::from(v.clone()),
            )))),
            Kind::BytesValue(v) => Ok(ScalarValue(InnerScalarValue::Buffer(Arc::new(
                ByteBuffer::from(v.clone()),
            )))),
            Kind::ListValue(v) => {
                let mut values = Vec::with_capacity(v.values.len());
                for elem in v.values.iter() {
                    values.push(elem.try_into()?);
                }
                Ok(ScalarValue(InnerScalarValue::List(values.into())))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_buffer::BufferString;
    use vortex_dtype::half::f16;
    use vortex_dtype::{DType, DecimalDType, FieldDType, Nullability, PType, StructFields};
    use vortex_error::vortex_panic;
    use vortex_proto::scalar as pb;

    use super::*;
    use crate::{InnerScalarValue, Scalar, ScalarValue, i256};

    fn round_trip(scalar: Scalar) {
        assert_eq!(
            scalar,
            Scalar::try_from(&pb::Scalar::from(&scalar)).unwrap(),
        );
    }

    #[test]
    fn test_null() {
        round_trip(Scalar::null(DType::Null));
    }

    #[test]
    fn test_bool() {
        round_trip(Scalar::new(
            DType::Bool(Nullability::Nullable),
            ScalarValue(InnerScalarValue::Bool(true)),
        ));
    }

    #[test]
    fn test_primitive() {
        round_trip(Scalar::new(
            DType::Primitive(PType::I32, Nullability::Nullable),
            ScalarValue(InnerScalarValue::Primitive(42i32.into())),
        ));
    }

    #[test]
    fn test_buffer() {
        round_trip(Scalar::new(
            DType::Binary(Nullability::Nullable),
            ScalarValue(InnerScalarValue::Buffer(Arc::new(vec![1, 2, 3].into()))),
        ));
    }

    #[test]
    fn test_buffer_string() {
        round_trip(Scalar::new(
            DType::Utf8(Nullability::Nullable),
            ScalarValue(InnerScalarValue::BufferString(Arc::new(
                BufferString::from("hello".to_string()),
            ))),
        ));
    }

    #[test]
    fn test_list() {
        round_trip(Scalar::new(
            DType::List(
                Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
                Nullability::Nullable,
            ),
            ScalarValue(InnerScalarValue::List(
                vec![
                    ScalarValue(InnerScalarValue::Primitive(42i32.into())),
                    ScalarValue(InnerScalarValue::Primitive(43i32.into())),
                ]
                .into(),
            )),
        ));
    }

    #[test]
    fn test_f16() {
        round_trip(Scalar::primitive(
            f16::from_f32(0.42),
            Nullability::Nullable,
        ));
    }

    #[test]
    fn test_i8() {
        round_trip(Scalar::new(
            DType::Primitive(PType::I8, Nullability::Nullable),
            ScalarValue(InnerScalarValue::Primitive(i8::MIN.into())),
        ));

        round_trip(Scalar::new(
            DType::Primitive(PType::I8, Nullability::Nullable),
            ScalarValue(InnerScalarValue::Primitive(0i8.into())),
        ));

        round_trip(Scalar::new(
            DType::Primitive(PType::I8, Nullability::Nullable),
            ScalarValue(InnerScalarValue::Primitive(i8::MAX.into())),
        ));
    }

    #[rstest]
    #[case(Scalar::binary(ByteBuffer::copy_from(b"hello"), Nullability::NonNullable))]
    #[case(Scalar::utf8("hello", Nullability::NonNullable))]
    #[case(Scalar::primitive(1u8, Nullability::NonNullable))]
    #[case(Scalar::primitive(
        f32::from_bits(u32::from_le_bytes([0xFFu8, 0x8A, 0xF9, 0xFF])),
        Nullability::NonNullable
    ))]
    #[case(Scalar::list(Arc::new(PType::U8.into()), vec![Scalar::primitive(1u8, Nullability::NonNullable)], Nullability::NonNullable
    ))]
    #[case(Scalar::struct_(DType::Struct(
        StructFields::from_iter([
            ("a", FieldDType::from(DType::Primitive(PType::U32, Nullability::NonNullable))),
            ("b", FieldDType::from(DType::Primitive(PType::F16, Nullability::NonNullable))),
        ]),
        Nullability::NonNullable),
        vec![
            Scalar::primitive(23592960u32, Nullability::NonNullable),
            Scalar::primitive(f16::from_f32(2.6584664e36f32), Nullability::NonNullable),
        ],
    ))]
    #[case(Scalar::struct_(DType::Struct(
        StructFields::from_iter([
            ("a", FieldDType::from(DType::Primitive(PType::U64, Nullability::NonNullable))),
            ("b", FieldDType::from(DType::Primitive(PType::F32, Nullability::NonNullable))),
            ("c", FieldDType::from(DType::Primitive(PType::F16, Nullability::NonNullable))),
        ]),
        Nullability::NonNullable),
        vec![
            Scalar::primitive(415118687234u64, Nullability::NonNullable),
            Scalar::primitive(2.6584664e36f32, Nullability::NonNullable),
            Scalar::primitive(f16::from_f32(2.6584664e36f32), Nullability::NonNullable),
        ],
    ))]
    #[case(Scalar::decimal(
        DecimalValue::I256(i256::from_i128(12345643673471)),
        DecimalDType::new(10, 2),
        Nullability::NonNullable
    ))]
    #[case(Scalar::decimal(
        DecimalValue::I16(23412),
        DecimalDType::new(3, 2),
        Nullability::NonNullable
    ))]
    fn test_scalar_value_serde_roundtrip(#[case] scalar: Scalar) {
        let written = scalar.value().to_protobytes::<Vec<u8>>();
        let scalar_read_back = ScalarValue::from_protobytes(&written).unwrap();
        assert_eq!(
            Scalar::new(scalar.dtype().clone(), scalar_read_back),
            scalar
        );
    }

    #[test]
    fn test_backcompat_f16_serialized_as_u64() {
        // Note that this is a backwards compatibility test for poor design in the previous implementation.
        // Previously, f16 ScalarValues were serialized as `pb::ScalarValue::Uint64Value(v.to_bits() as u64)`.
        let pb_scalar_value = pb::ScalarValue {
            kind: Some(Kind::Uint64Value(f16::from_f32(0.42).to_bits() as u64)),
        };
        let scalar_value = ScalarValue::try_from(&pb_scalar_value).unwrap();
        assert_eq!(
            scalar_value.as_pvalue().unwrap(),
            Some(PValue::U64(14008u64))
        );

        let scalar = Scalar::new(
            DType::Primitive(PType::F16, Nullability::Nullable),
            scalar_value,
        );

        assert_eq!(
            scalar.as_primitive().pvalue().unwrap(),
            PValue::F16(f16::from_f32(0.42))
        );
    }

    #[test]
    fn test_scalar_value_direct_roundtrip_f16() {
        // Test that ScalarValue with f16 roundtrips correctly without going through Scalar
        let f16_values = vec![
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(-1.0),
            f16::from_f32(0.42),
            f16::from_f32(5.722046e-6),
            f16::from_f32(std::f32::consts::PI),
            f16::INFINITY,
            f16::NEG_INFINITY,
            f16::NAN,
        ];

        for f16_val in f16_values {
            let scalar_value = ScalarValue(InnerScalarValue::Primitive(PValue::F16(f16_val)));
            let written = scalar_value.to_protobytes::<Vec<u8>>();
            let read_back = ScalarValue::from_protobytes(&written).unwrap();

            match (&scalar_value.0, &read_back.0) {
                (
                    InnerScalarValue::Primitive(PValue::F16(original)),
                    InnerScalarValue::Primitive(PValue::F16(roundtripped)),
                ) => {
                    if original.is_nan() && roundtripped.is_nan() {
                        // NaN values are equal for our purposes
                        continue;
                    }
                    assert_eq!(
                        original, roundtripped,
                        "F16 value {original:?} did not roundtrip correctly"
                    );
                }
                _ => {
                    vortex_panic!(
                        "Expected f16 primitive values, got {scalar_value:?} and {read_back:?}"
                    )
                }
            }
        }
    }

    #[test]
    fn test_scalar_value_direct_roundtrip_preserves_values() {
        // Test that ScalarValue roundtripping preserves values (but not necessarily exact types)
        // Note: Proto encoding consolidates integer types (u8/u16/u32 → u64, i8/i16/i32 → i64)

        // Test cases that should roundtrip exactly
        let exact_roundtrip_cases = vec![
            ("null", ScalarValue(InnerScalarValue::Null)),
            ("bool_true", ScalarValue(InnerScalarValue::Bool(true))),
            ("bool_false", ScalarValue(InnerScalarValue::Bool(false))),
            (
                "u64",
                ScalarValue(InnerScalarValue::Primitive(PValue::U64(
                    18446744073709551615,
                ))),
            ),
            (
                "i64",
                ScalarValue(InnerScalarValue::Primitive(PValue::I64(
                    -9223372036854775808,
                ))),
            ),
            (
                "f32",
                ScalarValue(InnerScalarValue::Primitive(PValue::F32(
                    std::f32::consts::E,
                ))),
            ),
            (
                "f64",
                ScalarValue(InnerScalarValue::Primitive(PValue::F64(
                    std::f64::consts::PI,
                ))),
            ),
            (
                "string",
                ScalarValue(InnerScalarValue::BufferString(Arc::new(
                    BufferString::from("test"),
                ))),
            ),
            (
                "bytes",
                ScalarValue(InnerScalarValue::Buffer(Arc::new(
                    vec![1, 2, 3, 4, 5].into(),
                ))),
            ),
        ];

        for (name, value) in exact_roundtrip_cases {
            let written = value.to_protobytes::<Vec<u8>>();
            let read_back = ScalarValue::from_protobytes(&written).unwrap();

            let original_debug = format!("{value:?}");
            let roundtrip_debug = format!("{read_back:?}");
            assert_eq!(
                original_debug, roundtrip_debug,
                "ScalarValue {name} did not roundtrip exactly"
            );
        }

        // Test cases where type changes but value is preserved
        // Unsigned integers consolidate to U64
        let unsigned_cases = vec![
            (
                "u8",
                ScalarValue(InnerScalarValue::Primitive(PValue::U8(255))),
                255u64,
            ),
            (
                "u16",
                ScalarValue(InnerScalarValue::Primitive(PValue::U16(65535))),
                65535u64,
            ),
            (
                "u32",
                ScalarValue(InnerScalarValue::Primitive(PValue::U32(4294967295))),
                4294967295u64,
            ),
        ];

        for (name, value, expected) in unsigned_cases {
            let written = value.to_protobytes::<Vec<u8>>();
            let read_back = ScalarValue::from_protobytes(&written).unwrap();

            match &read_back.0 {
                InnerScalarValue::Primitive(PValue::U64(v)) => {
                    assert_eq!(
                        *v, expected,
                        "ScalarValue {name} value not preserved: expected {expected}, got {v}"
                    );
                }
                _ => vortex_panic!("Unexpected type after roundtrip for {name}: {read_back:?}"),
            }
        }

        // Signed integers consolidate to I64
        let signed_cases = vec![
            (
                "i8",
                ScalarValue(InnerScalarValue::Primitive(PValue::I8(-128))),
                -128i64,
            ),
            (
                "i16",
                ScalarValue(InnerScalarValue::Primitive(PValue::I16(-32768))),
                -32768i64,
            ),
            (
                "i32",
                ScalarValue(InnerScalarValue::Primitive(PValue::I32(-2147483648))),
                -2147483648i64,
            ),
        ];

        for (name, value, expected) in signed_cases {
            let written = value.to_protobytes::<Vec<u8>>();
            let read_back = ScalarValue::from_protobytes(&written).unwrap();

            match &read_back.0 {
                InnerScalarValue::Primitive(PValue::I64(v)) => {
                    assert_eq!(
                        *v, expected,
                        "ScalarValue {name} value not preserved: expected {expected}, got {v}"
                    );
                }
                _ => vortex_panic!("Unexpected type after roundtrip for {name}: {read_back:?}"),
            }
        }
    }
}
