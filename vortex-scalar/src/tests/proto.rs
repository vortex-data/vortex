// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_dtype::DecimalDType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_dtype::half::f16;
use vortex_error::vortex_panic;
use vortex_proto::scalar as pb;
use vortex_proto::scalar::scalar_value::Kind;
use vortex_session::VortexSession;

use crate::DecimalValue;
use crate::PValue;
use crate::Scalar;
use crate::ScalarValue;
use crate::tests::SESSION;

fn session() -> VortexSession {
    VortexSession::empty()
}

fn round_trip(scalar: Scalar) {
    assert_eq!(
        scalar,
        Scalar::from_proto(&pb::Scalar::from(&scalar), &session()).unwrap(),
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
        Some(ScalarValue::Bool(true)),
    ));
}

#[test]
fn test_primitive() {
    round_trip(Scalar::new(
        DType::Primitive(PType::I32, Nullability::Nullable),
        Some(ScalarValue::Primitive(42i32.into())),
    ));
}

#[test]
fn test_buffer() {
    round_trip(Scalar::new(
        DType::Binary(Nullability::Nullable),
        Some(ScalarValue::Binary(vec![1, 2, 3].into())),
    ));
}

#[test]
fn test_buffer_string() {
    round_trip(Scalar::new(
        DType::Utf8(Nullability::Nullable),
        Some(ScalarValue::Utf8(BufferString::from("hello".to_string()))),
    ));
}

#[test]
fn test_list() {
    round_trip(Scalar::new(
        DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            Nullability::Nullable,
        ),
        Some(ScalarValue::List(vec![
            Some(ScalarValue::Primitive(42i32.into())),
            Some(ScalarValue::Primitive(43i32.into())),
        ])),
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
        Some(ScalarValue::Primitive(i8::MIN.into())),
    ));

    round_trip(Scalar::new(
        DType::Primitive(PType::I8, Nullability::Nullable),
        Some(ScalarValue::Primitive(0i8.into())),
    ));

    round_trip(Scalar::new(
        DType::Primitive(PType::I8, Nullability::Nullable),
        Some(ScalarValue::Primitive(i8::MAX.into())),
    ));
}

#[test]
fn test_decimal_i32_roundtrip() {
    // A typical decimal with moderate precision and scale.
    round_trip(Scalar::decimal(
        DecimalValue::I32(123_456),
        DecimalDType::new(10, 2),
        Nullability::NonNullable,
    ));
}

#[test]
fn test_decimal_i128_roundtrip() {
    // A large decimal value that requires i128 storage.
    round_trip(Scalar::decimal(
        DecimalValue::I128(99_999_999_999_999_999_999),
        DecimalDType::new(38, 6),
        Nullability::Nullable,
    ));
}

#[test]
fn test_decimal_null_roundtrip() {
    round_trip(Scalar::null(DType::Decimal(
        DecimalDType::new(10, 2),
        Nullability::Nullable,
    )));
}

#[test]
fn test_scalar_value_serde_roundtrip_binary() {
    round_trip(Scalar::binary(
        ByteBuffer::copy_from(b"hello"),
        Nullability::NonNullable,
    ));
}

#[test]
fn test_scalar_value_serde_roundtrip_utf8() {
    round_trip(Scalar::utf8("hello", Nullability::NonNullable));
}

#[test]
fn test_backcompat_f16_serialized_as_u64() {
    // Backwards compatibility test for the legacy f16 serialization format.
    //
    // Previously, f16 ScalarValues were serialized as `Uint64Value(v.to_bits() as u64)` because
    // the proto schema only had 64-bit integer types, and f16's underlying representation is
    // u16 which got widened to u64.
    //
    // The current implementation uses a dedicated `F16Value` proto field, but we must still be
    // able to deserialize the old format. This test verifies that:
    //
    // 1. A `Uint64Value` containing f16 bits can be read as a U64 primitive (the raw bits).
    // 2. When wrapped in a Scalar with F16 dtype, the value is correctly interpreted as f16.
    //
    // This ensures data written with the old serialization format remains readable.

    // Simulate the old serialization: f16(0.42) stored as Uint64Value with its bit pattern.
    let f16_value = f16::from_f32(0.42);
    let f16_bits_as_u64 = f16_value.to_bits() as u64; // 14008

    let pb_scalar_value = pb::ScalarValue {
        kind: Some(Kind::Uint64Value(f16_bits_as_u64)),
    };

    // Step 1: Verify the normal U64 scalar.
    let scalar_value = ScalarValue::from_proto(
        &pb_scalar_value,
        &DType::Primitive(PType::U64, Nullability::NonNullable),
        &SESSION,
    )
    .unwrap();
    assert_eq!(
        scalar_value.as_ref().map(|v| v.as_primitive()),
        Some(&PValue::U64(14008u64)),
    );

    // Step 2: Verify that when we use F16 dtype, the Uint64Value is correctly interpreted.
    let scalar_value_f16 = ScalarValue::from_proto(
        &pb_scalar_value,
        &DType::Primitive(PType::F16, Nullability::Nullable),
        &SESSION,
    )
    .unwrap();

    let scalar = Scalar::new(
        DType::Primitive(PType::F16, Nullability::Nullable),
        scalar_value_f16,
    );

    assert_eq!(
        scalar.as_primitive().pvalue().unwrap(),
        PValue::F16(f16::from_f32(0.42)),
        "Uint64Value should be correctly interpreted as f16 when dtype is F16"
    );
}

#[test]
fn test_scalar_value_direct_roundtrip_f16() {
    // Test that ScalarValue with f16 roundtrips correctly without going through Scalar.
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
        let scalar_value = ScalarValue::Primitive(PValue::F16(f16_val));
        let pb_value = ScalarValue::to_proto(Some(&scalar_value));
        let read_back = ScalarValue::from_proto(
            &pb_value,
            &DType::Primitive(PType::F16, Nullability::NonNullable),
            &SESSION,
        )
        .unwrap();

        match (&scalar_value, read_back.as_ref()) {
            (
                ScalarValue::Primitive(PValue::F16(original)),
                Some(ScalarValue::Primitive(PValue::F16(roundtripped))),
            ) => {
                if original.is_nan() && roundtripped.is_nan() {
                    // NaN values are equal for our purposes.
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
    // Test that ScalarValue roundtripping preserves values (but not necessarily exact types).
    // Note: Proto encoding consolidates integer types (u8/u16/u32 → u64, i8/i16/i32 → i64).

    // Test cases that should roundtrip exactly.
    let exact_roundtrip_cases: Vec<(&str, Option<ScalarValue>, DType)> = vec![
        ("null", None, DType::Null),
        (
            "bool_true",
            Some(ScalarValue::Bool(true)),
            DType::Bool(Nullability::Nullable),
        ),
        (
            "bool_false",
            Some(ScalarValue::Bool(false)),
            DType::Bool(Nullability::Nullable),
        ),
        (
            "u64",
            Some(ScalarValue::Primitive(PValue::U64(18446744073709551615))),
            DType::Primitive(PType::U64, Nullability::Nullable),
        ),
        (
            "i64",
            Some(ScalarValue::Primitive(PValue::I64(-9223372036854775808))),
            DType::Primitive(PType::I64, Nullability::Nullable),
        ),
        (
            "f32",
            Some(ScalarValue::Primitive(PValue::F32(std::f32::consts::E))),
            DType::Primitive(PType::F32, Nullability::Nullable),
        ),
        (
            "f64",
            Some(ScalarValue::Primitive(PValue::F64(std::f64::consts::PI))),
            DType::Primitive(PType::F64, Nullability::Nullable),
        ),
        (
            "string",
            Some(ScalarValue::Utf8(BufferString::from("test"))),
            DType::Utf8(Nullability::Nullable),
        ),
        (
            "bytes",
            Some(ScalarValue::Binary(vec![1, 2, 3, 4, 5].into())),
            DType::Binary(Nullability::Nullable),
        ),
    ];

    for (name, value, dtype) in exact_roundtrip_cases {
        let pb_value = ScalarValue::to_proto(value.as_ref());
        let read_back = ScalarValue::from_proto(&pb_value, &dtype, &SESSION).unwrap();

        let original_debug = format!("{value:?}");
        let roundtrip_debug = format!("{read_back:?}");
        assert_eq!(
            original_debug, roundtrip_debug,
            "ScalarValue {name} did not roundtrip exactly"
        );
    }

    // Test cases where type changes but value is preserved.
    // Unsigned integers consolidate to U64.
    let unsigned_cases = vec![
        (
            "u8",
            ScalarValue::Primitive(PValue::U8(255)),
            DType::Primitive(PType::U8, Nullability::Nullable),
            255u64,
        ),
        (
            "u16",
            ScalarValue::Primitive(PValue::U16(65535)),
            DType::Primitive(PType::U16, Nullability::Nullable),
            65535u64,
        ),
        (
            "u32",
            ScalarValue::Primitive(PValue::U32(4294967295)),
            DType::Primitive(PType::U32, Nullability::Nullable),
            4294967295u64,
        ),
    ];

    for (name, value, dtype, expected) in unsigned_cases {
        let pb_value = ScalarValue::to_proto(Some(&value));
        let read_back = ScalarValue::from_proto(&pb_value, &dtype, &SESSION).unwrap();

        match read_back.as_ref() {
            Some(ScalarValue::Primitive(pv)) => {
                let v = match pv {
                    PValue::U8(v) => *v as u64,
                    PValue::U16(v) => *v as u64,
                    PValue::U32(v) => *v as u64,
                    PValue::U64(v) => *v,
                    _ => vortex_panic!("Unexpected primitive type for {name}: {pv:?}"),
                };
                assert_eq!(
                    v, expected,
                    "ScalarValue {name} value not preserved: expected {expected}, got {v}"
                );
            }
            _ => vortex_panic!("Unexpected type after roundtrip for {name}: {read_back:?}"),
        }
    }

    // Signed integers consolidate to I64.
    let signed_cases = vec![
        (
            "i8",
            ScalarValue::Primitive(PValue::I8(-128)),
            DType::Primitive(PType::I8, Nullability::Nullable),
            -128i64,
        ),
        (
            "i16",
            ScalarValue::Primitive(PValue::I16(-32768)),
            DType::Primitive(PType::I16, Nullability::Nullable),
            -32768i64,
        ),
        (
            "i32",
            ScalarValue::Primitive(PValue::I32(-2147483648)),
            DType::Primitive(PType::I32, Nullability::Nullable),
            -2147483648i64,
        ),
    ];

    for (name, value, dtype, expected) in signed_cases {
        let pb_value = ScalarValue::to_proto(Some(&value));
        let read_back = ScalarValue::from_proto(&pb_value, &dtype, &SESSION).unwrap();

        match read_back.as_ref() {
            Some(ScalarValue::Primitive(pv)) => {
                let v = match pv {
                    PValue::I8(v) => *v as i64,
                    PValue::I16(v) => *v as i64,
                    PValue::I32(v) => *v as i64,
                    PValue::I64(v) => *v,
                    _ => vortex_panic!("Unexpected primitive type for {name}: {pv:?}"),
                };
                assert_eq!(
                    v, expected,
                    "ScalarValue {name} value not preserved: expected {expected}, got {v}"
                );
            }
            _ => vortex_panic!("Unexpected type after roundtrip for {name}: {read_back:?}"),
        }
    }
}
