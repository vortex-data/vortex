// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! JNI bindings for Vortex expressions.
//!
//! Expressions are built on the native side — Java holds opaque pointers and combines them
//! through these JNI entry points. Each `new*` call returns a pointer that must be freed
//! with [`Java_dev_vortex_jni_NativeExpression_free`]. Builders do not take ownership of
//! their inputs so Java remains responsible for freeing child expressions.

use std::sync::Arc;

use jni::EnvUnowned;
use jni::objects::JByteArray;
use jni::objects::JClass;
use jni::objects::JLongArray;
use jni::objects::JObjectArray;
use jni::objects::JString;
use jni::objects::ReleaseMode;
use jni::sys::jboolean;
use jni::sys::jbyte;
use jni::sys::jdouble;
use jni::sys::jfloat;
use jni::sys::jint;
use jni::sys::jlong;
use jni::sys::jshort;
use vortex::dtype::BigCast;
use vortex::dtype::DType;
use vortex::dtype::DecimalDType;
use vortex::dtype::FieldName;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::and_collect;
use vortex::expr::between;
use vortex::expr::get_item;
use vortex::expr::is_not_null;
use vortex::expr::is_null;
use vortex::expr::lit;
use vortex::expr::not;
use vortex::expr::or_collect;
use vortex::expr::root;
use vortex::expr::select;
use vortex::extension::datetime::Date;
use vortex::extension::datetime::TimeUnit;
use vortex::extension::datetime::Timestamp;
use vortex::scalar::DecimalValue;
use vortex::scalar::Scalar;
use vortex::scalar::ScalarValue;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::between::BetweenOptions;
use vortex::scalar_fn::fns::between::StrictComparison;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::like::Like;
use vortex::scalar_fn::fns::like::LikeOptions;
use vortex::scalar_fn::fns::operators::Operator;

use crate::errors::JNIError;
use crate::errors::try_or_throw;

fn into_raw(expr: Expression) -> jlong {
    Box::into_raw(Box::new(expr)) as jlong
}

/// SAFETY: pointer must originate from [`into_raw`] and not yet be freed.
unsafe fn expr_ref<'a>(ptr: jlong) -> &'a Expression {
    debug_assert!(ptr != 0, "null expression pointer");
    unsafe { &*(ptr as *const Expression) }
}

fn parse_op(op: jbyte) -> Result<Operator, JNIError> {
    Ok(match op {
        0 => Operator::Eq,
        1 => Operator::NotEq,
        2 => Operator::Gt,
        3 => Operator::Gte,
        4 => Operator::Lt,
        5 => Operator::Lte,
        6 => Operator::And,
        7 => Operator::Or,
        8 => Operator::Add,
        9 => Operator::Sub,
        10 => Operator::Mul,
        11 => Operator::Div,
        other => throw_runtime!("unknown binary operator code: {other}"),
    })
}

/// Parse a Vortex [`TimeUnit`] from the wire-encoded byte tag.
fn parse_time_unit(tag: jbyte) -> Result<TimeUnit, JNIError> {
    TimeUnit::try_from(tag as u8).map_err(JNIError::from)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_free(
    _env: EnvUnowned,
    _class: JClass,
    pointer: jlong,
) {
    if pointer == 0 {
        return;
    }
    // SAFETY: pointer was created via `into_raw` above.
    drop(unsafe { Box::from_raw(pointer as *mut Expression) });
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_root(
    _env: EnvUnowned,
    _class: JClass,
) -> jlong {
    into_raw(root())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_getItem(
    mut env: EnvUnowned,
    _class: JClass,
    name: JString,
    child: jlong,
) -> jlong {
    try_or_throw(&mut env, |env| {
        let field: String = name.try_to_string(env)?;
        let field: FieldName = Arc::<str>::from(field.as_str()).into();
        let child = unsafe { expr_ref(child) }.clone();
        Ok(into_raw(get_item(field, child)))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_select(
    mut env: EnvUnowned,
    _class: JClass,
    field_names: JObjectArray,
    child: jlong,
) -> jlong {
    try_or_throw(&mut env, |env| {
        let count = field_names.len(env)?;
        let mut fields: Vec<FieldName> = Vec::with_capacity(count);
        for idx in 0..count {
            let obj = field_names.get_element(env, idx)?;
            let s = env.cast_local::<JString>(obj)?;
            let name: String = s.try_to_string(env)?;
            fields.push(Arc::<str>::from(name.as_str()).into());
        }
        let child = unsafe { expr_ref(child) }.clone();
        Ok(into_raw(select(fields, child)))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_and(
    mut env: EnvUnowned,
    _class: JClass,
    operands: JLongArray,
) -> jlong {
    try_or_throw(&mut env, |env| {
        let exprs = collect_operands(env, &operands)?;
        and_collect(exprs)
            .map(into_raw)
            .ok_or_else(|| vortex_err!("empty AND expression").into())
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_or(
    mut env: EnvUnowned,
    _class: JClass,
    operands: JLongArray,
) -> jlong {
    try_or_throw(&mut env, |env| {
        let exprs = collect_operands(env, &operands)?;
        or_collect(exprs)
            .map(into_raw)
            .ok_or_else(|| vortex_err!("empty OR expression").into())
    })
}

fn collect_operands(
    env: &mut jni::Env,
    operands: &JLongArray,
) -> Result<Vec<Expression>, JNIError> {
    let ptrs = unsafe { operands.get_elements(env, ReleaseMode::NoCopyBack) }?;
    Ok(ptrs
        .iter()
        .map(|ptr| unsafe { expr_ref(*ptr) }.clone())
        .collect())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_binary(
    mut env: EnvUnowned,
    _class: JClass,
    op: jbyte,
    lhs: jlong,
    rhs: jlong,
) -> jlong {
    try_or_throw(&mut env, |_| {
        let operator = parse_op(op)?;
        let lhs = unsafe { expr_ref(lhs) }.clone();
        let rhs = unsafe { expr_ref(rhs) }.clone();
        Ok(into_raw(Binary.new_expr(operator, [lhs, rhs])))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_not(
    _env: EnvUnowned,
    _class: JClass,
    child: jlong,
) -> jlong {
    let child = unsafe { expr_ref(child) }.clone();
    into_raw(not(child))
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_isNull(
    _env: EnvUnowned,
    _class: JClass,
    child: jlong,
) -> jlong {
    let child = unsafe { expr_ref(child) }.clone();
    into_raw(is_null(child))
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_isNotNull(
    _env: EnvUnowned,
    _class: JClass,
    child: jlong,
) -> jlong {
    let child = unsafe { expr_ref(child) }.clone();
    into_raw(is_not_null(child))
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_like(
    _env: EnvUnowned,
    _class: JClass,
    child: jlong,
    pattern: jlong,
    negated: jboolean,
    case_insensitive: jboolean,
) -> jlong {
    let child = unsafe { expr_ref(child) }.clone();
    let pattern = unsafe { expr_ref(pattern) }.clone();
    into_raw(Like.new_expr(
        LikeOptions {
            negated,
            case_insensitive,
        },
        [child, pattern],
    ))
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_between(
    mut env: EnvUnowned,
    _class: JClass,
    value: jlong,
    lower: jlong,
    upper: jlong,
    lower_strict: jboolean,
    upper_strict: jboolean,
) -> jlong {
    try_or_throw(&mut env, |_| {
        let value = unsafe { expr_ref(value) }.clone();
        let lower = unsafe { expr_ref(lower) }.clone();
        let upper = unsafe { expr_ref(upper) }.clone();
        Ok(into_raw(between(
            value,
            lower,
            upper,
            BetweenOptions {
                lower_strict: strict_from_bool(lower_strict),
                upper_strict: strict_from_bool(upper_strict),
            },
        )))
    })
}

fn strict_from_bool(value: jboolean) -> StrictComparison {
    if value {
        StrictComparison::Strict
    } else {
        StrictComparison::NonStrict
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_literalBool(
    _env: EnvUnowned,
    _class: JClass,
    value: jboolean,
    is_null_flag: jboolean,
) -> jlong {
    if is_null_flag {
        let scalar = Scalar::null_native::<bool>();
        return into_raw(lit(scalar));
    }
    into_raw(lit(value))
}

macro_rules! literal_primitive {
    ($fname:ident, $jty:ty, $rust:ty) => {
        #[unsafe(no_mangle)]
        pub extern "system" fn $fname(
            _env: EnvUnowned,
            _class: JClass,
            value: $jty,
            is_null_flag: jboolean,
        ) -> jlong {
            if is_null_flag {
                let scalar = Scalar::null_native::<$rust>();
                return into_raw(lit(scalar));
            }
            into_raw(lit(value as $rust))
        }
    };
}

literal_primitive!(Java_dev_vortex_jni_NativeExpression_literalI8, jbyte, i8);
literal_primitive!(Java_dev_vortex_jni_NativeExpression_literalI16, jshort, i16);
literal_primitive!(Java_dev_vortex_jni_NativeExpression_literalI32, jint, i32);
literal_primitive!(Java_dev_vortex_jni_NativeExpression_literalI64, jlong, i64);
literal_primitive!(Java_dev_vortex_jni_NativeExpression_literalF32, jfloat, f32);
literal_primitive!(
    Java_dev_vortex_jni_NativeExpression_literalF64,
    jdouble,
    f64
);

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_literalString(
    mut env: EnvUnowned,
    _class: JClass,
    value: JString,
) -> jlong {
    try_or_throw(&mut env, |env| {
        if value.is_null() {
            let scalar = Scalar::null_native::<String>();
            return Ok(into_raw(lit(scalar)));
        }
        let s: String = value.try_to_string(env)?;
        Ok(into_raw(lit(s)))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_literalBinary(
    mut env: EnvUnowned,
    _class: JClass,
    value: JByteArray,
) -> jlong {
    try_or_throw(&mut env, |env| {
        if value.is_null() {
            let scalar = Scalar::null_native::<vortex::buffer::ByteBuffer>();
            return Ok(into_raw(lit(scalar)));
        }
        let bytes: Vec<u8> = env.convert_byte_array(&value)?;
        Ok(into_raw(lit(bytes.as_slice())))
    })
}

/// Build a decimal literal from a two's-complement big-endian byte representation of the
/// unscaled value (the format produced by Java's `BigInteger.toByteArray()`).
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_literalDecimal(
    mut env: EnvUnowned,
    _class: JClass,
    unscaled_big_endian: JByteArray,
    precision: jint,
    scale: jint,
    is_null_flag: jboolean,
) -> jlong {
    try_or_throw(&mut env, |env| {
        let precision = u8::try_from(precision)
            .map_err(|_| vortex_err!("decimal precision out of range: {precision}"))?;
        let scale =
            i8::try_from(scale).map_err(|_| vortex_err!("decimal scale out of range: {scale}"))?;
        let decimal_dtype = DecimalDType::try_new(precision, scale)?;
        if is_null_flag {
            return Ok(into_raw(lit(Scalar::null(DType::Decimal(
                decimal_dtype,
                Nullability::Nullable,
            )))));
        }
        if unscaled_big_endian.len(env)? > 32 {
            throw_runtime!("Decimal value must fit with 32 bytes");
        }

        let bytes = env.convert_byte_array(&unscaled_big_endian)?;
        let decimal_value = decimal_value_from_be_bytes(&bytes, &decimal_dtype)?;
        let scalar = Scalar::try_new(
            DType::Decimal(decimal_dtype, Nullability::NonNullable),
            Some(ScalarValue::from(decimal_value)),
        )?;
        Ok(into_raw(lit(scalar)))
    })
}

/// Decode a two's-complement big-endian byte array (Java `BigInteger.toByteArray()` format)
/// into the smallest [`DecimalValue`] variant that can hold the precision.
fn decimal_value_from_be_bytes(
    bytes: &[u8],
    dtype: &DecimalDType,
) -> Result<DecimalValue, JNIError> {
    if bytes.is_empty() {
        throw_runtime!("decimal unscaled value must have at least one byte");
    }
    let value = i256_from_twos_complement_be(bytes);
    // Pick the narrowest backing integer that fits the dtype's precision.
    let required_bits = dtype.required_bit_width();
    if required_bits <= 8 {
        let v =
            BigCast::from(value).ok_or_else(|| vortex_err!("decimal value does not fit in i8"))?;
        Ok(DecimalValue::I8(v))
    } else if required_bits <= 16 {
        let v =
            BigCast::from(value).ok_or_else(|| vortex_err!("decimal value does not fit in i16"))?;
        Ok(DecimalValue::I16(v))
    } else if required_bits <= 32 {
        let v =
            BigCast::from(value).ok_or_else(|| vortex_err!("decimal value does not fit in i32"))?;
        Ok(DecimalValue::I32(v))
    } else if required_bits <= 64 {
        let v =
            BigCast::from(value).ok_or_else(|| vortex_err!("decimal value does not fit in i64"))?;
        Ok(DecimalValue::I64(v))
    } else if required_bits <= 128 {
        let v = value
            .maybe_i128()
            .ok_or_else(|| vortex_err!("decimal value does not fit in i128"))?;
        Ok(DecimalValue::I128(v))
    } else {
        Ok(DecimalValue::I256(value))
    }
}

/// Sign-extend a two's-complement big-endian byte slice into an `i256`.
fn i256_from_twos_complement_be(bytes: &[u8]) -> vortex::dtype::i256 {
    let mut le = [0u8; 32];
    let len = bytes.len().min(32);
    // Most significant byte comes first in big-endian; copy lowest 32 bytes reversed into LE.
    for (i, b) in bytes.iter().rev().take(len).enumerate() {
        le[i] = *b;
    }
    // If the original value is negative (high bit of the most-significant byte is set),
    // sign-extend the remaining high bytes with 0xff.
    if !bytes.is_empty() && (bytes[0] & 0x80) != 0 {
        for byte in &mut le[len..] {
            *byte = 0xff;
        }
    }
    vortex::dtype::i256::from_le_bytes(le)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_literalDate(
    mut env: EnvUnowned,
    _class: JClass,
    value: jlong,
    time_unit_tag: jbyte,
    is_null_flag: jboolean,
) -> jlong {
    try_or_throw(&mut env, |_| {
        let unit = parse_time_unit(time_unit_tag)?;
        let nullability = if is_null_flag {
            Nullability::Nullable
        } else {
            Nullability::NonNullable
        };
        let ext = Date::try_new(unit, nullability)?;
        let dtype = DType::Extension(ext.erased());
        if is_null_flag {
            return Ok(into_raw(lit(Scalar::null(dtype))));
        }
        let storage_value = match unit {
            TimeUnit::Days => ScalarValue::from(
                i32::try_from(value)
                    .map_err(|_| vortex_err!("date value does not fit in i32 days: {value}"))?,
            ),
            TimeUnit::Milliseconds => ScalarValue::from(value),
            other => throw_runtime!("date does not support time unit {other}"),
        };
        Ok(into_raw(lit(Scalar::try_new(dtype, Some(storage_value))?)))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_literalTimestamp(
    mut env: EnvUnowned,
    _class: JClass,
    value: jlong,
    time_unit_tag: jbyte,
    timezone: JString,
    is_null_flag: jboolean,
) -> jlong {
    try_or_throw(&mut env, |env| {
        let unit = parse_time_unit(time_unit_tag)?;
        let tz: Option<Arc<str>> = if timezone.is_null() {
            None
        } else {
            let s: String = timezone.try_to_string(env)?;
            Some(Arc::<str>::from(s.as_str()))
        };
        let nullability = if is_null_flag {
            Nullability::Nullable
        } else {
            Nullability::NonNullable
        };
        let ext = Timestamp::new_with_tz(unit, tz, nullability);
        let dtype = DType::Extension(ext.erased());
        if is_null_flag {
            return Ok(into_raw(lit(Scalar::null(dtype))));
        }
        Ok(into_raw(lit(Scalar::try_new(
            dtype,
            Some(ScalarValue::from(value)),
        )?)))
    })
}

/// Build a typed null literal whose nullable dtype is selected by `dtype_tag`.
///
/// Tag values intentionally do not overlap with [`parse_time_unit`].
/// See `dev.vortex.api.Expression.DType` on the Java side for the source of truth.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_literalNull(
    mut env: EnvUnowned,
    _class: JClass,
    dtype_tag: jbyte,
) -> jlong {
    try_or_throw(&mut env, |_| {
        let dtype = match dtype_tag {
            0 => DType::Bool(Nullability::Nullable),
            1 => DType::Primitive(PType::I8, Nullability::Nullable),
            2 => DType::Primitive(PType::I16, Nullability::Nullable),
            3 => DType::Primitive(PType::I32, Nullability::Nullable),
            4 => DType::Primitive(PType::I64, Nullability::Nullable),
            5 => DType::Primitive(PType::F32, Nullability::Nullable),
            6 => DType::Primitive(PType::F64, Nullability::Nullable),
            7 => DType::Utf8(Nullability::Nullable),
            8 => DType::Binary(Nullability::Nullable),
            other => throw_runtime!("unknown null dtype tag: {other}"),
        };
        Ok(into_raw(lit(Scalar::null(dtype))))
    })
}
