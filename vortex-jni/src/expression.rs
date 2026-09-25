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
use vortex::authored_expr;
use vortex::authored_expr::Expression;
use vortex::dtype::DType;
use vortex::dtype::DecimalDType;
use vortex::dtype::FieldName;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::dtype::extension::ExtDType;
use vortex::error::vortex_err;
use vortex::extension::datetime::Date;
use vortex::extension::datetime::TimeUnit;
use vortex::extension::datetime::Timestamp;
use vortex::extension::uuid::Uuid;
use vortex::extension::uuid::UuidMetadata;
use vortex::scalar::Scalar;
use vortex::scalar::ScalarValue;
use vortex::scalar_fn::fns::between::BetweenOptions;
use vortex::scalar_fn::fns::between::StrictComparison;
use vortex::scalar_fn::fns::like::LikeOptions;
use vortex::scalar_fn::fns::merge::DuplicateHandling;
use vortex::scalar_fn::fns::pack::PackOptions;
use vortex::scalar_fn::fns::select::FieldSelection;

use crate::errors::JNIError;
use crate::errors::try_or_throw;
use crate::expression_args::decimal_value_from_be_bytes;
use crate::expression_args::parse_op;
use crate::expression_args::parse_time_unit;

fn into_raw(expr: Expression) -> jlong {
    Box::into_raw(Box::new(expr)) as jlong
}

/// SAFETY: pointer must originate from [`into_raw`] and not yet be freed.
pub(crate) unsafe fn expr_ref<'a>(ptr: jlong) -> &'a Expression {
    debug_assert!(ptr != 0, "null expression pointer");
    unsafe { &*(ptr as *const Expression) }
}

/// Parse a merge [`DuplicateHandling`] strategy from its wire-encoded byte tag.
///
/// See `dev.vortex.api.Expression.DuplicateHandling` on the Java side for the source of truth.
fn parse_duplicate_handling(tag: jbyte) -> Result<DuplicateHandling, JNIError> {
    Ok(match tag {
        0 => DuplicateHandling::RightMost,
        1 => DuplicateHandling::Error,
        other => throw_runtime!("unknown duplicate handling code: {other}"),
    })
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
    into_raw(authored_expr::root())
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
        Ok(into_raw(Expression::call("column", field, [child])))
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
        Ok(into_raw(Expression::call(
            "select",
            FieldSelection::include(fields.into()),
            [child],
        )))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_pack(
    mut env: EnvUnowned,
    _class: JClass,
    field_names: JObjectArray,
    expressions: JLongArray,
    nullable: jboolean,
) -> jlong {
    try_or_throw(&mut env, |env| {
        let count = field_names.len(env)?;
        let expressions = unsafe { expressions.get_elements(env, ReleaseMode::NoCopyBack)? };
        let mut elements = Vec::with_capacity(count);

        for idx in 0..count {
            let obj = field_names.get_element(env, idx)?;
            let s = env.cast_local::<JString>(obj)?;
            let name: FieldName = s.try_to_string(env)?.into();

            let expr_ptr = *expressions.get(idx).ok_or_else(|| -> JNIError {
                vortex_err!("missing pack expression child").into()
            })?;
            let expr = unsafe { expr_ref(expr_ptr) }.clone();

            elements.push((name, expr));
        }

        let (names, children): (Vec<_>, Vec<_>) = elements.into_iter().unzip();
        Ok(into_raw(Expression::call(
            "pack",
            PackOptions {
                names: names.into(),
                nullability: nullable.into(),
            },
            children,
        )))
    })
}

/// Merge zero or more struct-returning expressions into a single struct.
///
/// `duplicate_handling` selects how shared field names are resolved (see
/// [`parse_duplicate_handling`]). An empty `expressions` array yields an empty struct.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_merge(
    mut env: EnvUnowned,
    _class: JClass,
    expressions: JLongArray,
    duplicate_handling: jbyte,
) -> jlong {
    try_or_throw(&mut env, |env| {
        let exprs = collect_operands(env, &expressions)?;
        let handling = parse_duplicate_handling(duplicate_handling)?;
        Ok(into_raw(Expression::call("merge", handling, exprs)))
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
        authored_expr::and_collect(exprs)
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
        authored_expr::or_collect(exprs)
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
        Ok(into_raw(authored_expr::binary(operator, lhs, rhs)))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_not(
    _env: EnvUnowned,
    _class: JClass,
    child: jlong,
) -> jlong {
    let child = unsafe { expr_ref(child) }.clone();
    into_raw(authored_expr::not(child))
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_isNull(
    _env: EnvUnowned,
    _class: JClass,
    child: jlong,
) -> jlong {
    let child = unsafe { expr_ref(child) }.clone();
    into_raw(authored_expr::is_null(child))
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_isNotNull(
    _env: EnvUnowned,
    _class: JClass,
    child: jlong,
) -> jlong {
    let child = unsafe { expr_ref(child) }.clone();
    into_raw(authored_expr::is_not_null(child))
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
    into_raw(Expression::call(
        "like",
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
        Ok(into_raw(Expression::call(
            "between",
            BetweenOptions {
                lower_strict: strict_from_bool(lower_strict),
                upper_strict: strict_from_bool(upper_strict),
            },
            [value, lower, upper],
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
        return into_raw(authored_expr::lit(scalar));
    }
    into_raw(authored_expr::lit(value))
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
                return into_raw(authored_expr::lit(scalar));
            }
            into_raw(authored_expr::lit(value as $rust))
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
            return Ok(into_raw(authored_expr::lit(scalar)));
        }
        let s: String = value.try_to_string(env)?;
        Ok(into_raw(authored_expr::lit(s)))
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
            return Ok(into_raw(authored_expr::lit(scalar)));
        }
        let bytes: Vec<u8> = env.convert_byte_array(&value)?;
        Ok(into_raw(authored_expr::lit(bytes.as_slice())))
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
            return Ok(into_raw(authored_expr::lit(Scalar::null(DType::Decimal(
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
        Ok(into_raw(authored_expr::lit(scalar)))
    })
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
            return Ok(into_raw(authored_expr::lit(Scalar::null(dtype))));
        }
        let storage_value = match unit {
            TimeUnit::Days => ScalarValue::from(
                i32::try_from(value)
                    .map_err(|_| vortex_err!("date value does not fit in i32 days: {value}"))?,
            ),
            TimeUnit::Milliseconds => ScalarValue::from(value),
            other => throw_runtime!("date does not support time unit {other}"),
        };
        Ok(into_raw(authored_expr::lit(Scalar::try_new(
            dtype,
            Some(storage_value),
        )?)))
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
            return Ok(into_raw(authored_expr::lit(Scalar::null(dtype))));
        }
        Ok(into_raw(authored_expr::lit(Scalar::try_new(
            dtype,
            Some(ScalarValue::from(value)),
        )?)))
    })
}

/// Number of bytes in a UUID's big-endian representation.
const UUID_BYTE_LEN: usize = 16;

/// Build the version-agnostic UUID extension [`DType`] with the given nullability.
///
/// The storage is a non-nullable `FixedSizeList(U8, 16)`, matching Vortex's UUID extension and
/// Arrow's canonical UUID type. The metadata records no version constraint, so the dtype is
/// compatible with any UUID column regardless of the UUID versions it contains.
pub(crate) fn uuid_dtype(nullability: Nullability) -> Result<DType, JNIError> {
    let list_size = u32::try_from(UUID_BYTE_LEN)
        .map_err(|_| vortex_err!("UUID byte length {UUID_BYTE_LEN} does not fit in u32"))?;
    let storage_dtype = DType::FixedSizeList(
        Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
        list_size,
        nullability,
    );
    let ext = ExtDType::<Uuid>::try_new(UuidMetadata::default(), storage_dtype)?;
    Ok(DType::Extension(ext.erased()))
}

/// Build a non-null UUID [`Scalar`] from its 16-byte big-endian representation.
pub(crate) fn uuid_scalar(bytes: &[u8]) -> Result<Scalar, JNIError> {
    if bytes.len() != UUID_BYTE_LEN {
        throw_runtime!(
            "UUID literal must be exactly {UUID_BYTE_LEN} bytes, got {}",
            bytes.len()
        );
    }
    let children: Vec<Scalar> = bytes
        .iter()
        .map(|&b| Scalar::primitive(b, Nullability::NonNullable))
        .collect();
    let storage = Scalar::fixed_size_list(
        DType::Primitive(PType::U8, Nullability::NonNullable),
        children,
        Nullability::NonNullable,
    );
    Ok(Scalar::try_new(
        uuid_dtype(Nullability::NonNullable)?,
        storage.into_value(),
    )?)
}

/// Build a UUID literal from its 16-byte big-endian representation.
///
/// When `is_null_flag` is true the `value` array is ignored and a typed null UUID literal is
/// produced. Otherwise `value` must hold exactly 16 bytes in big-endian (network) order — the
/// same layout as a `java.util.UUID` written most-significant-bits first, and Arrow's canonical
/// UUID extension. The literal is version-agnostic so it compares against any UUID column.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_literalUuid(
    mut env: EnvUnowned,
    _class: JClass,
    value: JByteArray,
    is_null_flag: jboolean,
) -> jlong {
    try_or_throw(&mut env, |env| {
        if is_null_flag {
            return Ok(into_raw(authored_expr::lit(Scalar::null(uuid_dtype(
                Nullability::Nullable,
            )?))));
        }
        if value.is_null() {
            throw_runtime!("UUID literal bytes must not be null");
        }
        let bytes = env.convert_byte_array(&value)?;
        Ok(into_raw(authored_expr::lit(uuid_scalar(&bytes)?)))
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
        Ok(into_raw(authored_expr::lit(Scalar::null(dtype))))
    })
}
