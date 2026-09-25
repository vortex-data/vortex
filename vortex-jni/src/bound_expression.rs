// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! JNI handles for expressions that have been checked against a data source dtype.

use std::sync::Arc;

use jni::EnvUnowned;
use jni::objects::JByteArray;
use jni::objects::JClass;
use jni::objects::JObjectArray;
use jni::objects::JString;
use jni::sys::jboolean;
use jni::sys::jbyte;
use jni::sys::jdouble;
use jni::sys::jfloat;
use jni::sys::jint;
use jni::sys::jlong;
use jni::sys::jshort;
use vortex::array::expr;
use vortex::array::expr::BoundExpression;
use vortex::dtype::DType;
use vortex::dtype::DecimalDType;
use vortex::dtype::FieldName;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::error::vortex_err;
use vortex::extension::datetime::Date;
use vortex::extension::datetime::TimeUnit;
use vortex::extension::datetime::Timestamp;
use vortex::layout::layouts::row_idx::row_idx;
use vortex::scalar::Scalar;
use vortex::scalar::ScalarValue;
use vortex::scalar_fn::EmptyOptions;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::get_item::GetItem;
use vortex::scalar_fn::fns::is_not_null::IsNotNull;
use vortex::scalar_fn::fns::is_null::IsNull;
use vortex::scalar_fn::fns::like::Like;
use vortex::scalar_fn::fns::like::LikeOptions;
use vortex::scalar_fn::fns::not::Not;
use vortex::scalar_fn::fns::select::FieldSelection;
use vortex::scalar_fn::fns::select::Select;

use crate::data_source::NativeDataSource;
use crate::errors::try_or_throw;
use crate::expression::expr_ref;
use crate::expression_args::decimal_value_from_be_bytes;
use crate::expression_args::parse_op;
use crate::expression_args::parse_time_unit;

fn into_raw(expression: BoundExpression) -> jlong {
    Box::into_raw(Box::new(expression)) as jlong
}

fn literal(scalar: impl Into<Scalar>) -> jlong {
    into_raw(expr::lit(scalar))
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeBoundExpression_rowIdx(
    _env: EnvUnowned,
    _class: JClass,
) -> jlong {
    into_raw(row_idx())
}

unsafe fn bound_ref<'a>(pointer: jlong) -> &'a BoundExpression {
    unsafe { &*(pointer as *const BoundExpression) }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeBoundExpression_bind(
    mut env: EnvUnowned,
    _class: JClass,
    expression_ptr: jlong,
    data_source_ptr: jlong,
) -> jlong {
    try_or_throw(&mut env, |_env| {
        let data_source = unsafe { NativeDataSource::from_ptr(data_source_ptr) };
        let dtype = data_source.inner().dtype();
        let expression = unsafe { expr_ref(expression_ptr) };
        let bound = expression.bind(dtype)?.optimize_recursive()?;
        Ok(into_raw(bound))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeBoundExpression_root(
    mut env: EnvUnowned,
    _class: JClass,
    data_source_ptr: jlong,
) -> jlong {
    try_or_throw(&mut env, |_env| {
        let data_source = unsafe { NativeDataSource::from_ptr(data_source_ptr) };
        Ok(into_raw(BoundExpression::new_root(
            data_source.inner().dtype().clone(),
        )))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeBoundExpression_getItem(
    mut env: EnvUnowned,
    _class: JClass,
    name: JString,
    child_ptr: jlong,
) -> jlong {
    try_or_throw(&mut env, |env| {
        let name: FieldName = name.try_to_string(env)?.into();
        let child = unsafe { bound_ref(child_ptr) }.clone();
        Ok(into_raw(GetItem.try_new_bound_expr(name, [child])?))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeBoundExpression_select(
    mut env: EnvUnowned,
    _class: JClass,
    names: JObjectArray,
    child_ptr: jlong,
) -> jlong {
    try_or_throw(&mut env, |env| {
        let count = names.len(env)?;
        let mut fields = Vec::with_capacity(count);
        for idx in 0..count {
            let name = names.get_element(env, idx)?;
            let name = env.cast_local::<JString>(name)?;
            fields.push(FieldName::from(name.try_to_string(env)?));
        }
        let child = unsafe { bound_ref(child_ptr) }.clone();
        Ok(into_raw(Select.try_new_bound_expr(
            FieldSelection::include(fields.into()),
            [child],
        )?))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeBoundExpression_literalBool(
    _env: EnvUnowned,
    _class: JClass,
    value: jboolean,
    is_null: jboolean,
) -> jlong {
    if is_null {
        literal(Scalar::null_native::<bool>())
    } else {
        literal(value)
    }
}

macro_rules! literal_primitive {
    ($name:ident, $jni:ty, $native:ty) => {
        #[unsafe(no_mangle)]
        pub extern "system" fn $name(
            _env: EnvUnowned,
            _class: JClass,
            value: $jni,
            is_null: jboolean,
        ) -> jlong {
            if is_null {
                literal(Scalar::null_native::<$native>())
            } else {
                literal(value as $native)
            }
        }
    };
}

literal_primitive!(
    Java_dev_vortex_jni_NativeBoundExpression_literalI8,
    jbyte,
    i8
);
literal_primitive!(
    Java_dev_vortex_jni_NativeBoundExpression_literalI16,
    jshort,
    i16
);
literal_primitive!(
    Java_dev_vortex_jni_NativeBoundExpression_literalI32,
    jint,
    i32
);
literal_primitive!(
    Java_dev_vortex_jni_NativeBoundExpression_literalI64,
    jlong,
    i64
);

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeBoundExpression_literalU64(
    _env: EnvUnowned,
    _class: JClass,
    bits: jlong,
) -> jlong {
    literal(bits as u64)
}

literal_primitive!(
    Java_dev_vortex_jni_NativeBoundExpression_literalF32,
    jfloat,
    f32
);
literal_primitive!(
    Java_dev_vortex_jni_NativeBoundExpression_literalF64,
    jdouble,
    f64
);

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeBoundExpression_literalString(
    mut env: EnvUnowned,
    _class: JClass,
    value: JString,
) -> jlong {
    try_or_throw(&mut env, |env| {
        if value.is_null() {
            return Ok(literal(Scalar::null_native::<String>()));
        }
        Ok(literal(value.try_to_string(env)?))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeBoundExpression_literalBinary(
    mut env: EnvUnowned,
    _class: JClass,
    value: JByteArray,
) -> jlong {
    try_or_throw(&mut env, |env| {
        if value.is_null() {
            return Ok(literal(Scalar::null_native::<vortex::buffer::ByteBuffer>()));
        }
        let bytes = env.convert_byte_array(&value)?;
        Ok(literal(bytes.as_slice()))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeBoundExpression_literalNull(
    mut env: EnvUnowned,
    _class: JClass,
    tag: jbyte,
) -> jlong {
    try_or_throw(&mut env, |_| {
        let nullable = Nullability::Nullable;
        let dtype = match tag {
            0 => DType::Bool(nullable),
            1 => DType::Primitive(PType::I8, nullable),
            2 => DType::Primitive(PType::I16, nullable),
            3 => DType::Primitive(PType::I32, nullable),
            4 => DType::Primitive(PType::I64, nullable),
            5 => DType::Primitive(PType::F32, nullable),
            6 => DType::Primitive(PType::F64, nullable),
            7 => DType::Utf8(nullable),
            8 => DType::Binary(nullable),
            other => return Err(vortex_err!("unknown null dtype tag: {other}").into()),
        };
        Ok(literal(Scalar::null(dtype)))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeBoundExpression_literalDate(
    mut env: EnvUnowned,
    _class: JClass,
    value: jlong,
    unit_tag: jbyte,
    is_null: jboolean,
) -> jlong {
    try_or_throw(&mut env, |_| {
        let unit = parse_time_unit(unit_tag)?;
        let nullability = if is_null {
            Nullability::Nullable
        } else {
            Nullability::NonNullable
        };
        let dtype = DType::Extension(Date::try_new(unit, nullability)?.erased());
        if is_null {
            return Ok(literal(Scalar::null(dtype)));
        }
        let storage = match unit {
            TimeUnit::Days => ScalarValue::from(
                i32::try_from(value)
                    .map_err(|_| vortex_err!("date value does not fit in i32 days: {value}"))?,
            ),
            TimeUnit::Milliseconds => ScalarValue::from(value),
            other => return Err(vortex_err!("date does not support time unit {other}").into()),
        };
        Ok(literal(Scalar::try_new(dtype, Some(storage))?))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeBoundExpression_literalTimestamp(
    mut env: EnvUnowned,
    _class: JClass,
    value: jlong,
    unit_tag: jbyte,
    timezone: JString,
    is_null: jboolean,
) -> jlong {
    try_or_throw(&mut env, |env| {
        let unit = parse_time_unit(unit_tag)?;
        let tz: Option<Arc<str>> = if timezone.is_null() {
            None
        } else {
            Some(Arc::from(timezone.try_to_string(env)?.as_str()))
        };
        let nullability = if is_null {
            Nullability::Nullable
        } else {
            Nullability::NonNullable
        };
        let dtype = DType::Extension(Timestamp::new_with_tz(unit, tz, nullability).erased());
        if is_null {
            return Ok(literal(Scalar::null(dtype)));
        }
        Ok(literal(Scalar::try_new(
            dtype,
            Some(ScalarValue::from(value)),
        )?))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeBoundExpression_literalDecimal(
    mut env: EnvUnowned,
    _class: JClass,
    unscaled: JByteArray,
    precision: jint,
    scale: jint,
    is_null: jboolean,
) -> jlong {
    try_or_throw(&mut env, |env| {
        let precision = u8::try_from(precision)
            .map_err(|_| vortex_err!("decimal precision out of range: {precision}"))?;
        let scale =
            i8::try_from(scale).map_err(|_| vortex_err!("decimal scale out of range: {scale}"))?;
        let decimal_dtype = DecimalDType::try_new(precision, scale)?;
        if is_null {
            return Ok(literal(Scalar::null(DType::Decimal(
                decimal_dtype,
                Nullability::Nullable,
            ))));
        }
        if unscaled.len(env)? > 32 {
            return Err(vortex_err!("Decimal value must fit with 32 bytes").into());
        }
        let bytes = env.convert_byte_array(&unscaled)?;
        let value = decimal_value_from_be_bytes(&bytes, &decimal_dtype)?;
        let scalar = Scalar::try_new(
            DType::Decimal(decimal_dtype, Nullability::NonNullable),
            Some(ScalarValue::from(value)),
        )?;
        Ok(literal(scalar))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeBoundExpression_binary(
    mut env: EnvUnowned,
    _class: JClass,
    op: jbyte,
    lhs_ptr: jlong,
    rhs_ptr: jlong,
) -> jlong {
    try_or_throw(&mut env, |_env| {
        let operator = parse_op(op)?;
        let lhs = unsafe { bound_ref(lhs_ptr) }.clone();
        let rhs = unsafe { bound_ref(rhs_ptr) }.clone();
        Ok(into_raw(Binary.try_new_bound_expr(operator, [lhs, rhs])?))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeBoundExpression_not(
    mut env: EnvUnowned,
    _class: JClass,
    child_ptr: jlong,
) -> jlong {
    try_or_throw(&mut env, |_env| {
        let child = unsafe { bound_ref(child_ptr) }.clone();
        Ok(into_raw(Not.try_new_bound_expr(EmptyOptions, [child])?))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeBoundExpression_isNull(
    mut env: EnvUnowned,
    _class: JClass,
    child_ptr: jlong,
) -> jlong {
    try_or_throw(&mut env, |_env| {
        let child = unsafe { bound_ref(child_ptr) }.clone();
        Ok(into_raw(IsNull.try_new_bound_expr(EmptyOptions, [child])?))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeBoundExpression_isNotNull(
    mut env: EnvUnowned,
    _class: JClass,
    child_ptr: jlong,
) -> jlong {
    try_or_throw(&mut env, |_env| {
        let child = unsafe { bound_ref(child_ptr) }.clone();
        Ok(into_raw(
            IsNotNull.try_new_bound_expr(EmptyOptions, [child])?,
        ))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeBoundExpression_like(
    mut env: EnvUnowned,
    _class: JClass,
    value_ptr: jlong,
    pattern_ptr: jlong,
    negated: jboolean,
    case_insensitive: jboolean,
) -> jlong {
    try_or_throw(&mut env, |_env| {
        let value = unsafe { bound_ref(value_ptr) }.clone();
        let pattern = unsafe { bound_ref(pattern_ptr) }.clone();
        Ok(into_raw(Like.try_new_bound_expr(
            LikeOptions {
                negated,
                case_insensitive,
            },
            [value, pattern],
        )?))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeBoundExpression_free(
    _env: EnvUnowned,
    _class: JClass,
    pointer: jlong,
) {
    if pointer != 0 {
        drop(unsafe { Box::from_raw(pointer as *mut BoundExpression) });
    }
}
