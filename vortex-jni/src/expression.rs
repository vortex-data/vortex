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
use vortex::dtype::FieldName;
use vortex::expr::Expression;
use vortex::expr::and_collect;
use vortex::expr::get_item;
use vortex::expr::is_null;
use vortex::expr::lit;
use vortex::expr::not;
use vortex::expr::or_collect;
use vortex::expr::root;
use vortex::expr::select;
use vortex::scalar::Scalar;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::binary::Binary;
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
            .ok_or_else(|| vortex::error::vortex_err!("empty AND expression").into())
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
            .ok_or_else(|| vortex::error::vortex_err!("empty OR expression").into())
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
