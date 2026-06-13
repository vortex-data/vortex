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
use vortex::dtype::FieldNames;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::dtype::extension::ExtDType;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::BoundExpr;
use vortex::expr::lit;
use vortex::expr::root;
use vortex::extension::datetime::Date;
use vortex::extension::datetime::TimeUnit;
use vortex::extension::datetime::Timestamp;
use vortex::extension::uuid::Uuid;
use vortex::extension::uuid::UuidMetadata;
use vortex::layout::layouts::row_idx::row_idx;
use vortex::scalar::DecimalValue;
use vortex::scalar::Scalar;
use vortex::scalar::ScalarValue;
use vortex::scalar_fn::EmptyOptions;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::between::Between;
use vortex::scalar_fn::fns::between::BetweenOptions;
use vortex::scalar_fn::fns::between::StrictComparison;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::get_item::GetItem;
use vortex::scalar_fn::fns::is_not_null::IsNotNull;
use vortex::scalar_fn::fns::is_null::IsNull;
use vortex::scalar_fn::fns::like::Like;
use vortex::scalar_fn::fns::like::LikeOptions;
use vortex::scalar_fn::fns::merge::DuplicateHandling;
use vortex::scalar_fn::fns::merge::Merge;
use vortex::scalar_fn::fns::not::Not;
use vortex::scalar_fn::fns::operators::Operator;
use vortex::scalar_fn::fns::pack::Pack;
use vortex::scalar_fn::fns::pack::PackOptions;
use vortex::scalar_fn::fns::select::FieldSelection;
use vortex::scalar_fn::fns::select::Select;

use crate::errors::JNIError;
use crate::errors::try_or_throw;

#[derive(Clone)]
enum NativeExpr {
    Bound(BoundExpr),
    Root,
    GetItem {
        field: FieldName,
        child: Arc<NativeExpr>,
    },
    Select {
        fields: FieldNames,
        child: Arc<NativeExpr>,
    },
    Pack {
        elements: Vec<(FieldName, Arc<NativeExpr>)>,
        nullability: Nullability,
    },
    Merge {
        expressions: Vec<Arc<NativeExpr>>,
        duplicate_handling: DuplicateHandling,
    },
    And(Vec<Arc<NativeExpr>>),
    Or(Vec<Arc<NativeExpr>>),
    Binary {
        operator: Operator,
        lhs: Arc<NativeExpr>,
        rhs: Arc<NativeExpr>,
    },
    Not(Arc<NativeExpr>),
    IsNull(Arc<NativeExpr>),
    IsNotNull(Arc<NativeExpr>),
    Like {
        child: Arc<NativeExpr>,
        pattern: Arc<NativeExpr>,
        options: LikeOptions,
    },
    Between {
        value: Arc<NativeExpr>,
        lower: Arc<NativeExpr>,
        upper: Arc<NativeExpr>,
        options: BetweenOptions,
    },
}

impl NativeExpr {
    fn bind(&self, scope: &DType) -> VortexResult<BoundExpr> {
        match self {
            Self::Bound(expr) => Ok(expr.clone()),
            Self::Root => Ok(root(scope.clone())),
            Self::GetItem { field, child } => {
                GetItem.try_new_expr(field.clone(), [child.bind(scope)?])
            }
            Self::Select { fields, child } => Select.try_new_expr(
                FieldSelection::Include(fields.clone()),
                [child.bind(scope)?],
            ),
            Self::Pack {
                elements,
                nullability,
            } => {
                let (names, values): (Vec<_>, Vec<_>) = elements
                    .iter()
                    .map(|(name, expr)| Ok((name.clone(), expr.bind(scope)?)))
                    .collect::<VortexResult<Vec<_>>>()?
                    .into_iter()
                    .unzip();
                Pack.try_new_expr(
                    PackOptions {
                        names: names.into(),
                        nullability: *nullability,
                    },
                    values,
                )
            }
            Self::Merge {
                expressions,
                duplicate_handling,
            } => Merge.try_new_expr(
                *duplicate_handling,
                expressions
                    .iter()
                    .map(|expression| expression.bind(scope))
                    .collect::<VortexResult<Vec<_>>>()?,
            ),
            Self::And(expressions) => vortex::expr::try_and_collect(
                expressions
                    .iter()
                    .map(|expression| expression.bind(scope))
                    .collect::<VortexResult<Vec<_>>>()?,
            )?
            .ok_or_else(|| vortex_err!("empty AND expression")),
            Self::Or(expressions) => vortex::expr::try_or_collect(
                expressions
                    .iter()
                    .map(|expression| expression.bind(scope))
                    .collect::<VortexResult<Vec<_>>>()?,
            )?
            .ok_or_else(|| vortex_err!("empty OR expression")),
            Self::Binary { operator, lhs, rhs } => {
                Binary.try_new_expr(*operator, [lhs.bind(scope)?, rhs.bind(scope)?])
            }
            Self::Not(child) => Not.try_new_expr(EmptyOptions, [child.bind(scope)?]),
            Self::IsNull(child) => IsNull.try_new_expr(EmptyOptions, [child.bind(scope)?]),
            Self::IsNotNull(child) => IsNotNull.try_new_expr(EmptyOptions, [child.bind(scope)?]),
            Self::Like {
                child,
                pattern,
                options,
            } => Like.try_new_expr(*options, [child.bind(scope)?, pattern.bind(scope)?]),
            Self::Between {
                value,
                lower,
                upper,
                options,
            } => Between.try_new_expr(
                options.clone(),
                [value.bind(scope)?, lower.bind(scope)?, upper.bind(scope)?],
            ),
        }
    }
}

impl Drop for NativeExpr {
    fn drop(&mut self) {
        let mut children_to_drop = Vec::new();
        self.take_children(&mut children_to_drop);

        while let Some(mut child) = children_to_drop.pop() {
            let Some(child) = Arc::get_mut(&mut child) else {
                continue;
            };
            child.take_children(&mut children_to_drop);
        }
    }
}

impl NativeExpr {
    fn take_children(&mut self, children_to_drop: &mut Vec<Arc<Self>>) {
        match self {
            Self::Bound(_) | Self::Root => {}
            Self::GetItem { child, .. }
            | Self::Select { child, .. }
            | Self::Not(child)
            | Self::IsNull(child)
            | Self::IsNotNull(child) => {
                children_to_drop.push(std::mem::replace(child, Self::drop_tombstone()));
            }
            Self::Pack { elements, .. } => {
                children_to_drop.extend(
                    std::mem::take(elements)
                        .into_iter()
                        .map(|(_name, expression)| expression),
                );
            }
            Self::Merge { expressions, .. } | Self::And(expressions) | Self::Or(expressions) => {
                children_to_drop.extend(std::mem::take(expressions));
            }
            Self::Binary { lhs, rhs, .. }
            | Self::Like {
                child: lhs,
                pattern: rhs,
                ..
            } => {
                children_to_drop.push(std::mem::replace(lhs, Self::drop_tombstone()));
                children_to_drop.push(std::mem::replace(rhs, Self::drop_tombstone()));
            }
            Self::Between {
                value,
                lower,
                upper,
                ..
            } => {
                children_to_drop.push(std::mem::replace(value, Self::drop_tombstone()));
                children_to_drop.push(std::mem::replace(lower, Self::drop_tombstone()));
                children_to_drop.push(std::mem::replace(upper, Self::drop_tombstone()));
            }
        }
    }

    fn drop_tombstone() -> Arc<Self> {
        Arc::new(Self::Root)
    }
}

fn into_raw(expr: NativeExpr) -> jlong {
    Box::into_raw(Box::new(expr)) as jlong
}

/// SAFETY: pointer must originate from [`into_raw`] and not yet be freed.
unsafe fn expr_ref<'a>(ptr: jlong) -> &'a NativeExpr {
    debug_assert!(ptr != 0, "null expression pointer");
    unsafe { &*(ptr as *const NativeExpr) }
}

/// Bind an opaque Java expression pointer to the scan input dtype.
///
/// # Safety
///
/// `ptr` must originate from [`into_raw`] and not yet be freed.
pub(crate) unsafe fn bind_expr_ptr(ptr: jlong, scope: &DType) -> VortexResult<BoundExpr> {
    unsafe { expr_ref(ptr) }.bind(scope)
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
    drop(unsafe { Box::from_raw(pointer as *mut NativeExpr) });
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_root(
    _env: EnvUnowned,
    _class: JClass,
) -> jlong {
    into_raw(NativeExpr::Root)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_rowIdx(
    _env: EnvUnowned,
    _class: JClass,
) -> jlong {
    into_raw(NativeExpr::Bound(row_idx()))
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
        Ok(into_raw(NativeExpr::GetItem {
            field,
            child: Arc::new(child),
        }))
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
        Ok(into_raw(NativeExpr::Select {
            fields: fields.into(),
            child: Arc::new(child),
        }))
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

            elements.push((name, Arc::new(expr)));
        }

        Ok(into_raw(NativeExpr::Pack {
            elements,
            nullability: nullable.into(),
        }))
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
        Ok(into_raw(NativeExpr::Merge {
            expressions: exprs,
            duplicate_handling: handling,
        }))
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
        if exprs.is_empty() {
            return Err(vortex_err!("empty AND expression").into());
        }
        Ok(into_raw(NativeExpr::And(exprs)))
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
        if exprs.is_empty() {
            return Err(vortex_err!("empty OR expression").into());
        }
        Ok(into_raw(NativeExpr::Or(exprs)))
    })
}

fn collect_operands(
    env: &mut jni::Env,
    operands: &JLongArray,
) -> Result<Vec<Arc<NativeExpr>>, JNIError> {
    let ptrs = unsafe { operands.get_elements(env, ReleaseMode::NoCopyBack) }?;
    Ok(ptrs
        .iter()
        .map(|ptr| Arc::new(unsafe { expr_ref(*ptr) }.clone()))
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
        Ok(into_raw(NativeExpr::Binary {
            operator,
            lhs: Arc::new(lhs),
            rhs: Arc::new(rhs),
        }))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_not(
    _env: EnvUnowned,
    _class: JClass,
    child: jlong,
) -> jlong {
    let child = unsafe { expr_ref(child) }.clone();
    into_raw(NativeExpr::Not(Arc::new(child)))
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_isNull(
    _env: EnvUnowned,
    _class: JClass,
    child: jlong,
) -> jlong {
    let child = unsafe { expr_ref(child) }.clone();
    into_raw(NativeExpr::IsNull(Arc::new(child)))
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeExpression_isNotNull(
    _env: EnvUnowned,
    _class: JClass,
    child: jlong,
) -> jlong {
    let child = unsafe { expr_ref(child) }.clone();
    into_raw(NativeExpr::IsNotNull(Arc::new(child)))
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
    into_raw(NativeExpr::Like {
        child: Arc::new(child),
        pattern: Arc::new(pattern),
        options: LikeOptions {
            negated,
            case_insensitive,
        },
    })
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
        Ok(into_raw(NativeExpr::Between {
            value: Arc::new(value),
            lower: Arc::new(lower),
            upper: Arc::new(upper),
            options: BetweenOptions {
                lower_strict: strict_from_bool(lower_strict),
                upper_strict: strict_from_bool(upper_strict),
            },
        }))
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
        return into_raw(NativeExpr::Bound(lit(scalar)));
    }
    into_raw(NativeExpr::Bound(lit(value)))
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
                return into_raw(NativeExpr::Bound(lit(scalar)));
            }
            into_raw(NativeExpr::Bound(lit(value as $rust)))
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
            return Ok(into_raw(NativeExpr::Bound(lit(scalar))));
        }
        let s: String = value.try_to_string(env)?;
        Ok(into_raw(NativeExpr::Bound(lit(s))))
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
            return Ok(into_raw(NativeExpr::Bound(lit(scalar))));
        }
        let bytes: Vec<u8> = env.convert_byte_array(&value)?;
        Ok(into_raw(NativeExpr::Bound(lit(bytes.as_slice()))))
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
            return Ok(into_raw(NativeExpr::Bound(lit(Scalar::null(
                DType::Decimal(decimal_dtype, Nullability::Nullable),
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
        Ok(into_raw(NativeExpr::Bound(lit(scalar))))
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
            return Ok(into_raw(NativeExpr::Bound(lit(Scalar::null(dtype)))));
        }
        let storage_value = match unit {
            TimeUnit::Days => ScalarValue::from(
                i32::try_from(value)
                    .map_err(|_| vortex_err!("date value does not fit in i32 days: {value}"))?,
            ),
            TimeUnit::Milliseconds => ScalarValue::from(value),
            other => throw_runtime!("date does not support time unit {other}"),
        };
        Ok(into_raw(NativeExpr::Bound(lit(Scalar::try_new(
            dtype,
            Some(storage_value),
        )?))))
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
            return Ok(into_raw(NativeExpr::Bound(lit(Scalar::null(dtype)))));
        }
        Ok(into_raw(NativeExpr::Bound(lit(Scalar::try_new(
            dtype,
            Some(ScalarValue::from(value)),
        )?))))
    })
}

/// Number of bytes in a UUID's big-endian representation.
const UUID_BYTE_LEN: usize = 16;

/// Build the version-agnostic UUID extension [`DType`] with the given nullability.
///
/// The storage is a non-nullable `FixedSizeList(U8, 16)`, matching Vortex's UUID extension and
/// Arrow's canonical UUID type. The metadata records no version constraint, so the dtype is
/// compatible with any UUID column regardless of the UUID versions it contains.
fn uuid_dtype(nullability: Nullability) -> Result<DType, JNIError> {
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
fn uuid_scalar(bytes: &[u8]) -> Result<Scalar, JNIError> {
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
            return Ok(into_raw(NativeExpr::Bound(lit(Scalar::null(uuid_dtype(
                Nullability::Nullable,
            )?)))));
        }
        if value.is_null() {
            throw_runtime!("UUID literal bytes must not be null");
        }
        let bytes = env.convert_byte_array(&value)?;
        Ok(into_raw(NativeExpr::Bound(lit(uuid_scalar(&bytes)?))))
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
        Ok(into_raw(NativeExpr::Bound(lit(Scalar::null(dtype)))))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn i64_dtype(nullability: Nullability) -> DType {
        DType::Primitive(PType::I64, nullability)
    }

    fn scope_dtype() -> DType {
        DType::struct_(
            [
                ("i", i64_dtype(Nullability::NonNullable)),
                ("j", i64_dtype(Nullability::Nullable)),
                ("name", DType::Utf8(Nullability::NonNullable)),
                ("flag", DType::Bool(Nullability::NonNullable)),
            ],
            Nullability::NonNullable,
        )
    }

    fn root() -> Arc<NativeExpr> {
        Arc::new(NativeExpr::Root)
    }

    fn column(name: &str) -> Arc<NativeExpr> {
        Arc::new(NativeExpr::GetItem {
            field: name.into(),
            child: root(),
        })
    }

    fn literal_i64(value: i64) -> Arc<NativeExpr> {
        Arc::new(NativeExpr::Bound(lit(value)))
    }

    #[test]
    fn bind_binary() -> VortexResult<()> {
        let expr = NativeExpr::Binary {
            operator: Operator::Gt,
            lhs: column("i"),
            rhs: literal_i64(1),
        };

        assert_eq!(
            expr.bind(&scope_dtype())?.dtype(),
            &DType::Bool(Nullability::NonNullable)
        );
        Ok(())
    }

    #[test]
    fn bind_pack() -> VortexResult<()> {
        let expr = NativeExpr::Pack {
            elements: vec![
                ("packed_i".into(), column("i")),
                ("packed_flag".into(), column("flag")),
            ],
            nullability: Nullability::NonNullable,
        };

        let expected = DType::struct_(
            [
                ("packed_i", i64_dtype(Nullability::NonNullable)),
                ("packed_flag", DType::Bool(Nullability::NonNullable)),
            ],
            Nullability::NonNullable,
        );
        assert_eq!(expr.bind(&scope_dtype())?.dtype(), &expected);
        Ok(())
    }

    #[test]
    fn bind_merge() -> VortexResult<()> {
        let expr = NativeExpr::Merge {
            expressions: vec![
                Arc::new(NativeExpr::Select {
                    fields: vec!["i"].into(),
                    child: root(),
                }),
                Arc::new(NativeExpr::Select {
                    fields: vec!["flag"].into(),
                    child: root(),
                }),
            ],
            duplicate_handling: DuplicateHandling::Error,
        };

        let expected = DType::struct_(
            [
                ("i", i64_dtype(Nullability::NonNullable)),
                ("flag", DType::Bool(Nullability::NonNullable)),
            ],
            Nullability::NonNullable,
        );
        assert_eq!(expr.bind(&scope_dtype())?.dtype(), &expected);
        Ok(())
    }

    #[test]
    fn bind_like() -> VortexResult<()> {
        let expr = NativeExpr::Like {
            child: column("name"),
            pattern: Arc::new(NativeExpr::Bound(lit("a%".to_string()))),
            options: LikeOptions {
                negated: false,
                case_insensitive: false,
            },
        };

        assert_eq!(
            expr.bind(&scope_dtype())?.dtype(),
            &DType::Bool(Nullability::NonNullable)
        );
        Ok(())
    }

    #[test]
    fn bind_between() -> VortexResult<()> {
        let expr = NativeExpr::Between {
            value: column("i"),
            lower: literal_i64(0),
            upper: literal_i64(10),
            options: BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        };

        assert_eq!(
            expr.bind(&scope_dtype())?.dtype(),
            &DType::Bool(Nullability::NonNullable)
        );
        Ok(())
    }

    #[test]
    fn bind_empty_and_errors() {
        let err = NativeExpr::And(vec![]).bind(&scope_dtype()).unwrap_err();
        assert!(err.to_string().contains("empty AND expression"));
    }

    #[test]
    fn bind_empty_or_errors() {
        let err = NativeExpr::Or(vec![]).bind(&scope_dtype()).unwrap_err();
        assert!(err.to_string().contains("empty OR expression"));
    }

    #[test]
    fn bind_missing_field_errors() {
        let expr = NativeExpr::GetItem {
            field: "missing".into(),
            child: root(),
        };

        assert!(expr.bind(&scope_dtype()).is_err());
    }
}
