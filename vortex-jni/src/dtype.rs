// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use jni::JNIEnv;
use jni::objects::JClass;
use jni::objects::JLongArray;
use jni::objects::JObject;
use jni::objects::JObjectArray;
use jni::objects::JString;
use jni::objects::JValue;
use jni::sys::JNI_FALSE;
use jni::sys::JNI_TRUE;
use jni::sys::jboolean;
use jni::sys::jbyte;
use jni::sys::jint;
use jni::sys::jlong;
use jni::sys::jobject;
use jni::sys::jstring;
use vortex::dtype::DType;
use vortex::dtype::DecimalDType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::dtype::StructFields;
use vortex::error::vortex_err;
use vortex::extension::datetime::AnyTemporal;
use vortex::extension::datetime::Date;
use vortex::extension::datetime::Time;
use vortex::extension::datetime::TimeUnit;
use vortex::extension::datetime::Timestamp;

use crate::errors::JNIError;
use crate::errors::try_or_throw;

pub const DTYPE_NULL: jbyte = 0;
pub const DTYPE_BOOL: jbyte = 1;
pub const DTYPE_PRIMITIVE_U8: jbyte = 2;
pub const DTYPE_PRIMITIVE_U16: jbyte = 3;
pub const DTYPE_PRIMITIVE_U32: jbyte = 4;
pub const DTYPE_PRIMITIVE_U64: jbyte = 5;
pub const DTYPE_PRIMITIVE_I8: jbyte = 6;
pub const DTYPE_PRIMITIVE_I16: jbyte = 7;
pub const DTYPE_PRIMITIVE_I32: jbyte = 8;
pub const DTYPE_PRIMITIVE_I64: jbyte = 9;
pub const DTYPE_PRIMITIVE_F16: jbyte = 10;
pub const DTYPE_PRIMITIVE_F32: jbyte = 11;
pub const DTYPE_PRIMITIVE_F64: jbyte = 12;
pub const DTYPE_UTF8: jbyte = 13;
pub const DTYPE_BINARY: jbyte = 14;
pub const DTYPE_STRUCT: jbyte = 15;
pub const DTYPE_LIST: jbyte = 16;
pub const DTYPE_EXTENSION: jbyte = 17;
pub const DTYPE_DECIMAL: jbyte = 18;
pub const DTYPE_FIXED_SIZE_LIST: jbyte = 19;

static LONG_CLASS: &str = "java/lang/Long";

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_free(
    _env: JNIEnv,
    _class: JClass,
    dtype_ptr: jlong,
) {
    // SAFETY: caller must ensure that the pointer is valid and points to a `DType`.
    drop(unsafe { Box::from_raw(dtype_ptr as *mut DType) });
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_getVariant(
    _env: JNIEnv,
    _class: JClass,
    dtype_ptr: jlong,
) -> jbyte {
    // SAFETY: caller must ensure that the pointer is valid and points to a `DType`.
    let dtype = unsafe { &*(dtype_ptr as *const DType) };
    match dtype {
        DType::Null => DTYPE_NULL,
        DType::Bool(_) => DTYPE_BOOL,
        DType::Primitive(ptype, _) => match ptype {
            PType::U8 => DTYPE_PRIMITIVE_U8,
            PType::U16 => DTYPE_PRIMITIVE_U16,
            PType::U32 => DTYPE_PRIMITIVE_U32,
            PType::U64 => DTYPE_PRIMITIVE_U64,
            PType::I8 => DTYPE_PRIMITIVE_I8,
            PType::I16 => DTYPE_PRIMITIVE_I16,
            PType::I32 => DTYPE_PRIMITIVE_I32,
            PType::I64 => DTYPE_PRIMITIVE_I64,
            PType::F16 => DTYPE_PRIMITIVE_F16,
            PType::F32 => DTYPE_PRIMITIVE_F32,
            PType::F64 => DTYPE_PRIMITIVE_F64,
        },
        DType::Decimal(..) => DTYPE_DECIMAL,
        DType::Utf8(_) => DTYPE_UTF8,
        DType::Binary(_) => DTYPE_BINARY,
        DType::Struct(..) => DTYPE_STRUCT,
        DType::List(..) => DTYPE_LIST,
        DType::FixedSizeList(..) => DTYPE_FIXED_SIZE_LIST,
        DType::Extension(_) => DTYPE_EXTENSION,
        DType::Variant(_) => unimplemented!("Variant DType is not supported in JNI yet"),
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_isNullable(
    _env: JNIEnv,
    _class: JClass,
    dtype_ptr: jlong,
) -> jboolean {
    // SAFETY: caller must ensure that the pointer is valid and points to a `DType`.
    let dtype = unsafe { &*(dtype_ptr as *const DType) };
    if dtype.is_nullable() {
        JNI_TRUE
    } else {
        JNI_FALSE
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_getFieldNames(
    mut env: JNIEnv,
    _class: JClass,
    dtype_ptr: jlong,
) -> jobject {
    let dtype = unsafe { &*(dtype_ptr as *const DType) };

    try_or_throw(&mut env, |env| {
        let array_list = env.new_object("java/util/ArrayList", "()V", &[])?;
        let field_names = env.get_list(&array_list)?;
        let Some(struct_dtype) = dtype.as_struct_fields_opt() else {
            throw_runtime!("DType should be STRUCT, was {dtype}");
        };

        for name in struct_dtype.names().iter() {
            let field = env.new_string(name)?;
            field_names.add(env, field.as_ref())?;
        }

        Ok::<jobject, JNIError>(array_list.into_raw())
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_getFieldTypes(
    mut env: JNIEnv,
    _class: JClass,
    dtype_ptr: jlong,
) -> jobject {
    let dtype = unsafe { &*(dtype_ptr as *const DType) };

    try_or_throw(&mut env, |env| {
        let array_list = env
            .new_object("java/util/ArrayList", "()V", &[])
            .map_err(|e| JNIError::Vortex(vortex_err!("failure constructing ArrayList: {e}")))?;
        let field_types = env.get_list(&array_list)?;
        let Some(struct_dtype) = dtype.as_struct_fields_opt() else {
            throw_runtime!("DType should be STRUCT, was {dtype}");
        };

        for field_dtype in struct_dtype.fields() {
            let ptr: *mut DType = Box::into_raw(Box::new(field_dtype));
            let boxed = env
                .call_static_method(
                    LONG_CLASS,
                    "valueOf",
                    "(J)Ljava/lang/Long;",
                    &[JValue::Long(ptr.addr() as jlong)],
                )?
                .l()?;
            field_types.add(env, &boxed)?;
        }

        Ok(array_list.into_raw())
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_getElementType(
    mut env: JNIEnv,
    _class: JClass,
    dtype_ptr: jlong,
) -> jlong {
    let dtype = unsafe { &*(dtype_ptr as *const DType) };

    try_or_throw(&mut env, |_| {
        let element_type = dtype
            .as_list_element_opt()
            .or_else(|| dtype.as_fixed_size_list_element_opt());
        let Some(element_type) = element_type else {
            throw_runtime!("DType should be LIST or FIXED_SIZE_LIST, was {dtype}");
        };

        Ok(element_type.as_ref() as *const DType as jlong)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_isDate(
    mut env: JNIEnv,
    _class: JClass,
    dtype_ptr: jlong,
) -> jboolean {
    let dtype = unsafe { &*(dtype_ptr as *const DType) };

    try_or_throw(&mut env, |_| {
        let DType::Extension(ext_dtype) = dtype else {
            throw_runtime!("DType should be an EXTENSION, was {dtype}");
        };

        if ext_dtype.is::<Date>() {
            Ok(JNI_TRUE)
        } else {
            Ok(JNI_FALSE)
        }
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_isTime(
    mut env: JNIEnv,
    _class: JClass,
    dtype_ptr: jlong,
) -> jboolean {
    let dtype = unsafe { &*(dtype_ptr as *const DType) };

    try_or_throw(&mut env, |_| {
        let DType::Extension(ext_dtype) = dtype else {
            throw_runtime!("DType should be an EXTENSION, was {dtype}");
        };

        if ext_dtype.is::<Time>() {
            Ok(JNI_TRUE)
        } else {
            Ok(JNI_FALSE)
        }
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_isTimestamp(
    mut env: JNIEnv,
    _class: JClass,
    dtype_ptr: jlong,
) -> jboolean {
    let dtype = unsafe { &*(dtype_ptr as *const DType) };

    try_or_throw(&mut env, |_| {
        let DType::Extension(ext_dtype) = dtype else {
            throw_runtime!("DType should be an EXTENSION, was {dtype}");
        };
        if ext_dtype.is::<Timestamp>() {
            Ok(JNI_TRUE)
        } else {
            Ok(JNI_FALSE)
        }
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_getTimeUnit(
    mut env: JNIEnv,
    _class: JClass,
    dtype_ptr: jlong,
) -> jbyte {
    let dtype = unsafe { &*(dtype_ptr as *const DType) };

    try_or_throw(&mut env, |_| {
        let DType::Extension(ext_dtype) = dtype else {
            throw_runtime!("DType should be an EXTENSION, was {dtype}");
        };

        let Some(opts) = ext_dtype.metadata_opt::<AnyTemporal>() else {
            throw_runtime!("DType should be a temporal type, was {dtype}");
        };

        Ok(match opts.time_unit() {
            TimeUnit::Nanoseconds => 0,
            TimeUnit::Microseconds => 1,
            TimeUnit::Milliseconds => 2,
            TimeUnit::Seconds => 3,
            TimeUnit::Days => 4,
        })
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_getTimeZone(
    mut env: JNIEnv,
    _class: JClass,
    dtype_ptr: jlong,
) -> jstring {
    let dtype = unsafe { &*(dtype_ptr as *const DType) };

    try_or_throw(&mut env, |env| {
        let DType::Extension(ext_dtype) = dtype else {
            throw_runtime!("DType should be an EXTENSION, was {dtype}");
        };

        let Some(opts) = ext_dtype.metadata_opt::<Timestamp>() else {
            throw_runtime!("DType should be a TIMESTAMP, was {dtype}");
        };

        if let Some(time_zone) = opts.tz.as_ref() {
            Ok(env.new_string(time_zone)?.into_raw())
        } else {
            Ok(JObject::null().into_raw())
        }
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_isDecimal(
    _env: JNIEnv,
    _class: JClass,
    dtype_ptr: jlong,
) -> jboolean {
    let dtype = unsafe { &*(dtype_ptr as *const DType) };
    match dtype {
        DType::Decimal(..) => JNI_TRUE,
        _ => JNI_FALSE,
    }
}

// Decimal-related access methods
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_getDecimalPrecision(
    mut env: JNIEnv,
    _class: JClass,
    dtype_ptr: jlong,
) -> jint {
    let dtype = unsafe { &*(dtype_ptr as *const DType) };
    try_or_throw(&mut env, |_| {
        let DType::Decimal(decimal_dtype, ..) = dtype else {
            throw_runtime!("DType should be a DECIMAL, was {dtype}");
        };

        Ok(decimal_dtype.precision() as jint)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_getDecimalScale(
    mut env: JNIEnv,
    _class: JClass,
    dtype_ptr: jlong,
) -> jbyte {
    let dtype = unsafe { &*(dtype_ptr as *const DType) };
    try_or_throw(&mut env, |_| {
        let DType::Decimal(decimal_dtype, ..) = dtype else {
            throw_runtime!("DType should a DECIMAL, was {dtype}");
        };

        Ok(decimal_dtype.scale())
    })
}

// Constructors
//
// NOTE: Java only supports signed types, as does Spark and Iceberg, so we don't bother with
//  constructors for unsigned types.

/// I8 constructor
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_newByte(
    _env: JNIEnv,
    _class: JClass,
    is_nullable: jboolean,
) -> jlong {
    Box::into_raw(Box::new(DType::Primitive(
        PType::I8,
        to_nullability(is_nullable),
    ))) as jlong
}

/// I16 constructor
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_newShort(
    _env: JNIEnv,
    _class: JClass,
    is_nullable: jboolean,
) -> jlong {
    Box::into_raw(Box::new(DType::Primitive(
        PType::I16,
        to_nullability(is_nullable),
    ))) as jlong
}

/// I32 constructor
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_newInt(
    _env: JNIEnv,
    _class: JClass,
    is_nullable: jboolean,
) -> jlong {
    Box::into_raw(Box::new(DType::Primitive(
        PType::I32,
        to_nullability(is_nullable),
    ))) as jlong
}

/// I32 constructor
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_newLong(
    _env: JNIEnv,
    _class: JClass,
    is_nullable: jboolean,
) -> jlong {
    Box::into_raw(Box::new(DType::Primitive(
        PType::I64,
        to_nullability(is_nullable),
    ))) as jlong
}

/// F32 constructor
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_newFloat(
    _env: JNIEnv,
    _class: JClass,
    is_nullable: jboolean,
) -> jlong {
    Box::into_raw(Box::new(DType::Primitive(
        PType::F32,
        to_nullability(is_nullable),
    ))) as jlong
}

/// F64 constructor
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_newDouble(
    _env: JNIEnv,
    _class: JClass,
    is_nullable: jboolean,
) -> jlong {
    Box::into_raw(Box::new(DType::Primitive(
        PType::F64,
        to_nullability(is_nullable),
    ))) as jlong
}

/// Decimal constructor
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_newDecimal(
    mut env: JNIEnv,
    _class: JClass,
    precision: jint,
    scale: jint,
    is_nullable: jboolean,
) -> jlong {
    try_or_throw(&mut env, |_| {
        let precision = u8::try_from(precision)
            .map_err(|_| vortex_err!("precision {precision} out of bounds for Decimal"))?;
        let scale = i8::try_from(scale)
            .map_err(|_| vortex_err!("scale {scale} out of bounds for Decimal"))?;
        let decimal_type = DecimalDType::try_new(precision, scale).map_err(|_| {
            vortex_err!("Invalid (precision, scale) for Vortex Decimal ({precision}, {scale})")
        })?;

        Ok(Box::into_raw(Box::new(DType::Decimal(
            decimal_type,
            to_nullability(is_nullable),
        ))) as jlong)
    })
}

/// UTF-8 constructor
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_newUtf8(
    _env: JNIEnv,
    _class: JClass,
    is_nullable: jboolean,
) -> jlong {
    Box::into_raw(Box::new(DType::Utf8(to_nullability(is_nullable)))) as jlong
}

/// Binary constructor
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_newBinary(
    _env: JNIEnv,
    _class: JClass,
    is_nullable: jboolean,
) -> jlong {
    Box::into_raw(Box::new(DType::Binary(to_nullability(is_nullable)))) as jlong
}

/// Bool constructor
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_newBool(
    _env: JNIEnv,
    _class: JClass,
    is_nullable: jboolean,
) -> jlong {
    Box::into_raw(Box::new(DType::Bool(to_nullability(is_nullable)))) as jlong
}

/// List constructor
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_newList(
    _env: JNIEnv,
    _class: JClass,
    element_ptr: jlong,
    is_nullable: jboolean,
) -> jlong {
    let element_dtype = unsafe { *Box::from_raw(element_ptr as *mut DType) };
    let element_dtype = Arc::new(element_dtype);

    let list_type = DType::List(element_dtype, to_nullability(is_nullable));

    Box::into_raw(Box::new(list_type)) as jlong
}

/// FixedSizeList constructor
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_newFixedSizeList(
    _env: JNIEnv,
    _class: JClass,
    element_ptr: jlong,
    size: jint,
    is_nullable: jboolean,
) -> jlong {
    let element_dtype = unsafe { *Box::from_raw(element_ptr as *mut DType) };
    let element_dtype = Arc::new(element_dtype);

    let fsl_type = DType::FixedSizeList(element_dtype, size as u32, to_nullability(is_nullable));

    Box::into_raw(Box::new(fsl_type)) as jlong
}

/// Get the fixed size of a FixedSizeList DType.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_getFixedSizeListSize(
    mut env: JNIEnv,
    _class: JClass,
    dtype_ptr: jlong,
) -> jint {
    let dtype = unsafe { &*(dtype_ptr as *const DType) };

    try_or_throw(&mut env, |_| {
        let DType::FixedSizeList(_, size, _) = dtype else {
            throw_runtime!("DType should be FIXED_SIZE_LIST, was {dtype}");
        };

        Ok(*size as jint)
    })
}

/// Struct constructor
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_newStruct<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass,
    field_names: JObjectArray<'local>,
    field_types: JLongArray<'local>,
    is_nullable: jboolean,
) -> jlong {
    try_or_throw(&mut env, |env| {
        let field_count = env.get_array_length(&field_names)?;
        let field_types_count = env.get_array_length(&field_types)?;

        if field_count != field_types_count {
            throw_runtime!("fieldNames.length ≠ fieldTypes.length")
        }

        let mut field_type_ptrs = vec![0; field_types_count as usize];
        env.get_long_array_region(&field_types, 0, &mut field_type_ptrs[..])?;

        let mut field_names_arc = Vec::with_capacity(field_count as usize);
        let mut dtypes = Vec::with_capacity(field_count as usize);

        for field_idx in 0..field_count {
            let field_name = JString::from(env.get_object_array_element(&field_names, field_idx)?);
            // SAFETY: in Java this is a String[] not an Object[], so the type is checked in Java
            //  at compile time.
            let field_name = unsafe { env.get_string_unchecked(&field_name)? };
            let field_name_str = field_name
                .to_str()
                .map_err(|_| vortex_err!("Invalid UTF-8 in field name"))?;
            field_names_arc.push(Arc::<str>::from(field_name_str));

            let field_type = field_type_ptrs[field_idx as usize];

            let dtype = *unsafe { Box::from_raw(field_type as *mut DType) };
            dtypes.push(dtype);
        }

        let fields = StructFields::new(field_names_arc.into(), dtypes);

        let struct_type = DType::Struct(fields, to_nullability(is_nullable));
        Ok(Box::into_raw(Box::new(struct_type)) as jlong)
    })
}

/// Timestamp constructor
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_newTimestamp<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass,
    time_unit: jbyte,
    zone: JString<'local>,
    is_nullable: jboolean,
) -> jlong {
    try_or_throw(&mut env, |env| {
        let time_unit = TimeUnit::try_from(time_unit as u8).map_err(JNIError::Vortex)?;

        let tz = if zone.is_null() {
            None
        } else {
            Some(
                env.get_string(&zone)?
                    .to_str()
                    .map_err(|_| JNIError::Vortex(vortex_err!("Invalid UTF-8 in zone")))?
                    .into(),
            )
        };

        let dtype = DType::Extension(
            Timestamp::new_with_tz(time_unit, tz, to_nullability(is_nullable)).erased(),
        );
        Ok(Box::into_raw(Box::new(dtype)) as jlong)
    })
}

/// Date constructor
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_newDate(
    mut env: JNIEnv,
    _class: JClass,
    time_unit: jbyte,
    is_nullable: jboolean,
) -> jlong {
    try_or_throw(&mut env, |_| {
        let time_unit = TimeUnit::try_from(time_unit as u8).map_err(JNIError::Vortex)?;
        let dtype = DType::Extension(Date::new(time_unit, to_nullability(is_nullable)).erased());
        Ok(Box::into_raw(Box::new(dtype)) as jlong)
    })
}

/// Time constructor
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_newTime(
    mut env: JNIEnv,
    _class: JClass,
    time_unit: jbyte,
    is_nullable: jboolean,
) -> jlong {
    try_or_throw(&mut env, |_| {
        let time_unit = TimeUnit::try_from(time_unit as u8).map_err(JNIError::Vortex)?;
        let dtype = DType::Extension(Time::new(time_unit, to_nullability(is_nullable)).erased());
        Ok(Box::into_raw(Box::new(dtype)) as jlong)
    })
}

fn to_nullability(is_nullable: jboolean) -> Nullability {
    if is_nullable == JNI_FALSE {
        Nullability::NonNullable
    } else {
        Nullability::Nullable
    }
}
