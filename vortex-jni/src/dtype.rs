use jni::JNIEnv;
use jni::objects::{JClass, JList, JObject, JString, JValue};
use jni::sys::{JNI_FALSE, JNI_TRUE, jboolean, jbyte, jlong};
use vortex::dtype::datetime::{DATE_ID, TIME_ID, TIMESTAMP_ID, TemporalMetadata, TimeUnit};
use vortex::dtype::{DType, PType};
use vortex::error::vortex_err;

use crate::errors::Throwable;

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
        DType::Utf8(_) => DTYPE_UTF8,
        DType::Binary(_) => DTYPE_BINARY,
        DType::Struct(..) => DTYPE_STRUCT,
        DType::List(..) => DTYPE_LIST,
        DType::Extension(_) => DTYPE_EXTENSION,
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
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_getFieldNames<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    dtype_ptr: jlong,
) -> JObject<'local> {
    let dtype = unsafe { &*(dtype_ptr as *const DType) };
    let array_list = env
        .new_object("java/util/ArrayList", "()V", &[])
        .expect("Failed to create ArrayList");
    let field_names = JList::from_env(&mut env, &array_list).expect("ArrayList as JList");
    let Some(struct_dtype) = dtype.as_struct() else {
        vortex_err!("DType should be STRUCT, was {dtype}").throw_illegal_argument(&mut env);
        return array_list;
    };

    struct_dtype.names().iter().for_each(|name| {
        let field = env.new_string(name).expect("create new string");
        field_names
            .add(&mut env, field.as_ref())
            .expect("JList::add");
    });

    array_list
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_getFieldTypes<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    dtype_ptr: jlong,
) -> JObject<'local> {
    let dtype = unsafe { &*(dtype_ptr as *const DType) };
    let array_list = env
        .new_object("java/util/ArrayList", "()V", &[])
        .expect("Failed to create ArrayList");
    let field_types = JList::from_env(&mut env, &array_list).expect("JList.from_env");
    let Some(struct_dtype) = dtype.as_struct() else {
        vortex_err!("DType should be STRUCT, was {dtype}").throw_illegal_argument(&mut env);
        return array_list;
    };

    struct_dtype.fields().for_each(|field_dtype| {
        let ptr: *mut DType = Box::into_raw(Box::new(field_dtype));
        let boxed = env
            .call_static_method(
                LONG_CLASS,
                "valueOf",
                "(J)Ljava/lang/Long;",
                &[JValue::Long(ptr.addr() as jlong)],
            )
            .expect("Long.valueOf")
            .l()
            .expect("Long.valueOf should return an Object");
        field_types.add(&mut env, &boxed).expect("JList::add");
    });

    array_list
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_getElementType(
    mut env: JNIEnv,
    _class: JClass,
    dtype_ptr: jlong,
) -> jlong {
    let dtype = unsafe { &*(dtype_ptr as *const DType) };
    let Some(element_type) = dtype.as_list_element() else {
        vortex_err!("DType should be LIST, was {dtype}").throw_illegal_argument(&mut env);
        return 0;
    };

    element_type as *const DType as jlong
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_isDate(
    mut env: JNIEnv,
    _class: JClass,
    dtype_ptr: jlong,
) -> jboolean {
    let dtype = unsafe { &*(dtype_ptr as *const DType) };

    let DType::Extension(ext_dtype) = dtype else {
        vortex_err!("DType should be an EXTENSION, was {dtype}").throw_illegal_argument(&mut env);
        return JNI_FALSE;
    };

    if ext_dtype.id().as_ref() == DATE_ID.as_ref() {
        JNI_TRUE
    } else {
        JNI_FALSE
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_isTime(
    mut env: JNIEnv,
    _class: JClass,
    dtype_ptr: jlong,
) -> jboolean {
    let dtype = unsafe { &*(dtype_ptr as *const DType) };

    let DType::Extension(ext_dtype) = dtype else {
        vortex_err!("DType should be an EXTENSION, was {dtype}").throw_illegal_argument(&mut env);
        return JNI_FALSE;
    };

    if ext_dtype.id().as_ref() == TIME_ID.as_ref() {
        JNI_TRUE
    } else {
        JNI_FALSE
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_isTimestamp(
    mut env: JNIEnv,
    _class: JClass,
    dtype_ptr: jlong,
) -> jboolean {
    let dtype = unsafe { &*(dtype_ptr as *const DType) };

    let DType::Extension(ext_dtype) = dtype else {
        vortex_err!("DType should be an EXTENSION, was {dtype}").throw_illegal_argument(&mut env);
        return JNI_FALSE;
    };

    if ext_dtype.id().as_ref() == TIMESTAMP_ID.as_ref() {
        JNI_TRUE
    } else {
        JNI_FALSE
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_getTimeUnit(
    mut env: JNIEnv,
    _class: JClass,
    dtype_ptr: jlong,
) -> jbyte {
    let dtype = unsafe { &*(dtype_ptr as *const DType) };

    let DType::Extension(ext_dtype) = dtype else {
        vortex_err!("DType should be an EXTENSION, was {dtype}").throw_illegal_argument(&mut env);
        return -1;
    };

    match TemporalMetadata::try_from(ext_dtype) {
        Ok(temporal) => match temporal.time_unit() {
            TimeUnit::Ns => 0,
            TimeUnit::Us => 1,
            TimeUnit::Ms => 2,
            TimeUnit::S => 3,
            TimeUnit::D => 4,
        },
        Err(err) => {
            err.throw_illegal_argument(&mut env);
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeDTypeMethods_getTimeZone<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    dtype_ptr: jlong,
) -> JString<'local> {
    let dtype = unsafe { &*(dtype_ptr as *const DType) };

    let DType::Extension(ext_dtype) = dtype else {
        vortex_err!("DType should be an EXTENSION, was {dtype}").throw_illegal_argument(&mut env);
        return JObject::null().into();
    };

    if ext_dtype.id().as_ref() != TIMESTAMP_ID.as_ref() {
        vortex_err!("DType should be a TIMESTAMP, was {dtype}").throw_illegal_argument(&mut env);
        return JObject::null().into();
    }

    match TemporalMetadata::try_from(ext_dtype) {
        Ok(temporal) => {
            if let Some(time_zone) = temporal.time_zone() {
                env.new_string(time_zone).expect("Failed to create JString")
            } else {
                JObject::null().into()
            }
        }
        Err(err) => {
            err.throw_illegal_argument(&mut env);
            JObject::null().into()
        }
    }
}
