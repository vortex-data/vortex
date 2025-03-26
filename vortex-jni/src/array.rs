use jni::JNIEnv;
use jni::objects::{JByteArray, JClass, JIntArray, JLongArray, JObject, JString};
use jni::sys::{JNI_FALSE, JNI_TRUE, jboolean, jbyte, jdouble, jfloat, jint, jlong, jshort};
use vortex::arrays::{VarBinArray, VarBinViewArray};
use vortex::compute::{scalar_at, slice};
use vortex::dtype::DType;
use vortex::error::{VortexExpect, VortexResult, vortex_err};
use vortex::{Array, ArrayRef, ArrayVariants};

use crate::errors::Throwable;

pub struct NativeArray {
    inner: ArrayRef,
    is_extension: bool,
}

impl NativeArray {
    pub fn new(array_ref: ArrayRef) -> Box<Self> {
        Box::new(NativeArray {
            is_extension: array_ref.dtype().is_extension(),
            inner: array_ref,
        })
    }

    pub fn into_raw(self: Box<Self>) -> jlong {
        Box::into_raw(self) as jlong
    }

    /// Reconstruct a boxed `NativeArray` from a raw heap pointer.
    pub unsafe fn from_raw(pointer: jlong) -> Box<Self> {
        // SAFETY: caller must ensure that the pointer is valid and points to a `NativeArray`.
        unsafe { Box::from_raw(pointer as *mut NativeArray) }
    }

    pub unsafe fn from_ptr<'a>(pointer: jlong) -> &'a Self {
        unsafe {
            (pointer as *const NativeArray)
                .as_ref()
                .expect("Pointer should never be null")
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_free(
    _env: JNIEnv,
    _class: JClass,
    array_ptr: jlong,
) {
    drop(unsafe { NativeArray::from_raw(array_ptr) });
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_getLen(
    _env: JNIEnv,
    _class: JClass,
    array_ptr: jlong,
) -> jlong {
    unsafe { NativeArray::from_ptr(array_ptr) }.inner.len() as jlong
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_getDataType(
    _env: JNIEnv,
    _class: JClass,
    array_ptr: jlong,
) -> jlong {
    let array_ref = unsafe { NativeArray::from_ptr(array_ptr) };
    let dtype_ptr = array_ref.inner.dtype();
    // Return a pointer to the DType.
    (dtype_ptr as *const DType).addr() as jlong
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_getField(
    mut env: JNIEnv,
    _class: JClass,
    array_ptr: jlong,
    index: jint,
) -> jlong {
    let array_ref = unsafe { NativeArray::from_ptr(array_ptr) };
    let Some(struct_array) = array_ref.inner.as_struct_typed() else {
        vortex_err!("getField expected struct array").throw_illegal_argument(&mut env);
        return -1;
    };

    match struct_array.maybe_null_field_by_idx(index as usize) {
        Ok(field) => NativeArray::new(field).into_raw(),
        Err(err) => {
            err.throw_illegal_argument(&mut env);
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_slice(
    mut env: JNIEnv,
    _class: JClass,
    array_ptr: jlong,
    start: jint,
    end: jint,
) -> jlong {
    let array_ref = unsafe { NativeArray::from_ptr(array_ptr) };
    // Create a new sliced copy of this array.
    match slice(array_ref.inner.as_ref(), start as usize, end as usize) {
        Ok(sliced_array) => NativeArray::new(sliced_array).into_raw(),
        Err(err) => {
            err.throw_illegal_argument(&mut env);
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_getNull(
    mut env: JNIEnv,
    _class: JClass,
    array_ptr: jlong,
    index: jint,
) -> jboolean {
    let array_ref = unsafe { NativeArray::from_ptr(array_ptr) };
    match array_ref.inner.is_invalid(index as usize) {
        Ok(is_null) => {
            if is_null {
                JNI_TRUE
            } else {
                JNI_FALSE
            }
        }
        Err(err) => {
            err.throw_illegal_argument(&mut env);
            JNI_FALSE
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_getNullCount(
    mut env: JNIEnv,
    _class: JClass,
    array_ptr: jlong,
) -> jint {
    let array_ref = unsafe { NativeArray::from_ptr(array_ptr) };
    match array_ref.inner.invalid_count() {
        Ok(count) => jint::try_from(count).unwrap_or(-1),
        Err(err) => {
            err.throw_illegal_argument(&mut env);
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_getByte(
    mut env: JNIEnv,
    _class: JClass,
    array_ptr: jlong,
    index: jint,
) -> jbyte {
    let array_ref = unsafe { NativeArray::from_ptr(array_ptr) };
    match scalar_at(array_ref.inner.as_ref(), index as usize) {
        Ok(value) => match value.as_primitive().as_::<i8>() {
            Ok(None) => 0,
            Ok(Some(b)) => b,
            Err(err) => {
                err.throw_illegal_argument(&mut env);
                -1
            }
        },
        Err(err) => {
            err.throw_illegal_argument(&mut env);
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_getShort(
    mut env: JNIEnv,
    _class: JClass,
    array_ptr: jlong,
    index: jint,
) -> jshort {
    let array_ref = unsafe { NativeArray::from_ptr(array_ptr) };
    match scalar_at(array_ref.inner.as_ref(), index as usize) {
        Ok(value) => match value.as_primitive().as_::<i16>() {
            Ok(None) => 0,
            Ok(Some(s)) => s,
            Err(err) => {
                err.throw_illegal_argument(&mut env);
                -1
            }
        },
        Err(err) => {
            err.throw_illegal_argument(&mut env);
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_getInt(
    mut env: JNIEnv,
    _class: JClass,
    array_ptr: jlong,
    index: jint,
) -> jint {
    let array_ref = unsafe { NativeArray::from_ptr(array_ptr) };
    match scalar_at(array_ref.inner.as_ref(), index as usize) {
        Ok(value) => {
            let value = if array_ref.is_extension {
                value.as_extension().storage()
            } else {
                value
            };
            match value.as_primitive().as_::<i32>() {
                Ok(None) => 0,
                Ok(Some(i)) => i,
                Err(err) => {
                    err.throw_illegal_argument(&mut env);
                    -1
                }
            }
        }
        Err(err) => {
            err.throw_illegal_argument(&mut env);
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_getLong(
    mut env: JNIEnv,
    _class: JClass,
    array_ptr: jlong,
    index: jint,
) -> jlong {
    let array_ref = unsafe { NativeArray::from_ptr(array_ptr) };
    match scalar_at(array_ref.inner.as_ref(), index as usize) {
        Ok(value) => {
            let value = if array_ref.is_extension {
                value.as_extension().storage()
            } else {
                value
            };
            match value.as_primitive().as_::<i64>() {
                Ok(None) => 0,
                Ok(Some(val)) => val,
                Err(err) => {
                    err.throw_illegal_argument(&mut env);
                    -1
                }
            }
        }
        Err(err) => {
            err.throw_illegal_argument(&mut env);
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_getBool(
    mut env: JNIEnv,
    _class: JClass,
    array_ptr: jlong,
    index: jint,
) -> jboolean {
    let array_ref = unsafe { NativeArray::from_ptr(array_ptr) };
    match scalar_at(array_ref.inner.as_ref(), index as usize) {
        Ok(value) => match value.as_bool().value() {
            None => JNI_FALSE,
            Some(b) => {
                if b {
                    JNI_TRUE
                } else {
                    JNI_FALSE
                }
            }
        },
        Err(err) => {
            err.throw_illegal_argument(&mut env);
            JNI_FALSE
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_getFloat(
    mut env: JNIEnv,
    _class: JClass,
    array_ptr: jlong,
    index: jint,
) -> jfloat {
    let array_ref = unsafe { NativeArray::from_ptr(array_ptr) };
    match scalar_at(array_ref.inner.as_ref(), index as usize) {
        Ok(value) => value.as_primitive().typed_value::<f32>().unwrap_or(0.0),
        Err(err) => {
            err.throw_illegal_argument(&mut env);
            0.0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_getDouble(
    mut env: JNIEnv,
    _class: JClass,
    array_ptr: jlong,
    index: jint,
) -> jdouble {
    let array_ref = unsafe { NativeArray::from_ptr(array_ptr) };
    match scalar_at(array_ref.inner.as_ref(), index as usize) {
        Ok(value) => value.as_primitive().typed_value::<f64>().unwrap_or(0.0),
        Err(err) => {
            err.throw_illegal_argument(&mut env);
            0.0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_getUTF8<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    array_ptr: jlong,
    index: jint,
) -> JString<'local> {
    let array_ref = unsafe { NativeArray::from_ptr(array_ptr) };
    match scalar_at(array_ref.inner.as_ref(), index as usize) {
        Ok(value) => match value.as_utf8().value() {
            None => JObject::null().into(),
            Some(buf_str) => env.new_string(buf_str.as_str()).expect("create new string"),
        },
        Err(err) => {
            err.throw_illegal_argument(&mut env);
            JObject::null().into()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_getUTF8_1ptr_1len<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    array_ptr: jlong,
    index: jint,
    out_ptr: JLongArray<'local>,
    out_len: JIntArray<'local>,
) {
    let array_ref = unsafe { NativeArray::from_ptr(array_ptr) };
    if array_ref.inner.as_utf8_typed().is_none() {
        vortex_err!("getUTF8_ptr_len expected UTF8 array").throw_illegal_argument(&mut env);
        return;
    }

    if let Some(varbin) = array_ref.inner.as_any().downcast_ref::<VarBinArray>() {
        match get_ptr_len_varbin(index, varbin) {
            Ok((ptr, len)) => {
                env.set_long_array_region(&out_ptr, 0, &[ptr as jlong])
                    .expect("set_long_array_region");
                env.set_int_array_region(&out_len, 0, &[len as jint])
                    .expect("set_int_array_region");
            }
            Err(err) => {
                err.throw_runtime(&mut env, "get_ptr_len_varbin");
            }
        }
    } else if let Some(varbinview) = array_ref.inner.as_any().downcast_ref::<VarBinViewArray>() {
        let (ptr, len) = get_ptr_len_view(index, varbinview);
        env.set_long_array_region(&out_ptr, 0, &[ptr as jlong])
            .expect("set_long_array_region");
        env.set_int_array_region(&out_len, 0, &[len as jint])
            .expect("set_int_array_region");
    } else {
        vortex_err!("getUTF8_ptr_len expected VarBin or VarBinView")
            .throw_illegal_argument(&mut env);
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_getBinary<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    array_ptr: jlong,
    index: jint,
) -> JByteArray<'local> {
    let array_ref = unsafe { NativeArray::from_ptr(array_ptr) };
    match scalar_at(array_ref.inner.as_ref(), index as usize) {
        Ok(value) => match value.as_binary().value() {
            None => JObject::null().into(),
            Some(buf) => env
                .byte_array_from_slice(buf.as_slice())
                .expect("create new byte array"),
        },
        Err(err) => {
            err.throw_illegal_argument(&mut env);
            JObject::null().into()
        }
    }
}

/// Get a raw pointer + len to pass back to Java to avoid copying across the boundary.
fn get_ptr_len_varbin(index: jint, array: &VarBinArray) -> VortexResult<(*const u8, u32)> {
    let bytes = array.bytes_at(usize::try_from(index).vortex_expect("index must fit in usize"))?;
    Ok((
        bytes.as_ptr(),
        u32::try_from(bytes.len()).vortex_expect("string length must fit in u32"),
    ))
}

/// Get a raw pointer + len to pass back to Java to avoid copying across the boundary.
fn get_ptr_len_view(index: jint, array: &VarBinViewArray) -> (*const u8, u32) {
    let bytes = array.bytes_at(usize::try_from(index).expect("index must fit in usize"));
    (
        bytes.as_ptr(),
        u32::try_from(bytes.len()).vortex_expect("string length must fit in u32"),
    )
}
