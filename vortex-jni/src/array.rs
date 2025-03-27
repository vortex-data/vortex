use jni::JNIEnv;
use jni::objects::{JClass, JIntArray, JLongArray, JObject};
use jni::sys::{
    JNI_FALSE, JNI_TRUE, jboolean, jbyte, jbyteArray, jdouble, jfloat, jint, jlong, jshort, jstring,
};
use vortex::arrays::{VarBinArray, VarBinViewArray};
use vortex::compute::{scalar_at, slice};
use vortex::dtype::DType;
use vortex::error::{VortexExpect, VortexResult};
use vortex::{Array, ArrayRef, ArrayVariants};

use crate::errors::try_or_throw;

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

    #[allow(clippy::expect_used)]
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

    try_or_throw(&mut env, |_| {
        let Some(struct_array) = array_ref.inner.as_struct_typed() else {
            throw_runtime!("getField expected struct array");
        };

        let field = struct_array.maybe_null_field_by_idx(index as usize)?;
        Ok(NativeArray::new(field).into_raw())
    })
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

    try_or_throw(&mut env, |_| {
        let sliced_array = slice(array_ref.inner.as_ref(), start as usize, end as usize)?;
        Ok(NativeArray::new(sliced_array).into_raw())
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_getNull(
    mut env: JNIEnv,
    _class: JClass,
    array_ptr: jlong,
    index: jint,
) -> jboolean {
    let array_ref = unsafe { NativeArray::from_ptr(array_ptr) };
    try_or_throw(&mut env, |_| {
        let is_null = array_ref.inner.is_invalid(index as usize)?;
        if is_null { Ok(JNI_TRUE) } else { Ok(JNI_FALSE) }
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_getNullCount(
    mut env: JNIEnv,
    _class: JClass,
    array_ptr: jlong,
) -> jint {
    let array_ref = unsafe { NativeArray::from_ptr(array_ptr) };
    try_or_throw(&mut env, |_| {
        let count = array_ref.inner.invalid_count()?;
        Ok(jint::try_from(count).unwrap_or(-1))
    })
}

macro_rules! get_primitive {
    ($name:ident, $native:ty, $jtype:ty) => {
        #[unsafe(no_mangle)]
        pub extern "system" fn $name(
            mut env: JNIEnv,
            _class: JClass,
            array_ptr: jlong,
            index: jint,
        ) -> $jtype {
            let array_ref = unsafe { NativeArray::from_ptr(array_ptr) };
            try_or_throw(&mut env, |_| {
                let scalar_value = if array_ref.is_extension {
                    scalar_at(
                        array_ref
                            .inner
                            .as_extension_typed()
                            .vortex_expect("extension array")
                            .storage_data()
                            .as_ref(),
                        index as usize,
                    )?
                } else {
                    scalar_at(array_ref.inner.as_ref(), index as usize)?
                };

                Ok(scalar_value
                    .as_primitive()
                    .as_::<$native>()?
                    .unwrap_or_default())
            })
        }
    };
}

get_primitive!(Java_dev_vortex_jni_NativeArrayMethods_getByte, i8, jbyte);
get_primitive!(Java_dev_vortex_jni_NativeArrayMethods_getShort, i16, jshort);
get_primitive!(Java_dev_vortex_jni_NativeArrayMethods_getInt, i32, jint);
get_primitive!(Java_dev_vortex_jni_NativeArrayMethods_getLong, i64, jlong);
get_primitive!(Java_dev_vortex_jni_NativeArrayMethods_getFloat, f32, jfloat);
get_primitive!(
    Java_dev_vortex_jni_NativeArrayMethods_getDouble,
    f64,
    jdouble
);

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_getBool(
    mut env: JNIEnv,
    _class: JClass,
    array_ptr: jlong,
    index: jint,
) -> jboolean {
    let array_ref = unsafe { NativeArray::from_ptr(array_ptr) };
    try_or_throw(&mut env, |_| {
        let value = scalar_at(array_ref.inner.as_ref(), index as usize)?;
        match value.as_bool().value() {
            None => Ok(JNI_FALSE),
            Some(b) => {
                if b {
                    Ok(JNI_TRUE)
                } else {
                    Ok(JNI_FALSE)
                }
            }
        }
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_getUTF8<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    array_ptr: jlong,
    index: jint,
) -> jstring {
    let array_ref = unsafe { NativeArray::from_ptr(array_ptr) };
    try_or_throw(&mut env, |env| {
        let value = scalar_at(array_ref.inner.as_ref(), index as usize)?;
        match value.as_utf8().value() {
            None => Ok(JObject::null().into_raw()),
            Some(buf_str) => Ok(env.new_string(buf_str.as_str())?.into_raw()),
        }
    })
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

    try_or_throw(&mut env, |env| {
        if array_ref.inner.as_utf8_typed().is_none() {
            throw_runtime!("getUTF8_ptr_len expected UTF8 array");
        }

        if let Some(varbin) = array_ref.inner.as_any().downcast_ref::<VarBinArray>() {
            let (ptr, len) = get_ptr_len_varbin(index, varbin)?;
            env.set_long_array_region(&out_ptr, 0, &[ptr as jlong])?;
            env.set_int_array_region(&out_len, 0, &[len as jint])?;
        } else if let Some(varbinview) = array_ref.inner.as_any().downcast_ref::<VarBinViewArray>()
        {
            let (ptr, len) = get_ptr_len_view(index, varbinview);
            env.set_long_array_region(&out_ptr, 0, &[ptr as jlong])?;
            env.set_int_array_region(&out_len, 0, &[len as jint])?;
        } else {
            throw_runtime!("getUTF8_ptr_len expected VarBin or VarBinView");
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_getBinary<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    array_ptr: jlong,
    index: jint,
) -> jbyteArray {
    let array_ref = unsafe { NativeArray::from_ptr(array_ptr) };
    try_or_throw(&mut env, |env| {
        let value = scalar_at(array_ref.inner.as_ref(), index as usize)?;
        match value.as_binary().value() {
            None => Ok(JObject::null().into_raw()),
            Some(buf) => Ok(env.byte_array_from_slice(buf.as_slice())?.into_raw()),
        }
    })
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
    let bytes = array.bytes_at(usize::try_from(index).vortex_expect("index must fit in usize"));
    (
        bytes.as_ptr(),
        u32::try_from(bytes.len()).vortex_expect("string length must fit in u32"),
    )
}
