use arrow::datatypes::{DataType, FieldRef, Fields};
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use jni::JNIEnv;
use jni::objects::{JClass, JIntArray, JLongArray, JObject, JValue};
use jni::sys::{
    JNI_FALSE, JNI_TRUE, jboolean, jbyte, jbyteArray, jdouble, jfloat, jint, jlong, jobject,
    jshort, jstring,
};
use vortex::arrays::{VarBinArray, VarBinViewArray};
use vortex::arrow::IntoArrowArray;
use vortex::dtype::DType;
use vortex::error::{VortexError, VortexExpect, VortexResult, vortex_err};
use vortex::scalar::{DecimalValue, i256};
use vortex::{Array, ArrayRef, ToCanonical};

use crate::errors::{JNIError, try_or_throw};

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
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_nbytes(
    _env: JNIEnv,
    _class: JClass,
    array_ptr: jlong,
) -> jlong {
    let array_ref = unsafe { NativeArray::from_ptr(array_ptr) };
    array_ref.inner.nbytes() as jlong
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_exportToArrow<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass,
    array_ptr: jlong,
    arrow_schema_ptr: JLongArray<'local>,
    arrow_array_ptr: JLongArray<'local>,
) {
    let array_ref = unsafe { NativeArray::from_ptr(array_ptr) };

    try_or_throw(&mut env, |env| {
        let preferred_arrow_type = array_ref.inner.dtype().to_arrow_dtype()?;
        let viewless_arrow_type = data_type_no_views(preferred_arrow_type);

        let arrow_array = array_ref.inner.clone().into_arrow(&viewless_arrow_type)?;
        let (ffi_array, ffi_schema) =
            arrow::ffi::to_ffi(&arrow_array.to_data()).map_err(VortexError::from)?;

        let ffi_schema_ptr = Box::into_raw(Box::new(ffi_schema));
        let ffi_array_ptr = Box::into_raw(Box::new(ffi_array));

        // Return native Arrow FFI pointers to caller.
        env.set_long_array_region(arrow_schema_ptr, 0, &[ffi_schema_ptr as jlong])?;
        env.set_long_array_region(arrow_array_ptr, 0, &[ffi_array_ptr as jlong])?;
        Ok(())
    });
}

/// Visit the potentially nested DataType, replacing all instances of Utf8View and BinaryView
/// with non-Viewable equivalents. This is necessary because Spark and Iceberg do not support
/// Utf8View.
fn data_type_no_views(data_type: DataType) -> DataType {
    match data_type {
        DataType::BinaryView => DataType::Binary,
        DataType::Utf8View => DataType::Utf8,
        // List
        DataType::List(inner) | DataType::ListView(inner) => {
            let new_inner = (*inner)
                .clone()
                .with_data_type(data_type_no_views(inner.data_type().clone()));
            DataType::List(FieldRef::new(new_inner))
        }
        // LargeList
        DataType::LargeList(inner) | DataType::LargeListView(inner) => {
            let new_inner = (*inner)
                .clone()
                .with_data_type(data_type_no_views(inner.data_type().clone()));
            DataType::LargeList(FieldRef::new(new_inner))
        }
        DataType::Struct(fields) => {
            // Things
            let viewless_fields: Vec<FieldRef> = fields
                .iter()
                .map(|field_ref| {
                    let field = (*field_ref.clone()).clone();
                    let data_type = field.data_type().clone();
                    let field = field.with_data_type(data_type_no_views(data_type));
                    FieldRef::new(field)
                })
                .collect();
            DataType::Struct(Fields::from(viewless_fields))
        }
        DataType::Decimal128(precision, scale) => DataType::Decimal128(precision, scale),
        DataType::Decimal256(precision, scale) => DataType::Decimal256(precision, scale),
        DataType::FixedSizeList(..) => unreachable!("Vortex never returns FixedSizeList"),
        DataType::Union(..) => unreachable!("Vortex never returns Union"),
        DataType::Dictionary(..) => unreachable!("Vortex never returns Dictionary"),
        DataType::Map(..) => unreachable!("Vortex never returns Map"),
        DataType::RunEndEncoded(..) => unreachable!("Vortex never returns RunEndEncoded"),
        // The non-nested non-view types stay the same.
        dt => dt,
    }
}

/// Drop the native memory holding an Arrow FFI Schema behind the pointer.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_dropArrowSchema(
    _env: JNIEnv,
    _class: JClass,
    schema_ptr: jlong,
) {
    drop(unsafe { Box::from_raw(schema_ptr as *mut FFI_ArrowSchema) });
}

/// Drop FFI_ArrowArray behind the pointer.
///
/// Note that this doesn't not free the memory backing the arrow data buffers, those
/// are still backed by Vortex Buffers.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_dropArrowArray(
    _env: JNIEnv,
    _class: JClass,
    array_ptr: jlong,
) {
    drop(unsafe { Box::from_raw(array_ptr as *mut FFI_ArrowArray) });
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
        let field = array_ref
            .inner
            .to_struct()?
            .fields()
            .get(index as usize)
            .cloned()
            .ok_or_else(|| vortex_err!("Field index out of bounds"))?;
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
        let sliced_array = array_ref
            .inner
            .as_ref()
            .slice(start as usize, end as usize)?;
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
                    array_ref
                        .inner
                        .to_extension()
                        .vortex_expect("extension array")
                        .storage()
                        .scalar_at(index as usize)?
                } else {
                    array_ref.inner.scalar_at(index as usize)?
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
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_getBigDecimal(
    mut env: JNIEnv,
    _class: JClass,
    array_ptr: jlong,
    index: jint,
) -> jobject {
    let array_ref = unsafe { NativeArray::from_ptr(array_ptr) };
    try_or_throw(&mut env, |env| {
        let scalar_value = if array_ref.is_extension {
            array_ref
                .inner
                .to_extension()
                .vortex_expect("extension array")
                .storage()
                .scalar_at(index as usize)?
        } else {
            array_ref.inner.scalar_at(index as usize)?
        };

        let decimal_scalar = scalar_value.as_decimal();
        let DType::Decimal(decimal_type, ..) = decimal_scalar.dtype() else {
            return Err(vortex_err!("Expected Decimal type").into());
        };
        let scale = decimal_type.scale();
        if let Some(v) = decimal_scalar.decimal_value() {
            match v {
                DecimalValue::I8(v) => bigdecimal_i8(env, *v, scale),
                DecimalValue::I16(v) => bigdecimal_i16(env, *v, scale),
                DecimalValue::I32(v) => bigdecimal_i32(env, *v, scale),
                DecimalValue::I64(v) => bigdecimal_i64(env, *v, scale),
                DecimalValue::I128(v) => bigdecimal_i128(env, *v, scale),
                DecimalValue::I256(v) => bigdecimal_i256(env, *v, scale),
            }
        } else {
            Ok(JObject::null().into_raw())
        }
    })
}

static BIGDECIMAL_CLASS: &str = "java/math/BigDecimal";
static BIGINT_CLASS: &str = "java/math/BigInteger";

macro_rules! bigdecimal_from_bytes {
    ($typ:ty, $name:ident) => {
        fn $name(env: &mut JNIEnv, value: $typ, scale: i8) -> Result<jobject, JNIError> {
            // NOTE: BigInteger constructor expects big-endian bytes.
            let be_bytes = value.to_be_bytes();
            let bytearray = env.byte_array_from_slice(&be_bytes)?;
            let bigint = env.new_object(BIGINT_CLASS, "([B)V", &[JValue::from(&bytearray)])?;

            // Create the BigDecimal from a BigInteger + scale
            let bigdecimal = env.new_object(
                BIGDECIMAL_CLASS,
                "(Ljava/math/BigInteger;I)V",
                &[JValue::from(&bigint), JValue::from(scale as jint)],
            )?;

            Ok(bigdecimal.into_raw())
        }
    };
}

bigdecimal_from_bytes!(i8, bigdecimal_i8);
bigdecimal_from_bytes!(i16, bigdecimal_i16);
bigdecimal_from_bytes!(i32, bigdecimal_i32);
bigdecimal_from_bytes!(i64, bigdecimal_i64);
bigdecimal_from_bytes!(i128, bigdecimal_i128);
bigdecimal_from_bytes!(i256, bigdecimal_i256);

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayMethods_getBool(
    mut env: JNIEnv,
    _class: JClass,
    array_ptr: jlong,
    index: jint,
) -> jboolean {
    let array_ref = unsafe { NativeArray::from_ptr(array_ptr) };
    try_or_throw(&mut env, |_| {
        let value = array_ref.inner.scalar_at(index as usize)?;
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
        let value = array_ref.inner.scalar_at(index as usize)?;
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
        if !array_ref.inner.dtype().is_utf8() {
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
        let value = array_ref.inner.scalar_at(index as usize)?;
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
