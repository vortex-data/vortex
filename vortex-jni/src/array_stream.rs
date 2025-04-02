use std::pin::Pin;

use futures::StreamExt;
use jni::JNIEnv;
use jni::objects::JClass;
use jni::sys::jlong;
use vortex::dtype::DType;
use vortex::stream::ArrayStream;

use crate::array::NativeArray;
use crate::block_on;
use crate::errors::try_or_throw;

/// Blocking JNI bridge to a Vortex [`ArrayStream`].
pub struct NativeArrayStream {
    inner: Option<Pin<Box<dyn ArrayStream>>>,
}

impl NativeArrayStream {
    pub fn new(stream: Pin<Box<dyn ArrayStream>>) -> Box<Self> {
        Box::new(Self {
            inner: Some(stream),
        })
    }

    pub fn into_raw(self: Box<Self>) -> jlong {
        Box::into_raw(self) as jlong
    }

    #[allow(clippy::expect_used)]
    pub unsafe fn from_ptr<'a>(pointer: jlong) -> &'a Self {
        unsafe {
            (pointer as *const NativeArrayStream)
                .as_ref()
                .expect("Pointer should never be null")
        }
    }

    #[allow(clippy::expect_used)]
    pub unsafe fn from_ptr_mut<'a>(pointer: jlong) -> &'a mut Self {
        unsafe {
            (pointer as *mut NativeArrayStream)
                .as_mut()
                .expect("Pointer should never be null")
        }
    }

    pub unsafe fn from_raw(pointer: jlong) -> Box<Self> {
        unsafe { Box::from_raw(pointer as *mut NativeArrayStream) }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayStreamMethods_free(
    _env: JNIEnv,
    _class: JClass,
    pointer: jlong,
) {
    drop(unsafe { NativeArrayStream::from_raw(pointer) });
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayStreamMethods_take(
    mut env: JNIEnv,
    _class: JClass,
    pointer: jlong,
) -> jlong {
    let stream = unsafe { NativeArrayStream::from_ptr_mut(pointer) };

    try_or_throw(&mut env, |_| {
        if let Some(mut inner) = stream.inner.take() {
            let next_fut = inner.next();
            match block_on("stream.next", next_fut) {
                Some(result) => {
                    let array_ref = result?;
                    stream.inner = Some(inner);
                    // return the pointer to the next array element
                    Ok(NativeArray::new(array_ref).into_raw())
                }
                None => Ok(-1),
            }
        } else {
            throw_runtime!("attempted to take() on a closed ArrayStream");
        }
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayStreamMethods_getDType(
    mut env: JNIEnv,
    _class: JClass,
    pointer: jlong,
) -> jlong {
    let stream = unsafe { NativeArrayStream::from_ptr(pointer) };

    try_or_throw(&mut env, |_| {
        if let Some(ref inner) = stream.inner {
            let dtype = inner.dtype();
            Ok(dtype as *const DType as jlong)
        } else {
            throw_runtime!("NativeArrayMethods.getDType: closed stream");
        }
    })
}
