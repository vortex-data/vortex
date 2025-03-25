use std::pin::Pin;

use futures::StreamExt;
use jni::JNIEnv;
use jni::objects::JClass;
use jni::sys::jlong;
use vortex::dtype::DType;
use vortex::error::vortex_err;
use vortex::stream::ArrayStream;

use crate::TOKIO_RUNTIME;
use crate::array::NativeArray;
use crate::errors::Throwable;

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

    pub unsafe fn from_ptr<'a>(pointer: jlong) -> &'a Self {
        unsafe {
            (pointer as *const NativeArrayStream)
                .as_ref()
                .expect("Pointer should never be null")
        }
    }

    pub unsafe fn from_ptr_mtr<'a>(pointer: jlong) -> &'a mut Self {
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
    let stream = unsafe { NativeArrayStream::from_ptr_mtr(pointer) };

    if let Some(mut inner) = stream.inner.take() {
        let next_fut = inner.next();
        match TOKIO_RUNTIME.block_on(next_fut) {
            Some(Ok(array_ref)) => {
                stream.inner = Some(inner);
                // return the pointer to the next array element
                NativeArray::new(array_ref).into_raw()
            }
            Some(Err(err)) => {
                // Rethrow the exception in Java. Resources get cleaned up on drop.
                err.throw_runtime(&mut env, "ArrayStream failed to read next batch");
                -1
            }
            None => {
                // No next element available, drop the stream and return -1 to indicate
                // no more elements.
                -1
            }
        }
    } else {
        vortex_err!("closed stream").throw_runtime(&mut env, "NativeArrayMethods.take");
        -1
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayStreamMethods_getDType(
    mut env: JNIEnv,
    _class: JClass,
    pointer: jlong,
) -> jlong {
    let stream = unsafe { NativeArrayStream::from_ptr(pointer) };

    if let Some(ref inner) = stream.inner {
        let dtype = inner.dtype();
        dtype as *const DType as jlong
    } else {
        vortex_err!("closed stream").throw_runtime(&mut env, "NativeArrayMethods.getDType");
        -1
    }
}
