use jni::JNIEnv;
use jni::objects::JClass;
use jni::sys::jlong;
use vortex::dtype::DType;
use vortex::iter::ArrayIterator;

use crate::array::NativeArray;
use crate::errors::try_or_throw;

/// Blocking JNI bridge to a Vortex [`ArrayIterator`].
pub struct NativeArrayIterator {
    inner: Option<Box<dyn ArrayIterator>>,
}

impl NativeArrayIterator {
    pub fn new(stream: Box<dyn ArrayIterator>) -> Box<Self> {
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
            (pointer as *const NativeArrayIterator)
                .as_ref()
                .expect("Pointer should never be null")
        }
    }

    #[allow(clippy::expect_used)]
    pub unsafe fn from_ptr_mut<'a>(pointer: jlong) -> &'a mut Self {
        unsafe {
            (pointer as *mut NativeArrayIterator)
                .as_mut()
                .expect("Pointer should never be null")
        }
    }

    pub unsafe fn from_raw(pointer: jlong) -> Box<Self> {
        unsafe { Box::from_raw(pointer as *mut NativeArrayIterator) }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayIteratorMethods_free(
    _env: JNIEnv,
    _class: JClass,
    pointer: jlong,
) {
    drop(unsafe { NativeArrayIterator::from_raw(pointer) });
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayIteratorMethods_take(
    mut env: JNIEnv,
    _class: JClass,
    pointer: jlong,
) -> jlong {
    let iter = unsafe { NativeArrayIterator::from_ptr_mut(pointer) };

    try_or_throw(&mut env, |_| {
        if let Some(mut inner) = iter.inner.take() {
            match inner.next() {
                Some(result) => {
                    let array_ref = result?;
                    iter.inner = Some(inner);
                    // return the pointer to the next array element
                    Ok(NativeArray::new(array_ref).into_raw())
                }
                None => Ok(-1),
            }
        } else {
            throw_runtime!("attempted to take() on a closed ArrayIter");
        }
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayIteratorMethods_getDType(
    mut env: JNIEnv,
    _class: JClass,
    pointer: jlong,
) -> jlong {
    let iter = unsafe { NativeArrayIterator::from_ptr(pointer) };

    try_or_throw(&mut env, |_| {
        if let Some(ref inner) = iter.inner {
            let dtype = inner.dtype();
            Ok(dtype as *const DType as jlong)
        } else {
            throw_runtime!("NativeArrayMethods.getDType: closed stream");
        }
    })
}
