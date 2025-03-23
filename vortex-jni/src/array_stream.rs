use std::mem::ManuallyDrop;
use std::pin::Pin;

use jni::sys::jlong;
use vortex::ArrayRef;
use vortex::error::VortexExpect;
use vortex::stream::ArrayStream;

use crate::TOKIO_RUNTIME;
use crate::array::NativeArray;

pub struct NativeArrayStream {
    inner: Option<Pin<Box<dyn ArrayStream>>>,
    current: Option<ArrayRef>,
}

impl NativeArrayStream {
    pub fn new(stream: Pin<Box<dyn ArrayStream>>) -> Box<Self> {
        Box::new(Self {
            inner: Some(stream),
            current: None,
        })
    }

    pub fn into_raw(self: Box<Self>) -> jlong {
        Box::into_raw(self) as jlong
    }

    pub unsafe fn from_raw(pointer: jlong) -> Box<Self> {
        unsafe { Box::from_raw(pointer as *mut NativeArrayStream) }
    }

    pub fn deallocate(self: Box<Self>) {
        let this = *self;
        drop(this);
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayStreamMethods_free(
    _env: jni::JNIEnv,
    _class: jni::objects::JClass,
    pointer: jlong,
) {
    unsafe { NativeArrayStream::from_raw(pointer) }.deallocate();
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeArrayStreamMethods_take(
    _env: jni::JNIEnv,
    _class: jni::objects::JClass,
    pointer: jlong,
) -> jlong {
    let mut stream = unsafe { NativeArrayStream::from_raw(pointer) };

    let inner = stream
        .inner
        .as_mut()
        .vortex_expect("NativeArrayStream::new");

    let array_ref = TOKIO_RUNTIME.block_on(inner.next());
    stream.current = Some(array_ref);

    // Convert the stream back into raw pointer to avoid freeing resources.
    let _ = Box::into_raw(stream);

    NativeArray::new(array_ref).into_raw()
}

// /*
//  * Class:     dev_vortex_jni_NativeArrayStreamMethods
//  * Method:    take
//  * Signature: (J)J
//  */
// JNIEXPORT jlong JNICALL Java_dev_vortex_jni_NativeArrayStreamMethods_take
//   (JNIEnv *, jclass, jlong);
//
// /*
//  * Class:     dev_vortex_jni_NativeArrayStreamMethods
//  * Method:    getDType
//  * Signature: (J)J
//  */
// JNIEXPORT jlong JNICALL Java_dev_vortex_jni_NativeArrayStreamMethods_getDType
//   (JNIEnv *, jclass, jlong);
//
// /*
//  * Class:     dev_vortex_jni_NativeArrayStreamMethods
//  * Method:    hasNext
//  * Signature: (J)Z
//  */
// JNIEXPORT jboolean JNICALL Java_dev_vortex_jni_NativeArrayStreamMethods_hasNext
//   (JNIEnv *, jclass, jlong);
