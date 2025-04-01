use jni::JNIEnv;
use jni::objects::JObject;
use jni::sys::{JNI_FALSE, jboolean, jobject};
use vortex::error::VortexError;

#[derive(Debug, thiserror::Error)]
pub enum JNIError {
    #[error("Vortex Error: {0}")]
    Vortex(VortexError),
    #[error("JNI Error: {0}")]
    Custom(jni::errors::Error),
}

impl From<jni::errors::Error> for JNIError {
    fn from(error: jni::errors::Error) -> Self {
        JNIError::Custom(error)
    }
}

impl From<VortexError> for JNIError {
    fn from(error: VortexError) -> Self {
        JNIError::Vortex(error)
    }
}

/// Types that have a reasonable default value to use
/// across the FFI.
pub trait JNIDefault {
    fn jni_default() -> Self;
}

impl JNIDefault for () {
    fn jni_default() -> Self {}
}

impl JNIDefault for jboolean {
    fn jni_default() -> Self {
        JNI_FALSE
    }
}

macro_rules! default_integer {
    ($type:path) => {
        impl JNIDefault for $type {
            fn jni_default() -> Self {
                -1
            }
        }
    };
}

macro_rules! default_nan {
    ($type:path) => {
        impl JNIDefault for $type {
            fn jni_default() -> Self {
                <$type>::NAN
            }
        }
    };
}

default_integer!(jni::sys::jbyte);
default_integer!(jni::sys::jshort);
default_integer!(jni::sys::jint);
default_integer!(jni::sys::jlong);
default_nan!(jni::sys::jfloat);
default_nan!(jni::sys::jdouble);

// All objectful types default to null.
impl JNIDefault for jobject {
    fn jni_default() -> Self {
        JObject::null().into_raw()
    }
}

/// Run the provided function inside of the JNIEnv context. Throws an exception if the function returns an error.
#[allow(clippy::expect_used)]
pub fn try_or_throw<'a, F, T>(env: &mut JNIEnv<'a>, function: F) -> T
where
    F: FnOnce(&mut JNIEnv<'a>) -> Result<T, JNIError>,
    T: JNIDefault,
{
    match function(env) {
        Ok(result) => result,
        Err(error) => {
            let msg = error.to_string();
            env.throw((RUNTIME_EXC_CLASS, msg))
                .expect("throwing exception back to Java failed, everything is bad");
            T::jni_default()
        }
    }
}

pub static RUNTIME_EXC_CLASS: &str = "java/lang/RuntimeException";
