// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_schema::ArrowError;
use jni::Env;
use jni::EnvUnowned;
use jni::Outcome;
use jni::objects::JObject;
use jni::strings::JNIString;
use jni::sys::JNI_FALSE;
use jni::sys::jboolean;
use jni::sys::jobject;
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

impl From<ArrowError> for JNIError {
    fn from(error: ArrowError) -> Self {
        JNIError::Vortex(VortexError::from(error))
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

/// Run the provided function inside the EnvUnowned context. Throws an exception if the function returns an error.
///
/// A panic inside `function` is caught by the underlying `with_env` bridge. We translate
/// it into a Java `RuntimeException` so it surfaces at the call site instead of silently
/// returning a default value (which would look like success to Java).
#[inline]
pub fn try_or_throw<'a, F, T>(env: &mut EnvUnowned<'a>, function: F) -> T
where
    F: FnOnce(&mut Env<'a>) -> Result<T, JNIError>,
    T: JNIDefault,
{
    let outcome = env.with_env(|env| -> Result<T, JNIError> {
        match function(env) {
            Ok(result) => Ok(result),
            Err(error) => {
                // Propagate the exception instead of throwing our own.
                if env.exception_check() {
                    return Ok(T::jni_default());
                }

                let msg = JNIString::new(error.to_string());
                match env.throw_new(jni::jni_str!("java/lang/RuntimeException"), msg) {
                    Ok(()) => {}
                    Err(err) => {
                        tracing::warn!("Failed throwing exception back up to Java: {err}")
                    }
                }

                Ok(T::jni_default())
            }
        }
    });

    match outcome.into_outcome() {
        Outcome::Ok(result) => result,
        Outcome::Err(_) => T::jni_default(),
        Outcome::Panic(payload) => {
            let msg = panic_message(payload.as_ref());
            // Best-effort: open a short-lived Env to throw. If it fails we at least log
            // instead of silently returning a success-shaped default.
            let threw = env.with_env(|env| -> Result<(), JNIError> {
                if !env.exception_check() {
                    let jmsg = JNIString::new(msg.clone());
                    drop(env.throw_new(jni::jni_str!("java/lang/RuntimeException"), jmsg));
                }
                Ok(())
            });
            if !matches!(threw.into_outcome(), Outcome::Ok(())) {
                tracing::error!("panic in JNI call, failed to propagate to Java: {msg}");
            }
            T::jni_default()
        }
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        format!("panic in JNI call: {s}")
    } else if let Some(s) = payload.downcast_ref::<String>() {
        format!("panic in JNI call: {s}")
    } else {
        "panic in JNI call".to_string()
    }
}
