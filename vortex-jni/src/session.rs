// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! JNI bindings for [`vortex::session::VortexSession`].

use jni::EnvUnowned;
use jni::objects::JClass;
use jni::sys::jlong;
use vortex::VortexSessionDefault;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

use crate::RUNTIME;

/// Constructs a fresh [`VortexSession`] bound to the JNI-shared tokio runtime and returns
/// an opaque pointer that Java must pass to [`Java_dev_vortex_jni_NativeSession_free`].
pub(crate) fn new_session() -> Box<VortexSession> {
    let session = VortexSession::default().with_handle(RUNTIME.handle());
    vortex_parquet_variant::initialize(&session);
    Box::new(session)
}

/// SAFETY: caller must pass a pointer previously returned by [`new_session`].
pub(crate) unsafe fn session_ref<'a>(ptr: jlong) -> &'a VortexSession {
    debug_assert!(ptr != 0, "null session pointer");
    unsafe { &*(ptr as *const VortexSession) }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeSession_newSession(
    _env: EnvUnowned,
    _class: JClass,
) -> jlong {
    Box::into_raw(new_session()) as jlong
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeSession_free(
    _env: EnvUnowned,
    _class: JClass,
    pointer: jlong,
) {
    // SAFETY: pointer must have been created by `newSession` and not yet freed.
    drop(unsafe { Box::from_raw(pointer as *mut VortexSession) });
}
