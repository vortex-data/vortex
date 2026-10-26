// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! JNI bindings for [`vortex::session::VortexSession`].

use jni::EnvUnowned;
use jni::objects::JClass;
use jni::objects::JString;
use jni::sys::jlong;
use vortex::VortexSessionDefault;
use vortex::editions::CORE_2026_08_3;
use vortex::editions::EditionSessionExt;
use vortex::error::VortexExpect;
use vortex::error::vortex_err;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

use crate::RUNTIME;
use crate::errors::try_or_throw;

/// Constructs a fresh [`VortexSession`] bound to the JNI-shared tokio runtime and returns
/// an opaque pointer that Java must pass to [`Java_dev_vortex_jni_NativeSession_free`].
pub(crate) fn new_session() -> Box<VortexSession> {
    let session = VortexSession::default().with_handle(RUNTIME.handle());
    session
        .enable_edition(CORE_2026_08_3)
        .map_err(|error| vortex_err!("{error}"))
        .vortex_expect("JNI-supported draft core edition is registered");
    vortex_parquet_variant::initialize(&session);
    vortex_spatial::initialize(&session);
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

/// Enables the edition that has the name `edition` (for example `core2025.10.0`) for writes.
///
/// The edition replaces the enabled edition of the same family. Writers that start after this
/// call use the new selection. Throws a `RuntimeException` if the session has no edition with
/// this name.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeSession_enableEdition(
    mut env: EnvUnowned,
    _class: JClass,
    pointer: jlong,
    edition: JString,
) {
    try_or_throw(&mut env, |env| {
        let name: String = edition.try_to_string(env)?;

        // SAFETY: pointer must have been created by `newSession` and not yet freed.
        let session = unsafe { session_ref(pointer) };

        let edition = session
            .editions()
            .find_by_name(&name)
            .ok_or_else(|| vortex_err!("Unknown Vortex edition: {name}"))?;

        session
            .enable_edition(edition.id)
            .map_err(|error| vortex_err!("{error}"))?;

        Ok(())
    })
}
