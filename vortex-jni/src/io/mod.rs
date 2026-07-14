// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Rust implementations of Vortex I/O traits backed by Java objects (upcalls).
//!
//! These bridges let Java callers supply their own I/O — e.g. Iceberg's `FileIO`
//! streams — instead of having the native side open storage itself. Java objects
//! are held as JNI global references and their methods are invoked from whichever
//! runtime thread executes the I/O.
//!
//! # Threading
//!
//! Vortex runtime threads (smol workers and the `blocking` pool used by
//! [`Handle::spawn_blocking`](vortex::io::runtime::Handle::spawn_blocking)) are not
//! JVM threads. Every upcall goes through [`with_jvm`], which attaches the current
//! thread on first use (detached automatically at thread exit) — re-attachment is a
//! cheap thread-local lookup.
//!
//! # JavaVM lifetime
//!
//! Each bridge struct stores the process-wide [`JavaVM`] pointer, captured once at
//! construction from the [`Env`] of the JNI entry point that created it.
//! This is safe with respect to serialization: only *Java* objects are serialized
//! (e.g. Iceberg's `FileIO` shipped to executors), and they reconstruct their native
//! state through fresh JNI calls after deserialization, at which point the entry
//! point's `Env` provides the VM again. Native bridge objects never outlive the
//! process, and JNI guarantees a single VM per process for its entire lifetime.

mod read_at;
mod write;

use jni::Env;
use jni::JavaVM;
pub(crate) use read_at::JavaFileSystem;
use vortex::error::VortexError;
use vortex::error::VortexResult;
pub(crate) use write::JavaWrite;

use crate::errors::JNIError;

/// Run `f` on the current thread with a JVM attachment.
///
/// Attaching is a thread-local lookup when the thread is already attached; threads
/// attached here stay attached until they exit. A Java exception thrown inside `f`
/// is caught, cleared, and surfaced as a `VortexError` carrying the exception's
/// class, message, and stack trace.
pub(crate) fn with_jvm<T>(
    vm: &JavaVM,
    f: impl FnOnce(&mut Env) -> Result<T, JNIError>,
) -> VortexResult<T> {
    vm.attach_current_thread(f).map_err(VortexError::from)
}

/// View a mutable byte slice as `jbyte`s for JNI array-region calls.
fn as_jbyte_slice_mut(bytes: &mut [u8]) -> &mut [i8] {
    // SAFETY: `u8` and `i8` have identical size and alignment, and every bit
    // pattern is valid for both.
    unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<i8>(), bytes.len()) }
}
