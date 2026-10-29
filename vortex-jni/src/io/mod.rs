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

#[cfg(not(windows))]
use std::cell::RefCell;

use jni::Env;
use jni::JavaVM;
pub(crate) use read_at::JavaFileSystem;
pub(crate) use read_at::java_readable;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
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
    #[cfg(not(windows))]
    attach_as_daemon(vm)?;
    vm.attach_current_thread(f).map_err(VortexError::from)
}

/// Not on Windows, where detaching from a thread-local destructor can deadlock on the loader lock;
/// there the jni crate's own attachment, which detaches through fiber-local storage, is kept.
///
/// Native worker threads never exit, and the JVM waits for every non-daemon thread before it
/// exits, so attach them as daemons; `with_jvm` then reuses the attachment.
#[cfg(not(windows))]
fn attach_as_daemon(vm: &JavaVM) -> VortexResult<()> {
    if vm
        .is_thread_attached()
        .map_err(|e| vortex_err!("cannot check the JVM attachment: {e}"))?
    {
        return Ok(());
    }
    let raw = vm.get_raw();
    let mut env = std::ptr::null_mut();
    // SAFETY: `raw` is the live JavaVM `vm` wraps, and a null `args` attaches with defaults.
    let result = unsafe {
        ((**raw).v1_4.AttachCurrentThreadAsDaemon)(raw, &raw mut env, std::ptr::null_mut())
    };
    if result != jni::sys::JNI_OK {
        vortex_bail!("failed to attach a native thread to the JVM as a daemon: {result}");
    }
    DAEMON_ATTACHMENT.with(|attachment| *attachment.borrow_mut() = Some(DaemonAttachment(raw)));
    Ok(())
}

/// Detaches the thread when it exits. The jni crate only detaches attachments it made itself, so
/// without this the JVM would keep a thread registered for every worker that ever ran an upcall.
#[cfg(not(windows))]
struct DaemonAttachment(*mut jni::sys::JavaVM);

#[cfg(not(windows))]
impl Drop for DaemonAttachment {
    fn drop(&mut self) {
        // SAFETY: the JavaVM outlives every native thread, and this thread attached itself above.
        unsafe { ((**self.0).v1_1.DetachCurrentThread)(self.0) };
    }
}

#[cfg(not(windows))]
thread_local! {
    static DAEMON_ATTACHMENT: RefCell<Option<DaemonAttachment>> = const { RefCell::new(None) };
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;
    use std::time::Duration;

    use jni::InitArgsBuilder;
    use jni::JValue;
    use jni::JavaVM;
    use jni::errors::StartJvmError;
    use jni::objects::JObject;
    use jni::refs::Global;
    use vortex::error::VortexResult;
    use vortex::error::vortex_bail;
    use vortex::io::filesystem::FileSystem;
    use vortex::io::runtime::BlockingRuntime;
    use vortex::io::runtime::current::CurrentThreadRuntime;

    use super::JavaFileSystem;
    use super::with_jvm;

    /// Launch (or reuse) the process-wide JVM backing upcall tests. Returns `None`
    /// when no JVM installation can be located, so tests skip on machines without a
    /// JDK; any other launch failure panics.
    pub(crate) fn test_vm() -> Option<&'static JavaVM> {
        static VM: LazyLock<Option<JavaVM>> = LazyLock::new(|| {
            let args = InitArgsBuilder::new()
                .option("-Xcheck:jni")
                .build()
                .expect("valid JVM init args");
            match JavaVM::new(args) {
                Ok(vm) => Some(vm),
                Err(StartJvmError::NotFound(_)) => {
                    eprintln!("skipping JNI upcall test: no JVM found (set JAVA_HOME to run it)");
                    None
                }
                Err(e) => panic!("failed to launch test JVM: {e}"),
            }
        });
        VM.as_ref()
    }

    /// A fresh `java.lang.Object` global ref plus a `java.lang.ref.WeakReference`
    /// observing the same object, so tests can detect when the global ref is deleted.
    pub(crate) fn object_with_weak_ref(
        vm: &JavaVM,
    ) -> VortexResult<(Global<JObject<'static>>, Global<JObject<'static>>)> {
        with_jvm(vm, |env| {
            let obj =
                env.new_object(jni::jni_str!("java/lang/Object"), jni::jni_sig!("()V"), &[])?;
            let weak = env.new_object(
                jni::jni_str!("java/lang/ref/WeakReference"),
                jni::jni_sig!("(Ljava/lang/Object;)V"),
                &[JValue::Object(&obj)],
            )?;
            Ok((env.new_global_ref(&obj)?, env.new_global_ref(&weak)?))
        })
    }

    /// Assert that the referent observed by `weak` becomes collectible, i.e. that no
    /// JNI global reference pins it anymore.
    pub(crate) fn assert_weak_ref_clears(
        vm: &JavaVM,
        weak: &Global<JObject<'static>>,
    ) -> VortexResult<()> {
        for _ in 0..100 {
            let cleared = with_jvm(vm, |env| {
                env.call_static_method(
                    jni::jni_str!("java/lang/System"),
                    jni::jni_str!("gc"),
                    jni::jni_sig!("()V"),
                    &[],
                )?;
                let referent = env
                    .call_method(
                        weak.as_ref(),
                        jni::jni_str!("get"),
                        jni::jni_sig!("()Ljava/lang/Object;"),
                        &[],
                    )?
                    .l()?;
                Ok(referent.is_null())
            })?;
            if cleared {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        vortex_bail!("JNI global reference leaked: weak reference still has a referent")
    }

    #[test]
    fn test_file_system_drop_releases_global_refs() -> VortexResult<()> {
        let Some(vm) = test_vm() else {
            return Ok(());
        };
        let (readable, weak) = object_with_weak_ref(vm)?;

        let runtime = CurrentThreadRuntime::new();
        let mut fs = JavaFileSystem::new(vm.clone(), runtime.handle(), None);
        fs.insert("data/file.vortex".to_string(), Arc::new(readable), 4)?;
        // `open_read` clones the global ref into a `JavaReadable`; both it and the
        // file system must let go of the Java object once dropped.
        let reader = runtime.block_on(fs.open_read("data/file.vortex"))?;
        drop(fs);
        drop(reader);

        assert_weak_ref_clears(vm, &weak)
    }
}
