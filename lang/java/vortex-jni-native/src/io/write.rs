// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::sync::Arc;

use jni::JValue;
use jni::JavaVM;
use jni::objects::JObject;
use jni::refs::Global;
use vortex::io::IoBuf;
use vortex::io::VortexWrite;

use crate::io::with_jvm;

/// Largest chunk pushed through a single `write` upcall. Java arrays are indexed by
/// `int`, and keeping chunks bounded also keeps per-call array allocations modest.
const MAX_WRITE_CHUNK: usize = 1 << 30;

/// A [`VortexWrite`] backed by a Java object implementing `dev.vortex.io.NativeWritable`.
///
/// Bytes are forwarded as blocking `write(byte[], int, int)` upcalls, `flush` maps to
/// `flush()`, and `shutdown` maps to `flush()` as well: the Java caller created the
/// underlying stream and remains responsible for closing it once the writer finishes.
/// Writes run inline on the runtime thread driving the write task, which is attached
/// to the JVM on first use.
pub(crate) struct JavaWrite {
    vm: JavaVM,
    writable: Arc<Global<JObject<'static>>>,
}

impl JavaWrite {
    pub(crate) fn new(vm: JavaVM, writable: Arc<Global<JObject<'static>>>) -> Self {
        Self { vm, writable }
    }

    fn write_slice(&self, bytes: &[u8]) -> io::Result<()> {
        for chunk in bytes.chunks(MAX_WRITE_CHUNK) {
            let jlen = i32::try_from(chunk.len()).map_err(io::Error::other)?;
            with_jvm(&self.vm, |env| {
                let array = env.byte_array_from_slice(chunk)?;
                env.call_method(
                    self.writable.as_ref(),
                    jni::jni_str!("write"),
                    jni::jni_sig!("([BII)V"),
                    &[
                        JValue::Object(array.as_ref()),
                        JValue::Int(0),
                        JValue::Int(jlen),
                    ],
                )?;
                Ok(())
            })
            .map_err(io::Error::other)?;
        }
        Ok(())
    }

    fn flush_upcall(&self) -> io::Result<()> {
        with_jvm(&self.vm, |env| {
            env.call_method(
                self.writable.as_ref(),
                jni::jni_str!("flush"),
                jni::jni_sig!("()V"),
                &[],
            )?;
            Ok(())
        })
        .map_err(io::Error::other)
    }
}

impl VortexWrite for JavaWrite {
    async fn write_all<B: IoBuf>(&mut self, buffer: B) -> io::Result<B> {
        self.write_slice(buffer.as_slice())?;
        Ok(buffer)
    }

    async fn flush(&mut self) -> io::Result<()> {
        self.flush_upcall()
    }

    async fn shutdown(&mut self) -> io::Result<()> {
        self.flush_upcall()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use jni::JValue;
    use jni::objects::JByteArray;
    use vortex::error::VortexResult;
    use vortex::error::vortex_err;

    use super::JavaWrite;
    use crate::io::tests::assert_weak_ref_clears;
    use crate::io::tests::test_vm;
    use crate::io::with_jvm;

    /// Write upcalls from a fresh native thread must reach the Java object, and once
    /// the writer (and every other native reference) is dropped, its JNI global ref
    /// must be deleted so the Java object becomes collectible.
    #[test]
    fn test_write_upcalls_release_global_ref() -> VortexResult<()> {
        let Some(vm) = test_vm() else {
            return Ok(());
        };

        // `ByteArrayOutputStream` has the same `write([BII)V`/`flush()V` shape as
        // `dev.vortex.io.NativeWritable`, so it stands in for a caller-provided sink.
        let (writable, content, weak) = with_jvm(vm, |env| {
            let sink = env.new_object(
                jni::jni_str!("java/io/ByteArrayOutputStream"),
                jni::jni_sig!("()V"),
                &[],
            )?;
            let weak = env.new_object(
                jni::jni_str!("java/lang/ref/WeakReference"),
                jni::jni_sig!("(Ljava/lang/Object;)V"),
                &[JValue::Object(&sink)],
            )?;
            Ok((
                env.new_global_ref(&sink)?,
                env.new_global_ref(&sink)?,
                env.new_global_ref(&weak)?,
            ))
        })?;

        let thread_vm = vm.clone();
        std::thread::spawn(move || {
            let write = JavaWrite::new(thread_vm.clone(), Arc::new(writable));
            write.write_slice(b"vortex")?;
            write.flush_upcall()?;
            // `with_jvm` attaches permanently: the thread stays attached until it
            // exits, at which point the attachment is torn down automatically.
            let attached = thread_vm
                .is_thread_attached()
                .map_err(|e| vortex_err!("is_thread_attached failed: {e}"))?;
            assert!(attached, "write upcalls should leave the thread attached");
            VortexResult::Ok(())
        })
        .join()
        .map_err(|_| vortex_err!("writer thread panicked"))??;

        let written = with_jvm(vm, |env| {
            let array = env
                .call_method(
                    content.as_ref(),
                    jni::jni_str!("toByteArray"),
                    jni::jni_sig!("()[B"),
                    &[],
                )?
                .l()?;
            let array = env.cast_local::<JByteArray>(array)?;
            let mut bytes = vec![0i8; array.len(env)?];
            array.get_region(env, 0, &mut bytes)?;
            Ok(bytes.into_iter().map(|b| b as u8).collect::<Vec<u8>>())
        })?;
        assert_eq!(written, b"vortex");

        drop(content);
        assert_weak_ref_clears(vm, &weak)
    }
}
