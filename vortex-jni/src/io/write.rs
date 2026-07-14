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
