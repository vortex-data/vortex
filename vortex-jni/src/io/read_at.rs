// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Debug;
use std::sync::Arc;

use async_lock::Semaphore;
use async_trait::async_trait;
use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream;
use futures::stream::BoxStream;
use jni::JValue;
use jni::JavaVM;
use jni::objects::JObject;
use jni::refs::Global;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Alignment;
use vortex::buffer::ByteBufferMut;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::io::CoalesceConfig;
use vortex::io::VortexReadAt;
use vortex::io::filesystem::FileListing;
use vortex::io::filesystem::FileSystem;
use vortex::io::runtime::Handle;
use vortex::utils::aliases::hash_map::EntryRef;
use vortex::utils::aliases::hash_map::HashMap;

use crate::io::with_jvm;

/// Default number of concurrent `readFully` upcalls to allow across all files of one
/// [`JavaFileSystem`]. Matches the object-store default since the backing storage is
/// typically remote.
const DEFAULT_CONCURRENCY: usize = 192;

/// Shared cap on in-flight `readFully` upcalls across every file of one
/// [`JavaFileSystem`], so a wide scan cannot pin `files x concurrency` blocking
/// threads and Java streams.
#[derive(Clone)]
struct UpcallLimiter {
    semaphore: Arc<Semaphore>,
    concurrency: usize,
}

impl UpcallLimiter {
    fn new(concurrency: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(concurrency)),
            concurrency,
        }
    }
}

/// A [`VortexReadAt`] backed by a Java object implementing `dev.vortex.io.NativeReadable`.
///
/// Positional reads are forwarded as blocking `readFully(long, ByteBuffer)` upcalls
/// executed on the runtime's blocking pool. The destination is a direct `ByteBuffer`
/// wrapping the Rust-side allocation, so Java writes land in the buffer handed to the
/// scan without a copy through a heap `byte[]`. The file size is captured at
/// construction time (Java callers know it from metadata), so `size()` never crosses
/// the JNI boundary.
pub(crate) struct JavaReadable {
    vm: JavaVM,
    readable: Arc<Global<JObject<'static>>>,
    len: u64,
    handle: Handle,
    limiter: UpcallLimiter,
}

impl JavaReadable {
    fn new(
        vm: JavaVM,
        readable: Arc<Global<JObject<'static>>>,
        len: u64,
        handle: Handle,
        limiter: UpcallLimiter,
    ) -> Self {
        Self {
            vm,
            readable,
            len,
            handle,
            limiter,
        }
    }
}

impl VortexReadAt for JavaReadable {
    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        // Upcalls have a fixed JNI overhead and the backing storage is usually
        // remote, so favor fewer, larger reads.
        Some(CoalesceConfig::object_storage())
    }

    fn concurrency(&self) -> usize {
        self.limiter.concurrency
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let len = self.len;
        async move { Ok(len) }.boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let vm = self.vm.clone();
        let readable = Arc::clone(&self.readable);
        let len = self.len;
        let handle = self.handle.clone();
        let semaphore = Arc::clone(&self.limiter.semaphore);

        async move {
            // Take a permit before occupying a blocking thread. The lock-free
            // `try_acquire_arc` fast path deliberately barges ahead of queued
            // waiters: slight unfairness is fine, the limiter must never become
            // the bottleneck itself.
            let permit = match semaphore.try_acquire_arc() {
                Some(permit) => permit,
                None => semaphore.acquire_arc().await,
            };

            handle
                .spawn_blocking(move || {
                    // Keep the permit with the blocking work. Dropping the read future can
                    // cancel its task handle, but it cannot interrupt a `readFully` upcall
                    // that has already started.
                    let _permit = permit;
                    let end = offset
                        .checked_add(length as u64)
                        .ok_or_else(|| vortex_err!("read {offset}+{length} overflows u64"))?;
                    if end > len {
                        vortex_bail!("read {offset}..{end} out of bounds for file of length {len}");
                    }
                    // `java.nio.Buffer` capacities are Java ints.
                    if i32::try_from(length).is_err() {
                        vortex_bail!("read length {length} exceeds ByteBuffer limit");
                    }
                    let joffset = i64::try_from(offset)
                        .map_err(|_| vortex_err!("read offset {offset} exceeds i64"))?;

                    let mut buffer = ByteBufferMut::with_capacity_aligned(length, alignment);
                    with_jvm(&vm, |env| {
                        // SAFETY: the pointer covers `length` bytes of live allocation, and
                        // the direct buffer wrapping it does not outlive this call —
                        // `readFully` is synchronous and the interface forbids retaining
                        // the buffer after it returns.
                        let dst = unsafe {
                            env.new_direct_byte_buffer(
                                buffer.spare_capacity_mut().as_mut_ptr().cast(),
                                length,
                            )?
                        };
                        env.call_method(
                            readable.as_ref(),
                            jni::jni_str!("readFully"),
                            jni::jni_sig!("(JLjava/nio/ByteBuffer;)V"),
                            &[JValue::Long(joffset), JValue::Object(dst.as_ref())],
                        )?;
                        // Guard against implementations that return without filling the
                        // buffer, which would hand uninitialized memory to the scan.
                        let remaining = env
                            .call_method(
                                &dst,
                                jni::jni_str!("remaining"),
                                jni::jni_sig!("()I"),
                                &[],
                            )?
                            .i()?;
                        if remaining != 0 {
                            return Err(vortex_err!(
                                "readFully returned with {remaining} of {length} bytes unfilled"
                            )
                            .into());
                        }
                        Ok(())
                    })
                    .map_err(|e| e.with_context("readFully upcall failed"))?;

                    // SAFETY: `readFully` wrote all `length` bytes through the direct
                    // buffer, verified by the `remaining()` check above.
                    unsafe { buffer.set_len(length) };
                    Ok(BufferHandle::new_host(buffer.freeze()))
                })
                .await
        }
        .boxed()
    }
}

/// Wrap a single Java `NativeReadable` as a [`VortexReadAt`], without registering it in a
/// [`JavaFileSystem`].
///
/// For one-shot reads (opening a file to inspect its metadata, say) where there is no glob to
/// resolve and no other file to share an upcall budget with.
pub(crate) fn java_readable(
    vm: JavaVM,
    readable: Arc<Global<JObject<'static>>>,
    len: u64,
    handle: Handle,
) -> Arc<dyn VortexReadAt> {
    Arc::new(JavaReadable::new(
        vm,
        readable,
        len,
        handle,
        UpcallLimiter::new(DEFAULT_CONCURRENCY),
    ))
}

struct JavaFileEntry {
    readable: Arc<Global<JObject<'static>>>,
    size: u64,
}

/// A [`FileSystem`] over a fixed set of Java-provided readables, keyed by path.
///
/// Built by `NativeDataSource.openFiles`: every file the data source may touch is
/// registered up front together with its size, so `head` (and therefore exact-path
/// glob resolution) is answered without any upcall. `open_read` wraps the registered
/// Java object in a [`JavaReadable`].
pub(crate) struct JavaFileSystem {
    vm: JavaVM,
    files: HashMap<String, JavaFileEntry>,
    handle: Handle,
    limiter: UpcallLimiter,
}

impl JavaFileSystem {
    pub(crate) fn new(vm: JavaVM, handle: Handle, concurrency: Option<usize>) -> Self {
        Self {
            vm,
            files: HashMap::new(),
            handle,
            limiter: UpcallLimiter::new(concurrency.unwrap_or(DEFAULT_CONCURRENCY)),
        }
    }

    /// Register a Java readable for `path` with a known size.
    pub(crate) fn insert(
        &mut self,
        path: String,
        readable: Arc<Global<JObject<'static>>>,
        size: u64,
    ) -> VortexResult<()> {
        match self.files.entry_ref(&path) {
            EntryRef::Occupied(_) => {
                vortex_bail!("multiple Java readables normalize to path '{path}'");
            }
            EntryRef::Vacant(v) => v.insert(JavaFileEntry { readable, size }),
        };

        Ok(())
    }
}

impl Debug for JavaFileSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JavaFileSystem")
            .field("files", &self.files.keys())
            .finish()
    }
}

#[async_trait]
impl FileSystem for JavaFileSystem {
    fn list(&self, prefix: &str) -> BoxStream<'_, VortexResult<FileListing>> {
        let listings: Vec<VortexResult<FileListing>> = self
            .files
            .iter()
            .filter(|(path, _)| path.starts_with(prefix))
            .map(|(path, entry)| {
                Ok(FileListing {
                    path: path.clone(),
                    size: Some(entry.size),
                })
            })
            .collect();
        stream::iter(listings).boxed()
    }

    async fn head(&self, path: &str) -> VortexResult<Option<FileListing>> {
        Ok(self.files.get(path).map(|entry| FileListing {
            path: path.to_string(),
            size: Some(entry.size),
        }))
    }

    async fn open_read(&self, path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
        let entry = self
            .files
            .get(path)
            .ok_or_else(|| vortex_err!("no Java readable registered for path '{path}'"))?;
        Ok(Arc::new(JavaReadable::new(
            self.vm.clone(),
            Arc::clone(&entry.readable),
            entry.size,
            self.handle.clone(),
            self.limiter.clone(),
        )))
    }

    async fn delete(&self, path: &str) -> VortexResult<()> {
        vortex_bail!("delete('{path}') is not supported by a Java-readable file system")
    }
}
