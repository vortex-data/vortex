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
use jni::Env;
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

use crate::errors::JNIError;
use crate::io::with_jvm;

/// Capture and validate a Java readable's coalescing limits before registering it.
pub(crate) fn read_coalesce_config(
    env: &mut Env,
    readable: &JObject,
) -> Result<CoalesceConfig, JNIError> {
    let distance = env
        .call_method(
            readable,
            jni::jni_str!("coalesceDistance"),
            jni::jni_sig!("()I"),
            &[],
        )?
        .i()?;
    let max_size = env
        .call_method(
            readable,
            jni::jni_str!("coalesceMaxSize"),
            jni::jni_sig!("()I"),
            &[],
        )?
        .i()?;
    Ok(checked_coalesce_config(distance, max_size)?)
}

fn checked_coalesce_config(distance: i32, max_size: i32) -> VortexResult<CoalesceConfig> {
    if distance < 0 {
        vortex_bail!("coalesceDistance must be >= 0, got {distance}");
    }
    if max_size <= 0 {
        vortex_bail!("coalesceMaxSize must be > 0, got {max_size}");
    }
    // FileSegmentSource adds leading alignment padding to the coalescing window. Reserve
    // headroom for every alignment accepted from a file so merged reads still fit a ByteBuffer.
    let max_padding = Alignment::MAX_UNTRUSTED.as_usize() as u64 - 1;
    let max_size = (max_size as u64).min(i32::MAX as u64 - max_padding);
    Ok(CoalesceConfig::new(distance as u64, max_size))
}

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
    coalesce_config: CoalesceConfig,
    handle: Handle,
    limiter: UpcallLimiter,
}

impl JavaReadable {
    fn new(
        vm: JavaVM,
        readable: Arc<Global<JObject<'static>>>,
        len: u64,
        coalesce_config: CoalesceConfig,
        handle: Handle,
        limiter: UpcallLimiter,
    ) -> Self {
        Self {
            vm,
            readable,
            len,
            coalesce_config,
            handle,
            limiter,
        }
    }
}

impl VortexReadAt for JavaReadable {
    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        Some(self.coalesce_config)
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
    coalesce_config: CoalesceConfig,
    handle: Handle,
) -> Arc<dyn VortexReadAt> {
    Arc::new(JavaReadable::new(
        vm,
        readable,
        len,
        coalesce_config,
        handle,
        UpcallLimiter::new(DEFAULT_CONCURRENCY),
    ))
}

struct JavaFileEntry {
    readable: Arc<Global<JObject<'static>>>,
    size: u64,
    coalesce_config: CoalesceConfig,
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
        coalesce_config: CoalesceConfig,
    ) -> VortexResult<()> {
        match self.files.entry_ref(&path) {
            EntryRef::Occupied(_) => {
                vortex_bail!("multiple Java readables normalize to path '{path}'");
            }
            EntryRef::Vacant(v) => v.insert(JavaFileEntry {
                readable,
                size,
                coalesce_config,
            }),
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
            entry.coalesce_config,
            self.handle.clone(),
            self.limiter.clone(),
        )))
    }

    async fn delete(&self, path: &str) -> VortexResult<()> {
        vortex_bail!("delete('{path}') is not supported by a Java-readable file system")
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex::error::VortexResult;

    use super::checked_coalesce_config;

    #[rstest]
    #[case(0, 1, 1)]
    #[case(64 * 1024, 16 * 1024 * 1024, 16 * 1024 * 1024)]
    #[case(i32::MAX, i32::MAX, 2_147_418_112)]
    fn test_coalescing_limits(
        #[case] distance: i32,
        #[case] max_size: i32,
        #[case] expected_max_size: u64,
    ) -> VortexResult<()> {
        let config = checked_coalesce_config(distance, max_size)?;
        assert_eq!(config.distance, distance as u64);
        assert_eq!(config.max_size, expected_max_size);
        Ok(())
    }

    #[rstest]
    #[case(-1, 1024, "coalesceDistance")]
    #[case(0, 0, "coalesceMaxSize")]
    #[case(0, -1, "coalesceMaxSize")]
    fn test_invalid_coalescing_limits(
        #[case] distance: i32,
        #[case] max_size: i32,
        #[case] setting: &str,
    ) {
        let error =
            checked_coalesce_config(distance, max_size).expect_err("invalid coalescing limits");
        assert!(error.to_string().contains(setting));
    }
}
