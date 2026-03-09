// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A [`VortexReadAt`] implementation backed by monoio's io_uring on a dedicated thread.
//!
//! Read requests are sent via a channel to a background thread running a monoio
//! event loop. That thread submits io_uring read operations and sends results back.

use std::os::unix::io::IntoRawFd;
use std::path::Path;
use std::sync::Arc;
use std::thread;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::CoalesceConfig;
use crate::VortexReadAt;

struct ReadRequest {
    offset: u64,
    length: usize,
    alignment: Alignment,
    response_tx: oneshot::Sender<VortexResult<BufferHandle>>,
}

/// A [`VortexReadAt`] backed by monoio's io_uring on a dedicated thread.
///
/// The reader spawns a background thread running a monoio event loop. Read
/// requests are submitted to that thread via a channel, and the monoio runtime
/// submits them to io_uring.
#[allow(clippy::expect_used, clippy::unwrap_used)]
pub struct MonoioReadAt {
    uri: Arc<str>,
    file_size: u64,
    request_tx: kanal::Sender<ReadRequest>,
}

// SAFETY: The file descriptor is only accessed by the monoio thread via its own File handle.
// The channel-based design ensures no concurrent access from multiple threads.
unsafe impl Send for MonoioReadAt {}
unsafe impl Sync for MonoioReadAt {}

#[allow(clippy::expect_used, clippy::unwrap_used, clippy::unwrap_in_result)]
impl MonoioReadAt {
    /// Open a file and spawn a monoio I/O thread to serve reads via io_uring.
    pub fn open(path: impl AsRef<Path>) -> VortexResult<Self> {
        let path = path.as_ref();
        let uri: Arc<str> = Arc::from(path.to_string_lossy().to_string());

        let file = std::fs::File::open(path)?;
        let file_size = file.metadata()?.len();

        let (request_tx, request_rx) = kanal::unbounded::<ReadRequest>();

        // We need to pass the std File to the monoio thread where it will be converted
        // into a monoio File. We use into_raw_fd + from_raw_fd to move ownership across
        // the thread boundary safely.
        let raw_fd = file.into_raw_fd();

        thread::Builder::new()
            .name("monoio-io".into())
            .spawn(move || {
                // Reconstruct the std::fs::File on this thread so we can pass it to monoio.
                let std_file = unsafe { std::fs::File::from_raw_fd(raw_fd) };

                monoio::RuntimeBuilder::<monoio::IoUringDriver>::new()
                    .build()
                    .expect("failed to build monoio runtime")
                    .block_on(io_thread_loop(std_file, request_rx));
            })
            .map_err(|e| vortex_err!("failed to spawn monoio thread: {e}"))?;

        Ok(Self {
            uri,
            file_size,
            request_tx,
        })
    }

    /// Open a file with `O_DIRECT` for true direct I/O (bypasses page cache).
    ///
    /// This requires Linux and a filesystem that supports `O_DIRECT`.
    /// The implementation handles alignment requirements internally by
    /// rounding offset down and length up to 4096-byte boundaries, then
    /// slicing the result to the requested range.
    #[allow(clippy::unwrap_in_result)]
    pub fn open_direct(path: impl AsRef<Path>) -> VortexResult<Self> {
        let path = path.as_ref();
        let uri: Arc<str> = Arc::from(path.to_string_lossy().to_string());

        use std::os::unix::fs::OpenOptionsExt;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)?;

        let file_size = file.metadata()?.len();
        let raw_fd = file.into_raw_fd();

        let (request_tx, request_rx) = kanal::unbounded::<ReadRequest>();

        thread::Builder::new()
            .name("monoio-io-direct".into())
            .spawn(move || {
                let std_file = unsafe { std::fs::File::from_raw_fd(raw_fd) };

                monoio::RuntimeBuilder::<monoio::IoUringDriver>::new()
                    .build()
                    .expect("failed to build monoio runtime")
                    .block_on(io_thread_loop_direct(std_file, request_rx));
            })
            .map_err(|e| vortex_err!("failed to spawn monoio direct I/O thread: {e}"))?;

        Ok(Self {
            uri,
            file_size,
            request_tx,
        })
    }
}

use std::os::unix::io::FromRawFd;

/// The event loop running on the monoio thread (normal I/O, no O_DIRECT).
#[allow(clippy::expect_used)]
async fn io_thread_loop(std_file: std::fs::File, request_rx: kanal::Receiver<ReadRequest>) {
    let file = monoio::fs::File::from_std(std_file).expect("failed to create monoio File");

    while let Ok(req) = request_rx.recv() {
        let buf = vec![0u8; req.length];

        // monoio's read_exact_at takes ownership of the buffer (completion-based model)
        // and returns (Result<()>, Vec<u8>) on completion.
        let (result, returned_buf) = file.read_exact_at(buf, req.offset).await;

        let response = match result {
            Ok(()) => {
                let mut aligned = ByteBufferMut::with_capacity_aligned(req.length, req.alignment);
                unsafe { aligned.set_len(req.length) };
                aligned.as_mut_slice().copy_from_slice(&returned_buf);
                Ok(BufferHandle::new_host(aligned.freeze()))
            }
            Err(e) => Err(vortex_err!(
                "monoio read error at offset={} len={}: {e}",
                req.offset,
                req.length
            )),
        };

        // Ignore send errors — the caller may have dropped their receiver.
        drop(req.response_tx.send(response));
    }

    // The monoio File will close the fd when dropped here.
    drop(file);
}

/// The event loop for O_DIRECT I/O. Handles alignment requirements by
/// rounding offset down and length up to 4096-byte boundaries, then slicing
/// the result to the requested range.
#[allow(clippy::expect_used, clippy::cast_possible_truncation)]
async fn io_thread_loop_direct(std_file: std::fs::File, request_rx: kanal::Receiver<ReadRequest>) {
    const BLOCK_SIZE: u64 = 4096;

    let file = monoio::fs::File::from_std(std_file).expect("failed to create monoio File");

    while let Ok(req) = request_rx.recv() {
        // Align offset down to block boundary.
        let aligned_offset = req.offset & !(BLOCK_SIZE - 1);
        let offset_adjustment = (req.offset - aligned_offset) as usize;

        // Align total read length up to block boundary.
        let total_needed = offset_adjustment + req.length;
        let aligned_length =
            (total_needed + (BLOCK_SIZE as usize - 1)) & !(BLOCK_SIZE as usize - 1);

        // Allocate a page-aligned buffer for O_DIRECT.
        let buf = aligned_vec(aligned_length, BLOCK_SIZE as usize);

        let (result, returned_buf) = file.read_exact_at(buf, aligned_offset).await;

        let response = match result {
            Ok(()) => {
                // Slice out the originally requested range.
                let slice = &returned_buf[offset_adjustment..offset_adjustment + req.length];
                let mut aligned = ByteBufferMut::with_capacity_aligned(req.length, req.alignment);
                unsafe { aligned.set_len(req.length) };
                aligned.as_mut_slice().copy_from_slice(slice);
                Ok(BufferHandle::new_host(aligned.freeze()))
            }
            Err(e) => Err(vortex_err!(
                "monoio direct read error at offset={} len={} (aligned: offset={} len={}): {e}",
                req.offset,
                req.length,
                aligned_offset,
                aligned_length
            )),
        };

        drop(req.response_tx.send(response));
    }

    drop(file);
}

/// Allocate a `Vec<u8>` with the given length, aligned to `align` bytes.
/// This is needed for O_DIRECT which requires page-aligned buffers.
#[allow(clippy::expect_used)]
fn aligned_vec(len: usize, align: usize) -> Vec<u8> {
    let layout =
        std::alloc::Layout::from_size_align(len, align).expect("invalid alignment for buffer");
    // SAFETY: layout has non-zero size and valid alignment.
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    // SAFETY: ptr was just allocated with the given layout and zeroed.
    unsafe { Vec::from_raw_parts(ptr, len, len) }
}

impl VortexReadAt for MonoioReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        Some(CoalesceConfig::file())
    }

    fn concurrency(&self) -> usize {
        // The monoio thread processes requests sequentially for now.
        // In a production implementation, we'd spawn concurrent monoio tasks
        // or use multiple io_uring SQEs in flight.
        32
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let size = self.file_size;
        async move { Ok(size) }.boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let (response_tx, response_rx) = oneshot::channel();
        let req = ReadRequest {
            offset,
            length,
            alignment,
            response_tx,
        };

        if let Err(_e) = self.request_tx.send(req) {
            return async move { Err(vortex_err!("monoio I/O thread is gone")) }.boxed();
        }

        async move {
            response_rx
                .await
                .map_err(|e| vortex_err!("monoio response channel closed: {e}"))?
        }
        .boxed()
    }
}

impl Drop for MonoioReadAt {
    fn drop(&mut self) {
        // Dropping all senders will close the channel, causing the monoio thread's
        // recv loop to exit. The monoio::fs::File will be dropped on that thread,
        // which closes the fd.
    }
}
