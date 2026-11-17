// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Positional reads of one descriptor through a shared ring.

use std::fmt;
use std::future::Future;
use std::os::fd::AsRawFd;
use std::os::fd::OwnedFd;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use futures::FutureExt;
use futures::future;
use futures::future::BoxFuture;
use io_uring::opcode;
use io_uring::types;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::UringRing;
use super::ring::Keep;
use crate::CoalesceConfig;
use crate::VortexReadAt;

/// The largest single read submission; longer reads continue as short reads do.
const MAX_CHUNK: usize = 1 << 30;

/// A file opened through a [`UringRing`], read through the same ring.
pub struct UringReadAt {
    uri: Arc<str>,
    fd: Arc<OwnedFd>,
    ring: UringRing,
}

impl UringReadAt {
    /// Reads `fd` through `ring`.
    pub fn new(uri: impl Into<Arc<str>>, fd: OwnedFd, ring: UringRing) -> Self {
        Self {
            uri: uri.into(),
            fd: Arc::new(fd),
            ring,
        }
    }

    /// The ring this file is read through.
    pub fn ring(&self) -> &UringRing {
        &self.ring
    }
}

impl fmt::Debug for UringReadAt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UringReadAt")
            .field("uri", &self.uri)
            .field("fd", &self.fd.as_raw_fd())
            .finish()
    }
}

impl VortexReadAt for UringReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        Some(CoalesceConfig::file())
    }

    /// A queued read costs one submission entry, not a thread.
    fn concurrency(&self) -> usize {
        256
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let size = rustix::fs::fstat(&*self.fd)
            .map_err(|err| vortex_err!("fstat {}: {err}", self.uri))
            .and_then(|stat| {
                u64::try_from(stat.st_size).map_err(|_| vortex_err!("negative size"))
            });
        future::ready(size).boxed()
    }

    /// Queues the read now; it is submitted when any operation of the ring is next polled.
    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let buf = ByteBufferMut::with_capacity_aligned(length, alignment);
        if length == 0 {
            return future::ready(Ok(BufferHandle::new_host(buf.freeze()))).boxed();
        }
        let mut read = ReadFuture {
            ring: self.ring.clone(),
            fd: Arc::clone(&self.fd),
            offset,
            length,
            filled: 0,
            token: None,
            buf: Some(buf),
        };
        match read.queue() {
            Ok(()) => read.boxed(),
            Err(err) => future::ready(Err(err)).boxed(),
        }
    }
}

struct ReadFuture {
    ring: UringRing,
    fd: Arc<OwnedFd>,
    offset: u64,
    length: usize,
    filled: usize,
    /// The operation in flight.
    token: Option<u64>,
    /// The buffer, while no operation holds it.
    buf: Option<ByteBufferMut>,
}

impl ReadFuture {
    /// Queues a read of the bytes not yet filled.
    fn queue(&mut self) -> VortexResult<()> {
        let buf = self
            .buf
            .take()
            .ok_or_else(|| vortex_err!("io_uring read queued twice"))?;
        let filled = self.filled;
        let chunk = (self.length - filled).min(MAX_CHUNK);
        let offset = self.offset + filled as u64;
        let fd = self.fd.as_raw_fd();
        let token = self.ring.shared().queue(
            Keep::Read {
                buf,
                _fd: Arc::clone(&self.fd),
            },
            |keep| {
                let Keep::Read { buf, .. } = keep else {
                    unreachable!("queued a read")
                };
                // SAFETY: `filled + chunk` is within the buffer's capacity of `length` bytes.
                let ptr = unsafe { buf.as_mut_ptr().add(filled) };
                opcode::Read::new(types::Fd(fd), ptr, chunk as u32)
                    .offset(offset)
                    .build()
            },
        )?;
        self.token = Some(token);
        Ok(())
    }
}

impl Future for ReadFuture {
    type Output = VortexResult<BufferHandle>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        loop {
            let Some(token) = this.token else {
                return Poll::Ready(Err(vortex_err!("io_uring read polled after completion")));
            };
            let (result, keep) = match this.ring.shared().poll_op(token, cx) {
                Poll::Ready(Ok(done)) => done,
                Poll::Ready(Err(err)) => return Poll::Ready(Err(err)),
                Poll::Pending => return Poll::Pending,
            };
            this.token = None;
            let Keep::Read { mut buf, .. } = keep else {
                unreachable!("a read keeps its buffer")
            };
            let read = match usize::try_from(result) {
                Ok(read) => read,
                Err(_) => {
                    let err = std::io::Error::from_raw_os_error(-result);
                    return Poll::Ready(Err(vortex_err!(
                        "io_uring read of {} bytes at {}: {err}",
                        this.length,
                        this.offset
                    )));
                }
            };
            if read == 0 {
                return Poll::Ready(Err(vortex_err!(
                    "io_uring read of {} bytes at {}: unexpected end of file after {} bytes",
                    this.length,
                    this.offset,
                    this.filled
                )));
            }
            this.filled += read;
            if this.filled == this.length {
                // SAFETY: the kernel wrote all `length` bytes.
                unsafe { buf.set_len(this.length) };
                return Poll::Ready(Ok(BufferHandle::new_host(buf.freeze())));
            }
            if this.filled > this.length {
                return Poll::Ready(Err(vortex_err!("io_uring read more bytes than asked for")));
            }
            this.buf = Some(buf);
            if let Err(err) = this.queue() {
                return Poll::Ready(Err(err));
            }
        }
    }
}

impl Drop for ReadFuture {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            self.ring.shared().abandon(token);
        }
    }
}
