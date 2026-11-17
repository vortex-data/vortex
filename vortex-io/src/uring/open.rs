// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Opening files through a shared ring.

use std::ffi::CString;
use std::fmt;
use std::future::Future;
use std::os::fd::FromRawFd;
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
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::UringReadAt;
use super::UringRing;
use super::ring::Keep;
use crate::VortexReadAt;
use crate::request::Openable;

/// A local path opened with `IORING_OP_OPENAT` on a shared ring, and read through it.
///
/// Every file opened through one ring shares its submission queue, so the opens a planner
/// requests together go to the kernel in one `io_uring_enter`, and so do later reads of all of
/// them. The path is the key.
#[derive(Clone)]
pub struct UringOpenable {
    ring: UringRing,
    path: Arc<str>,
}

impl UringOpenable {
    /// Opens `path` through `ring`.
    pub fn new(ring: UringRing, path: impl Into<Arc<str>>) -> Self {
        Self {
            ring,
            path: path.into(),
        }
    }
}

impl fmt::Debug for UringOpenable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("UringOpenable").field(&self.path).finish()
    }
}

impl Openable for UringOpenable {
    fn key(&self) -> &Arc<str> {
        &self.path
    }

    /// Queues the open now; it is submitted when any operation of the ring is next polled.
    fn open(&self) -> BoxFuture<'static, VortexResult<Arc<dyn VortexReadAt>>> {
        let path = match CString::new(self.path.as_bytes()) {
            Ok(path) => path,
            Err(err) => return future::ready(Err(vortex_err!("{}: {err}", self.path))).boxed(),
        };
        let queued = self.ring.shared().queue(Keep::Path(path), |keep| {
            let Keep::Path(path) = keep else {
                unreachable!("queued an open")
            };
            opcode::OpenAt::new(types::Fd(libc::AT_FDCWD), path.as_ptr())
                .flags(libc::O_RDONLY | libc::O_CLOEXEC)
                .build()
        });
        match queued {
            Ok(token) => OpenFuture {
                ring: self.ring.clone(),
                path: Arc::clone(&self.path),
                token: Some(token),
            }
            .map(|read| read.map(|read| Arc::new(read) as Arc<dyn VortexReadAt>))
            .boxed(),
            Err(err) => future::ready(Err(err)).boxed(),
        }
    }
}

struct OpenFuture {
    ring: UringRing,
    path: Arc<str>,
    token: Option<u64>,
}

impl Future for OpenFuture {
    type Output = VortexResult<UringReadAt>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let Some(token) = this.token else {
            return Poll::Ready(Err(vortex_err!("io_uring open polled after completion")));
        };
        let (result, _path) = match this.ring.shared().poll_op(token, cx) {
            Poll::Ready(Ok(done)) => done,
            Poll::Ready(Err(err)) => return Poll::Ready(Err(err)),
            Poll::Pending => return Poll::Pending,
        };
        this.token = None;
        if result < 0 {
            let err = std::io::Error::from_raw_os_error(-result);
            return Poll::Ready(Err(vortex_err!("io_uring open {}: {err}", this.path)));
        }
        // SAFETY: a successful OPENAT returns a new descriptor that nothing else owns.
        let fd = unsafe { OwnedFd::from_raw_fd(result) };
        Poll::Ready(Ok(UringReadAt::new(
            Arc::clone(&this.path),
            fd,
            this.ring.clone(),
        )))
    }
}

impl Drop for OpenFuture {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            self.ring.shared().abandon(token);
        }
    }
}
