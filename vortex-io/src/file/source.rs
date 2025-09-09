// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::FutureExt;
use futures::future::{BoxFuture, LocalBoxFuture};
use futures::stream::BoxStream;
use vortex_error::VortexResult;

use crate::file::request::IoRequest;
use crate::runtime::Handle;

/// An object-safe trait representing an open file-like I/O object.
pub trait IoSource: Send + Sync {
    fn uri(&self) -> &Arc<str>;

    fn coalesce_window(&self) -> Option<CoalesceWindow>;

    /// Returns a shared future that resolves to the byte size of the underlying data source.
    fn size(&self) -> BoxFuture<'static, VortexResult<u64>>;

    /// Drive a stream of I/O requests to completion.
    fn drive_send(
        self: Arc<Self>,
        requests: BoxStream<'static, IoRequest>,
    ) -> BoxFuture<'static, ()>;

    /// Drive a stream of I/O requests to completion on the local thread.
    fn drive_local(
        self: Arc<Self>,
        requests: BoxStream<'static, IoRequest>,
    ) -> LocalBoxFuture<'static, ()> {
        self.drive_send(requests).boxed_local()
    }
}

pub type IoSourceRef = Arc<dyn IoSource>;

#[derive(Clone, Copy, Debug)]
pub struct CoalesceWindow {
    /// The maximum "empty" distance between two requests to consider them for coalescing.
    pub distance: u64,
    /// The maximum total size spanned by a coalesced request.
    pub max_size: u64,
}

/// A trait for types that can be opened as an `IoSource`.
pub trait IntoIoSource {
    fn into_io_source(self, handle: Handle) -> VortexResult<IoSourceRef>;
}
