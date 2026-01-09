// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use futures::future::LocalBoxFuture;
use futures::stream::BoxStream;
use vortex_error::VortexResult;

use crate::file::IoRequest;

pub type ReadSourceRef = Arc<dyn ReadSource>;

/// An object-safe trait representing an open file-like I/O object for reading.
pub trait ReadSource: Send + Sync {
    /// The URI of this source, for logging and debugging purposes.
    fn uri(&self) -> &Arc<str>;

    /// The coalescing window to use for this source, if any.
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

#[derive(Clone, Copy, Debug)]
pub struct CoalesceWindow {
    /// The maximum "empty" distance between two requests to consider them for coalescing.
    pub distance: u64,
    /// The maximum total size spanned by a coalesced request.
    pub max_size: u64,
}
