// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Extension traits for `Stream`.

mod buffer_unordered;

use std::sync::atomic::AtomicUsize;

pub use buffer_unordered::BufferUnordered;
use futures::Stream;

/// Extension trait for `Stream`.
pub trait StreamExt: Sized + Stream {
    /// Buffers unordered futures from this stream, with a maximum concurrency of `concurrency`.
    ///
    /// There are two main differences vs [`futures::stream::StreamExt::buffer_unordered`]:
    /// 1. This version takes an `AtomicUsize` for concurrency, allowing it to be modified at
    ///    runtime to adjust concurrency on the fly.
    /// 2. This version re-fills the in-progress queue prior to returning a value instead of
    ///    on the next iteration of `poll_next`. This can be important when the consumer does a lot
    ///    of work in between polls and the items of the stream are spawned.
    fn buffer_unordered2(self, concurrency: AtomicUsize) -> BufferUnordered<Self>;
}

impl<S: Stream> StreamExt for S {
    fn buffer_unordered2(self, concurrency: AtomicUsize) -> BufferUnordered<Self> {
        BufferUnordered::new(self, concurrency)
    }
}
