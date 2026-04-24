// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compatibility layer similar to Rust async-compat that allows non-Tokio runtimes to use
//! Tokio-based implementations of the core traits, such as the ObjectStore implementations.
//!
//! This works in the same way as async-compat, by either pulling a Tokio runtime from the
//! current context, or by creating a new global single-thread Tokio runtime if one is not found.
//!
//! We cannot use async-compat directly because we need to wrap Streams as well as Futures,
//! and async-compat only supports the latter.

mod filesystem;
#[cfg(feature = "object_store")]
mod obj_store;
mod read_at;
mod write;

use std::pin::Pin;
use std::sync::LazyLock;
use std::task::Context;
use std::task::Poll;
use std::thread;

use futures::Stream;
use pin_project_lite::pin_project;
use vortex_error::VortexExpect;

/// Get a handle to the current Tokio runtime, or create a new global single-thread runtime if one
/// is not found.
///
/// From
fn runtime_handle() -> tokio::runtime::Handle {
    static TOKIO: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
        thread::Builder::new()
            .name("vortex-async-compat".into())
            .spawn(|| TOKIO.block_on(Pending))
            .vortex_expect("cannot start tokio runtime thread");
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .vortex_expect("cannot start tokio runtime")
    });

    tokio::runtime::Handle::try_current().unwrap_or_else(|_| TOKIO.handle().clone())
}

struct Pending;
impl Future for Pending {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

pin_project! {
    /// Compatibility adapter for futures and I/O types.
    #[derive(Clone, Debug)]
    pub struct Compat<T> {
        #[pin]
        inner: Option<T>,
    }

    impl<T> PinnedDrop for Compat<T> {
        fn drop(this: Pin<&mut Self>) {
            if this.inner.is_some() {
                // If the inner future wasn't moved out using into_inner,
                // enter the tokio context while the inner value is dropped.
                let _guard = runtime_handle().enter();
                this.project().inner.set(None);
            }
        }
    }
}

impl<T> Compat<T> {
    /// Create a new Compat wrapper around the given value.
    pub fn new(inner: T) -> Self {
        Self { inner: Some(inner) }
    }

    #[inline]
    fn inner(&self) -> &T {
        self.inner
            .as_ref()
            .vortex_expect("inner is only None when Compat is about to drop")
    }

    #[inline]
    fn inner_mut(&mut self) -> &mut T {
        self.inner
            .as_mut()
            .vortex_expect("inner is only None when Compat is about to drop")
    }

    fn get_pin_mut(self: Pin<&mut Self>) -> Pin<&mut T> {
        self.project()
            .inner
            .as_pin_mut()
            .vortex_expect("inner is only None when Compat is about to drop")
    }
}

#[deny(clippy::missing_trait_methods)]
impl<T: Future> Future for Compat<T> {
    type Output = T::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let _guard = runtime_handle().enter();
        self.get_pin_mut().poll(cx)
    }
}

#[deny(clippy::missing_trait_methods)]
impl<S: Stream> Stream for Compat<S> {
    type Item = S::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let _guard = runtime_handle().enter();
        self.get_pin_mut().poll_next(cx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner().size_hint()
    }
}
