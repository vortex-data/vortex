// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;

use tokio::runtime::Handle;
use vortex::stream::{ArrayStream, ArrayStreamToIterator, AsyncRuntime};

use crate::TOKIO_RUNTIME;

/// Tokio runtime adapter for use with ArrayStreamToIterator
pub(crate) struct TokioRuntimeAdapter(Handle);

impl TokioRuntimeAdapter {
    pub(crate) fn new() -> Self {
        Self(TOKIO_RUNTIME.handle().clone())
    }
}

impl AsyncRuntime for TokioRuntimeAdapter {
    fn block_on<F: Future>(&self, fut: F) -> F::Output {
        self.0.block_on(fut)
    }
}

/// Convenience function to create an ArrayStreamToIterator with the global tokio runtime
pub(crate) fn array_stream_to_iterator<S>(stream: S) -> ArrayStreamToIterator<S, TokioRuntimeAdapter>
where
    S: ArrayStream + Unpin + Send,
{
    ArrayStreamToIterator::new(stream, TokioRuntimeAdapter::new())
}
