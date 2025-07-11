// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;

use futures_util::StreamExt;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::{ArrayRef, iter::ArrayIterator};
use super::ArrayStream;

/// Trait for abstracting over async runtimes
pub trait AsyncRuntime {
    /// Block on a future until it completes
    fn block_on<F: Future>(&self, fut: F) -> F::Output;
}

/// Adapter for converting an [`ArrayStream`] into an [`ArrayIterator`].
///
/// This struct allows you to bridge the gap between async stream processing
/// and synchronous iterator processing by using a provided async runtime.
pub struct ArrayStreamToIterator<S, AR> {
    stream: S,
    runtime: AR,
}

impl<S, AR> ArrayStreamToIterator<S, AR>
where
    S: ArrayStream + Unpin + Send,
    AR: AsyncRuntime,
{
    /// Create a new adapter with the given stream and runtime
    pub fn new(stream: S, runtime: AR) -> Self {
        Self { stream, runtime }
    }
}

impl<S, AR> ArrayIterator for ArrayStreamToIterator<S, AR>
where
    S: ArrayStream + Unpin + Send,
    AR: AsyncRuntime,
{
    fn dtype(&self) -> &DType {
        self.stream.dtype()
    }
}

impl<S, AR> Iterator for ArrayStreamToIterator<S, AR>
where
    S: ArrayStream + Unpin + Send,
    AR: AsyncRuntime,
{
    type Item = VortexResult<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        self.runtime.block_on(self.stream.next())
    }
}