// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::pin::Pin;

pub use adapter::*;
pub use ext::*;
use futures::Stream;
use futures::stream;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::dtype::DType;

mod adapter;
mod ext;

/// A stream of array chunks along with a DType.
///
/// Can be thought of as equivalent to Arrow's RecordBatchReader.
pub trait ArrayStream: Stream<Item = VortexResult<ArrayRef>> {
    fn dtype(&self) -> &DType;
}

/// Trait for a [`Stream`] of [`ArrayRef`]s that can be passed between threads.
pub type SendableArrayStream = Pin<Box<dyn ArrayStream + Send>>;

impl ArrayStream for SendableArrayStream {
    fn dtype(&self) -> &DType {
        (**self).dtype()
    }
}

impl ArrayRef {
    /// Create an [`ArrayStream`] over the array.
    pub fn to_array_stream(&self) -> impl ArrayStream + 'static {
        ArrayStreamAdapter::new(self.dtype().clone(), stream::iter(self.to_array_iterator()))
    }
}
