// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_mask::Mask;

pub type ReaderRef = Arc<dyn Reader>;

/// A reader provides an interface for loading data from row-indexed layouts.
///
/// Unlike a [`super::source::DataSource`], readers have a concrete row count allowing fixed
/// partitions over a known set of rows. Readers are driven by providing an input stream of
/// array data that can be used to provide arguments to parameterized filter and projection
/// expressions.
pub trait Reader: 'static + Send + Sync {
    /// Downcast the reader to a concrete type.
    fn as_any(&self) -> &dyn Any;

    /// Get the data type of the layout being read.
    fn dtype(&self) -> &DType;

    /// Returns the number of rows in the reader.
    fn row_count(&self) -> u64;

    /// Reduces the reader, simplifying its internal structure if possible.
    fn try_reduce(&self) -> VortexResult<Option<ReaderRef>> {
        Ok(None)
    }

    /// Reduce the parent reader if possible, returning a new reader if successful.
    fn try_reduce_parent(
        &self,
        parent: &ReaderRef,
        child_idx: usize,
    ) -> VortexResult<Option<ReaderRef>> {
        let _ = (parent, child_idx);
        Ok(None)
    }

    /// Creates a scan over the given row range of the reader.
    ///
    /// TODO(ngates): we may want to pass `&dyn SegmentSource` here to force readers to construct
    ///  segment futures at this time. This allows for I/O pre-fetching to begin.
    fn execute(&self, row_range: Range<u64>) -> VortexResult<ReaderStreamRef>;
}

pub type ReaderStreamRef = Box<dyn ReaderStream>;

pub trait ReaderStream: 'static + Send + Sync {
    /// The data type of the returned data.
    fn dtype(&self) -> &DType;

    /// The preferred maximum row count for the next chunk.
    ///
    /// Returns [`None`] if there are no more chunks.
    fn next_chunk_len(&self) -> Option<usize>;

    /// Returns the next chunk of data given an input array.
    ///
    /// The returned chunk must have the same number of rows as the [`Mask::true_count`].
    /// The provided mask will have at most [`next_chunk_len`] rows.
    ///
    /// The returned future has a `'static` lifetime allowing the calling to drive the stream
    /// arbitrarily far without awaiting any data.
    fn next_chunk(
        &mut self,
        mask: &Mask,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>>;
}
