// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;

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
    fn project(&self, row_range: Range<u64>) -> VortexResult<ReaderStreamRef>;

    /// Creates a scan over the given row range of the reader.
    ///
    /// TODO(ngates): we may want to pass `&dyn SegmentSource` here to force readers to construct
    ///  segment futures at this time. This allows for I/O pre-fetching to begin.
    fn filter(&self, row_range: Range<u64>) -> VortexResult<MaskStreamRef>;
}

pub type ReaderStreamRef = Box<dyn ReaderStream>;

/// A stream of data provided by a reader when driven by an input mask.
pub trait ReaderStream: 'static + Send + Sync {
    /// The data type of the returned data.
    fn dtype(&self) -> &DType;

    /// The preferred maximum row count for the next chunk.
    ///
    /// Returns [`None`] if there are no more chunks.
    fn next_chunk_len(&self) -> Option<usize>;

    /// Returns the next chunk of data given an input mask.
    ///
    /// The returned chunk must have the same number of rows as the [`Mask::true_count`].
    /// The provided mask will have at most [`next_chunk_len`] rows.
    ///
    /// The returned future has a `'static` lifetime allowing the calling to drive the stream
    /// arbitrarily far without awaiting any data.
    ///
    // TODO(ngates): we may want to take a MaskFuture here.
    fn next_chunk(
        &mut self,
        mask: &MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>>;
}

pub type MaskStreamRef = Box<dyn MaskStream>;

/// A stream of non-nullable boolean arrays provided by a reader when driven by an input mask.
///
/// Note that this is similar to [`ReaderStream`], except the returned arrays have row count
/// equal to the input mask's row count, rather than the mask's true count.
pub trait MaskStream: 'static + Send + Sync {
    /// The preferred maximum row count for the next chunk.
    ///
    /// Returns [`None`] if there are no more chunks.
    fn next_chunk_len(&self) -> Option<usize>;

    /// Returns the next chunk of data given an input mask.
    ///
    /// The returned chunk must have the same number of rows as the [`Mask::len`].
    /// The provided mask will have at most [`next_chunk_len`] rows.
    ///
    /// The returned future has a `'static` lifetime allowing the calling to drive the stream
    /// arbitrarily far without awaiting any data.
    // TODO(ngates): we may want to take a MaskFuture here.
    fn next_chunk(
        &mut self,
        mask: &MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>>;
}
