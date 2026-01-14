// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use vortex_array::ArrayRef;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_mask::Mask;

/// A reader provides an interface for loading data from row-indexed layouts.
///
/// Unlike a [`super::source::Source`], readers have a concrete row count allowing fixed
/// partitions over a known set of rows. Readers are driven by providing an input stream of
/// array data that can be used to provide arguments to parameterized filter and projection
/// expressions.
#[async_trait]
pub trait Reader: 'static + Send + Sync {
    /// Get the data type of the layout being read.
    fn dtype(&self) -> &DType;

    /// Returns the number of rows in the reader.
    fn row_count(&self) -> u64;

    /// Creates a scan where an input stream is used to drive the output data.
    ///
    /// TODO(ngates): should this take a RowSelection?
    async fn scan(
        &self,
        input_dtype: &DType,
        row_offset: u64,
        row_mask: Mask,
    ) -> VortexResult<ReaderScanRef>;
}

pub type ReaderScanRef = Box<dyn ReaderScan>;

/// A scan over a reader, producing output arrays given an input array to parameterize the filter
/// and projection expressions.
#[async_trait]
pub trait ReaderScan {
    /// The data type of the returned data.
    fn dtype(&self) -> &DType;

    /// The preferred maximum row count for the next batch.
    ///
    /// Returns [`None`] if there are no more batches.
    fn next_batch_size(&mut self) -> Option<usize>;

    /// Returns the next batch of data given an input array.
    ///
    /// The returned batch must have the same number of rows as the input array.
    async fn next_batch(&mut self, input: ArrayRef) -> VortexResult<ArrayRef>;
}
