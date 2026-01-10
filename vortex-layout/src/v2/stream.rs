// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_mask::Mask;

pub type SendableLayoutReaderStream = Box<dyn LayoutReaderStream + 'static + Send + Sync>;

/// A stream of data produced by a [`LayoutReader2`](crate::v2::reader::LayoutReader2).
///
/// Layout readers are driven by requesting chunks of data using a given selection masks.
pub trait LayoutReaderStream {
    /// Returns the length in rows of the next chunk in the stream.
    ///
    /// Returns [`None`] if the stream has ended.
    fn next_chunk_len(&self) -> Option<usize>;

    /// Returns the next chunk of data given a selection mask of the requested length.
    ///
    /// The length of the provided selection mask must be `<=` the size returned from
    /// [`LayoutReaderStream::next_chunk_len`].
    ///
    /// The length of the returned chunk must be equal to the [`Mask::true_count`] of the selection
    /// mask.
    ///
    /// The returned future has a `'static` lifetime allowing the calling to drive the stream
    /// arbitrarily far without awaiting any data.
    fn next_chunk(
        &mut self,
        selection: &Mask,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>>;
}
