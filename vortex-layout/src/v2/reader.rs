// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::v2::stream::SendableLayoutReaderStream;

pub type LayoutReader2Ref = Arc<dyn LayoutReader2>;

pub trait LayoutReader2: 'static + Send + Sync {
    /// Returns the number of rows in the layout.
    ///
    /// TODO(ngates): if we relaxed this to be a cardinality estimate, we could support arbitrary
    ///  data streams including joins, group bys, scans, etc. The problem is, invoking execute with
    ///  some row range becomes weird...
    ///  Perhaps we borrow DataFusion's style of partitioning where we ask the reader to partition
    ///  into `n` and then pass the partition index to execute? Or perhaps we just pass `n` to the
    ///  execute call and have the reader return all `n` partitions at once? That would also make
    ///  sharing cached resources a lot easier.
    fn row_count(&self) -> u64;

    /// Returns the [`DType`] of the layout.
    fn dtype(&self) -> &DType;

    /// Returns the number of child layouts.
    fn nchildren(&self) -> usize;

    /// Returns the nth child reader of the layout.
    fn child(&self, idx: usize) -> &LayoutReader2Ref;

    /// Execute the layout reader for the given range of data, returning a masked array stream.
    ///
    /// TODO(ngates): this bit feels weird to me.
    ///   It's odd that we don't know when a particular reader is done executing. Meaning we don't
    ///   have a good lifetime for cached resources. The returned reader stream _does_ have a good
    ///   lifetime for caching (the duration of the stream), so perhaps we just say that layout
    ///   readers should not hold data and instead each call to execute should make its own segment
    ///   requests? Assuming we can de-dupe also within a segment source, this seems reasonable.
    fn execute(&self, row_range: Range<u64>) -> VortexResult<SendableLayoutReaderStream>;

    /// Attempt to reduce the layout reader to a more simple representation.
    ///
    /// Returns `Ok(None)` if no optimization is possible.
    fn try_reduce(&self) -> VortexResult<Option<LayoutReader2Ref>> {
        _ = self;
        Ok(None)
    }

    /// Attempt to perform a reduction of the parent of this layout reader.
    ///
    /// Returns `Ok(None)` if no reduction is possible.
    fn try_reduce_parent(
        &self,
        parent: &LayoutReader2Ref,
        child_idx: usize,
    ) -> VortexResult<Option<LayoutReader2Ref>> {
        _ = (self, parent, child_idx);
        Ok(None)
    }
}
