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
    fn row_count(&self) -> u64;

    /// Returns the [`DType`] of the layout.
    fn dtype(&self) -> &DType;

    /// Returns the number of child layouts.
    fn nchildren(&self) -> usize;

    /// Returns the nth child reader of the layout.
    fn child(&self, idx: usize) -> &LayoutReader2Ref;

    /// Execute the layout reader for the given range of data, returning a masked array stream.
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
