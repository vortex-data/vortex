// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::expr::Expression;
use vortex_array::MaskFuture;
use vortex_dtype::DType;
use vortex_dtype::FieldMask;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::layouts::rtree::RTreeLayout;
use crate::ArrayFuture;
use crate::LayoutReader;
use crate::LayoutReaderRef;
use crate::LazyReaderChildren;

pub struct RTreeReader {
    pub(crate) name: Arc<str>,
    pub(crate) layout: RTreeLayout,
    pub(crate) children: LazyReaderChildren,
}

impl RTreeReader {
    fn data_child(&self) -> VortexResult<&LayoutReaderRef> {
        self.children.get(0)
    }
}

impl LayoutReader for RTreeReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        &self.layout.dtype
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_range: &Range<u64>,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        // Register splits from the data.
        self.children
            .get(0)?
            .register_splits(field_mask, row_range, splits)?;

        Ok(())
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        // TODO(aduffy): if we have an ST_Contains expression, scan the RTree first to see what we
        //  can prune. If we get anything, merge it instead.
        self.data_child()?.pruning_evaluation(row_range, expr, mask)
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        // Let the child handle it, like ZonedReader
        self.data_child()?.filter_evaluation(row_range, expr, mask)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        // TODO(aduffy): can we do anything better here?
        self.data_child()?
            .projection_evaluation(row_range, expr, mask)
    }
}
