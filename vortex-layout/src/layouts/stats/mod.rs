mod eval_expr;
mod reader;
pub mod stats_table;
pub mod writer;

use std::collections::BTreeSet;
use std::sync::Arc;

use vortex_array::ArrayContext;
use vortex_dtype::FieldMask;
use vortex_error::VortexResult;

use crate::data::LayoutData;
use crate::layouts::stats::reader::StatsReader;
use crate::reader::LayoutReader;
use crate::segments::SegmentSource;
use crate::{LayoutId, STATS_LAYOUT_ID};

#[derive(Default, Debug)]
pub struct StatsLayout;

/// First child contains the data, second child contains the statistics table.
impl LayoutVTable for StatsLayout {
    fn id(&self) -> LayoutId {
        STATS_LAYOUT_ID
    }

    fn reader(
        &self,
        layout: LayoutData,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<Arc<dyn LayoutReader>> {
        Ok(StatsReader::try_new(layout, segment_source, ctx)?.into_arc())
    }

    fn register_splits(
        &self,
        layout: &LayoutData,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        layout
            .child(0, layout.dtype().clone(), "data")?
            .register_splits(field_mask, row_offset, splits)
    }
}
