mod evaluator;
mod reader;
// mod stats;
pub mod writer;

use std::collections::BTreeSet;
use std::sync::Arc;

use vortex_array::ContextRef;
use vortex_error::VortexResult;

use crate::encoding::{LayoutEncoding, LayoutId};
use crate::layouts::flat::reader::FlatReader;
use crate::reader::{LayoutReader, LayoutScanExt};
use crate::segments::AsyncSegmentReader;
use crate::{LayoutData, FLAT_LAYOUT_ID};

#[derive(Debug)]
pub struct FlatLayout;

impl LayoutEncoding for FlatLayout {
    fn id(&self) -> LayoutId {
        FLAT_LAYOUT_ID
    }

    fn reader(
        &self,
        layout: LayoutData,
        ctx: ContextRef,
        segments: Arc<dyn AsyncSegmentReader>,
    ) -> VortexResult<Arc<dyn LayoutReader>> {
        Ok(FlatReader::try_new(layout, ctx, segments)?.into_arc())
    }

    fn register_splits(
        &self,
        layout: &LayoutData,
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        splits.insert(row_offset + layout.row_count());
        Ok(())
    }
}
