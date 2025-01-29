mod eval_expr;
mod eval_stats;
mod reader;
pub mod writer;

use std::collections::BTreeSet;
use std::sync::Arc;

use vortex_array::ContextRef;
use vortex_dtype::FieldMask;
use vortex_error::VortexResult;

use crate::layouts::flat::reader::FlatReader;
use crate::reader::{LayoutReader, LayoutReaderExt};
use crate::segments::AsyncSegmentReader;
use crate::vtable::{LayoutId, LayoutVTable};
use crate::{LayoutData, FLAT_LAYOUT_ID};

#[derive(Debug)]
pub struct FlatLayout;

impl LayoutVTable for FlatLayout {
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
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        for path in field_mask {
            if path.matches_root() {
                splits.insert(row_offset + layout.row_count());
                break;
            }
        }
        Ok(())
    }
}
