mod eval_expr;
mod reader;
pub mod writer;
use std::collections::BTreeSet;
use std::sync::Arc;

use reader::DictReader;
use vortex_array::ArrayContext;
use vortex_dtype::FieldMask;
use vortex_error::VortexResult;

use crate::segments::SegmentSource;
use crate::{DICT_LAYOUT_ID, Layout, LayoutReader, LayoutReaderExt as _, LayoutVTable};

#[derive(Default, Debug)]
pub struct DictLayout;

impl LayoutVTable for DictLayout {
    fn id(&self) -> crate::LayoutId {
        DICT_LAYOUT_ID
    }

    fn reader(
        &self,
        layout: Layout,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<Arc<dyn LayoutReader>> {
        Ok(DictReader::try_new(layout, segment_source, ctx)?.into_arc())
    }

    fn register_splits(
        &self,
        layout: &Layout,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        layout
            .child(1, layout.dtype().clone(), "codes")?
            .register_splits(field_mask, row_offset, splits)
    }
}
