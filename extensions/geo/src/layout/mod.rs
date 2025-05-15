use std::collections::BTreeSet;
use std::sync::Arc;

use vortex_array::ArrayContext;
use vortex_dtype::FieldMask;
use vortex_error::VortexResult;
use vortex_layout::segments::SegmentSource;
use vortex_layout::{Layout, LayoutId, LayoutReader, LayoutVTable};

use crate::layout::reader::BBoxReader;

pub mod reader;
pub mod writer;

pub const ID: LayoutId = LayoutId::new_ref("geovortex.bbox");

#[derive(Debug)]
pub struct BBoxLayout;

impl LayoutVTable for BBoxLayout {
    fn id(&self) -> LayoutId {
        ID
    }

    fn reader(
        &self,
        layout: Layout,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<Arc<dyn LayoutReader>> {
        Ok(Arc::new(BBoxReader::try_new(layout)?))
    }

    fn register_splits(
        &self,
        layout: &Layout,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        todo!()
    }
}
