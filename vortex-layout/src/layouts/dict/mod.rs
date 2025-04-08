pub mod writer;
use std::collections::BTreeSet;
use std::sync::Arc;

use vortex_array::ArrayContext;
use vortex_dtype::FieldMask;
use vortex_error::VortexResult;

use crate::segments::SegmentSource;
use crate::{DICT_LAYOUT_ID, Layout, LayoutReader, LayoutVTable};

#[derive(Default, Debug)]
pub struct DictLayout;

impl LayoutVTable for DictLayout {
    fn id(&self) -> crate::LayoutId {
        DICT_LAYOUT_ID
    }

    fn reader(
        &self,
        _layout: Layout,
        _segment_source: &Arc<dyn SegmentSource>,
        _ctx: &ArrayContext,
    ) -> VortexResult<Arc<dyn LayoutReader>> {
        todo!()
    }

    fn register_splits(
        &self,
        _layout: &Layout,
        _field_mask: &[FieldMask],
        _row_offset: u64,
        _splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        // read code ptype from metadata, call register splits to codes child
        todo!();
    }
}
