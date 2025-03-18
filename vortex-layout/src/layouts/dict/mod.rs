pub mod writer;
use crate::{DICT_LAYOUT_ID, LayoutVTable};

#[derive(Default, Debug)]
pub struct DictLayout;

impl LayoutVTable for DictLayout {
    fn id(&self) -> crate::LayoutId {
        DICT_LAYOUT_ID
    }

    fn reader(
        &self,
        layout: crate::Layout,
        ctx: vortex_array::ArrayContext,
        segment_reader: std::sync::Arc<dyn crate::segments::AsyncSegmentReader>,
    ) -> vortex_error::VortexResult<std::sync::Arc<dyn crate::LayoutReader>> {
        todo!()
    }

    fn register_splits(
        &self,
        layout: &crate::Layout,
        field_mask: &[vortex_dtype::FieldMask],
        row_offset: u64,
        splits: &mut std::collections::BTreeSet<u64>,
    ) -> vortex_error::VortexResult<()> {
        // read code ptype from metadata, call register splits to codes child
        todo!();
    }

    fn required_segments(
        &self,
        layout: &crate::Layout,
        row_offset: u64,
        filter_field_mask: &[vortex_dtype::FieldMask],
        projection_field_mask: &[vortex_dtype::FieldMask],
        segments: &mut crate::segments::SegmentCollector,
    ) -> vortex_error::VortexResult<()> {
        // add values segment
        // push down filter & projection to codes
        todo!()
    }
}
