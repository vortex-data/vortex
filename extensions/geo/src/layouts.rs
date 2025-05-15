use std::collections::BTreeSet;
use std::sync::Arc;

use vortex_array::ArrayContext;
use vortex_dtype::{DType, FieldMask};
use vortex_error::{VortexResult, vortex_assert};
use vortex_layout::segments::SegmentSource;
use vortex_layout::{
    Layout, LayoutId, LayoutReader, LayoutRegistry, LayoutVTable, LayoutVTableRef,
};

/// Special layout that stores a bounding box for every geometry in a
/// file to make filtering easier.

impl LayoutVTable for BBoxLayout {
    fn id(&self) -> LayoutId {
        LayoutId::new_ref("geovortex.flat")
    }

    fn reader(
        &self,
        layout: Layout,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<Arc<dyn LayoutReader>> {
        vortex_assert!(
            layout.vtable().id() == self.id(),
            "Invalid layout ID for bbox {}",
            layout.vtable().id()
        );
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
