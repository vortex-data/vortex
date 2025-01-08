mod scan;
pub mod writer;

use std::collections::BTreeSet;
use std::sync::Arc;

use vortex_array::ContextRef;
use vortex_error::VortexResult;

use crate::data::LayoutData;
use crate::encoding::{LayoutEncoding, LayoutId};
use crate::layouts::struct_::scan::StructScan;
use crate::reader::{LayoutReader, LayoutScanExt};
use crate::COLUMNAR_LAYOUT_ID;

#[derive(Debug)]
pub struct StructLayout;

impl LayoutEncoding for StructLayout {
    fn id(&self) -> LayoutId {
        COLUMNAR_LAYOUT_ID
    }

    fn reader(&self, layout: LayoutData, ctx: ContextRef) -> VortexResult<Arc<dyn LayoutReader>> {
        Ok(StructScan::try_new(layout, ctx)?.into_arc())
    }

    fn register_splits(
        &self,
        layout: &LayoutData,
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        for child_idx in 0..layout.nchildren() {
            let child = layout.child(child_idx, layout.dtype().clone())?;
            child.register_splits(row_offset, splits)?;
        }
        Ok(())
    }
}
