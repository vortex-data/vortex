mod scan;

use vortex_array::ContextRef;
use vortex_dtype::DType;

use crate::encoding::{LayoutEncoding, LayoutId};
use crate::layouts::flat::scan::FlatScan;
use crate::scanner::{LayoutScan, Scan};
use crate::segments::SegmentId;
use crate::{LayoutData, FLAT_LAYOUT_ID};

#[derive(Debug)]
pub struct FlatLayout;

impl LayoutEncoding for FlatLayout {
    fn id(&self) -> LayoutId {
        FLAT_LAYOUT_ID
    }

    fn scan(&self, layout: LayoutData, scan: Scan, ctx: ContextRef) -> Box<dyn LayoutScan> {
        Box::new(FlatScan::new(layout, scan, ctx)) as _
    }
}

impl FlatLayout {
    /// Create a new flat layout with the given row count and segment id.
    pub fn new(dtype: DType, row_count: u64, segment_id: SegmentId) -> LayoutData {
        LayoutData::new_owned(
            &FlatLayout,
            dtype,
            row_count,
            Some(vec![segment_id]),
            None,
            None,
        )
    }
}
