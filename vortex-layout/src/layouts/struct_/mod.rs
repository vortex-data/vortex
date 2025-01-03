mod scan;

use vortex_array::ContextRef;
use vortex_dtype::DType;

use crate::data::LayoutData;
use crate::encoding::{LayoutEncoding, LayoutId};
use crate::layouts::struct_::scan::StructScan;
use crate::scanner::{LayoutScan, Scan};
use crate::COLUMNAR_LAYOUT_ID;

#[derive(Debug)]
pub struct StructLayout;

impl LayoutEncoding for StructLayout {
    fn id(&self) -> LayoutId {
        COLUMNAR_LAYOUT_ID
    }

    fn scan(&self, layout: LayoutData, scan: Scan, ctx: ContextRef) -> Box<dyn LayoutScan> {
        Box::new(StructScan::new(layout, scan, ctx)) as _
    }
}

impl StructLayout {
    /// Create a new columnar layout with the given row count and field layouts.
    pub fn new(row_count: u64, dtype: DType, fields: Vec<LayoutData>) -> LayoutData {
        LayoutData::new_owned(&StructLayout, dtype, row_count, None, Some(fields), None)
    }
}
