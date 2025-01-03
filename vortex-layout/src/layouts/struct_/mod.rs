mod scan;
pub mod writer;

use vortex_array::ContextRef;

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
