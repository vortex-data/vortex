mod scan;
pub mod writer;

use vortex_array::ContextRef;

use crate::encoding::{LayoutEncoding, LayoutId};
use crate::layouts::flat::scan::FlatScan;
use crate::scanner::{LayoutScan, Scan};
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
