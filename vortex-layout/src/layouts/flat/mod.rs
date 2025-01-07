mod scan;
pub mod writer;

use std::sync::Arc;

use vortex_array::ContextRef;
use vortex_error::VortexResult;

use crate::encoding::{LayoutEncoding, LayoutId};
use crate::layouts::flat::scan::FlatScan;
use crate::scanner::{LayoutScan, LayoutScanExt, Scan};
use crate::{LayoutData, FLAT_LAYOUT_ID};

#[derive(Debug)]
pub struct FlatLayout;

impl LayoutEncoding for FlatLayout {
    fn id(&self) -> LayoutId {
        FLAT_LAYOUT_ID
    }

    fn scan(
        &self,
        layout: LayoutData,
        scan: Scan,
        ctx: ContextRef,
    ) -> VortexResult<Arc<dyn LayoutScan>> {
        Ok(FlatScan::try_new(layout, scan, ctx)?.into_arc())
    }
}
