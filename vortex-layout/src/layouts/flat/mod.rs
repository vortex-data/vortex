mod scan;
mod stats;
pub mod writer;

use std::sync::Arc;

use vortex_array::ContextRef;
use vortex_error::VortexResult;

use crate::encoding::{LayoutEncoding, LayoutId};
use crate::layouts::flat::scan::FlatScan;
use crate::reader::{LayoutReader, LayoutScanExt};
use crate::scanner::Scan;
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
    ) -> VortexResult<Arc<dyn LayoutReader>> {
        Ok(FlatScan::try_new(layout, scan, ctx)?.into_arc())
    }
}
