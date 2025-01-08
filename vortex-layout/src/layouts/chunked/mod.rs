mod eval;
mod scan;
mod stats;
pub mod stats_table;
pub mod writer;

use std::sync::Arc;

use vortex_array::ContextRef;
use vortex_error::VortexResult;

use crate::data::LayoutData;
use crate::encoding::{LayoutEncoding, LayoutId};
use crate::layouts::chunked::scan::ChunkedReader;
use crate::reader::{LayoutReader, LayoutScanExt};
use crate::scanner::Scan;
use crate::CHUNKED_LAYOUT_ID;

#[derive(Default, Debug)]
pub struct ChunkedLayout;

/// In-memory representation of Chunked layout.
///
/// First child in the list is the metadata table
/// Subsequent children are consecutive chunks of this layout
impl LayoutEncoding for ChunkedLayout {
    fn id(&self) -> LayoutId {
        CHUNKED_LAYOUT_ID
    }

    // TODO(ngates): we probably need some reader options that we can downcast here? But how does
    //  the user configure the tree of readers? e.g. batch size
    fn scan(
        &self,
        layout: LayoutData,
        scan: Scan,
        ctx: ContextRef,
    ) -> VortexResult<Arc<dyn LayoutReader>> {
        Ok(ChunkedReader::try_new(layout, scan, ctx)?.into_arc())
    }
}
