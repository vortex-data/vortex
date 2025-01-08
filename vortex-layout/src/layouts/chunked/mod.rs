mod evaluator;
mod reader;
// mod stats;
pub mod stats_table;
pub mod writer;

use std::sync::Arc;

use vortex_array::ContextRef;
use vortex_error::VortexResult;

use crate::data::LayoutData;
use crate::encoding::{LayoutEncoding, LayoutId};
use crate::layouts::chunked::reader::ChunkedReader;
use crate::reader::{LayoutReader, LayoutScanExt};
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

    fn reader(&self, layout: LayoutData, ctx: ContextRef) -> VortexResult<Arc<dyn LayoutReader>> {
        Ok(ChunkedReader::try_new(layout, ctx)?.into_arc())
    }
}
