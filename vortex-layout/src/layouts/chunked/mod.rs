mod scan;
pub mod stats;

use bytes::Bytes;
use vortex_array::ContextRef;
use vortex_dtype::DType;

use crate::data::LayoutData;
use crate::encoding::{LayoutEncoding, LayoutId};
use crate::layouts::chunked::scan::ChunkedScan;
use crate::scanner::{LayoutScan, Scan};
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
    fn scan(&self, layout: LayoutData, scan: Scan, ctx: ContextRef) -> Box<dyn LayoutScan> {
        Box::new(ChunkedScan::new(layout, scan, ctx)) as _
    }
}

impl ChunkedLayout {
    /// Create a new chunked layout with the given row count and children.
    pub fn new(
        dtype: DType,
        row_count: u64,
        children: Vec<LayoutData>,
        metadata: Option<Bytes>,
    ) -> LayoutData {
        LayoutData::new_owned(
            &ChunkedLayout,
            dtype,
            row_count,
            None,
            Some(children),
            metadata,
        )
    }
}
