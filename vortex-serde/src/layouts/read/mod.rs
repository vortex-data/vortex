use std::collections::BTreeSet;
use std::fmt::Debug;
use std::sync::Arc;

use vortex::Array;
use vortex_error::VortexResult;

mod batch;
mod buffered;
mod builder;
mod cache;
mod context;
mod expr_project;
mod filtering;
mod footer;
mod layouts;
mod mask;
mod recordbatchreader;
mod stream;

pub use builder::LayoutReaderBuilder;
pub use cache::*;
pub use context::*;
pub use filtering::RowFilter;
pub use footer::LayoutDescriptorReader;
pub use recordbatchreader::{AsyncRuntime, VortexRecordBatchReader};
pub use stream::LayoutBatchStream;
use vortex_expr::VortexExpr;
pub use vortex_schema::projection::Projection;
pub use vortex_schema::Schema;

use crate::layouts::read::mask::RowMask;
use crate::stream_writer::ByteRange;

// Recommended read-size according to the AWS performance guide
pub const INITIAL_READ_SIZE: usize = 8 * 1024 * 1024;

/// Operation to apply to data returned by the layout
#[derive(Debug, Clone)]
pub struct Scan {
    expr: Option<Arc<dyn VortexExpr>>,
}

impl Scan {
    pub fn new(expr: Option<Arc<dyn VortexExpr>>) -> Self {
        Self { expr }
    }
}

/// Unique identifier for a message within a layout
pub type LayoutPartId = u16;
/// Path through layout tree to given message
pub type MessageId = Vec<LayoutPartId>;
/// ID and Range of atomic element of the file
pub type Message = (MessageId, ByteRange);

#[derive(Debug)]
pub enum ReadResult {
    ReadMore(Vec<Message>),
    Batch(Array),
}

pub trait LayoutReader: Debug + Send {
    /// Register all horizontal boundaries of this layout.
    ///
    /// Layout should register its begging adjusted by row_offset and do so for all of its children
    fn add_splits(&self, row_offset: usize, splits: &mut BTreeSet<usize>) -> VortexResult<()>;

    /// Reads the data from the underlying layout within given selection
    ///
    /// Layout is required to return all data for given selection in one batch.
    /// Layout can either return a batch data, i.e. an Array or ask for more layout messages to
    /// be read. When requesting messages to be read the caller should populate the message cache used
    /// when creating instance of this trait and then call back into this function.
    ///
    /// The layout is finished producing data for selection when it returns None
    fn read_selection(&mut self, selector: RowMask) -> VortexResult<Option<ReadResult>>;
}
