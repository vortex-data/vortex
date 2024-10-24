use std::fmt::Debug;
use std::sync::Arc;

use vortex::Array;
use vortex_error::VortexResult;

mod batch;
mod buffered;
mod builder;
mod cache;
mod context;
mod filter_project;
mod filtering;
mod footer;
mod layouts;
mod recordbatchreader;
mod selection;
mod stream;

pub use builder::LayoutReaderBuilder;
pub use cache::LayoutMessageCache;
pub use context::*;
pub use filtering::RowFilter;
pub use footer::LayoutDescriptorReader;
pub use recordbatchreader::{AsyncRuntime, VortexRecordBatchReader};
pub use stream::LayoutBatchStream;
use vortex_expr::VortexExpr;
pub use vortex_schema::projection::Projection;
pub use vortex_schema::Schema;

use crate::layouts::read::selection::RowSelector;
use crate::stream_writer::ByteRange;

// Recommended read-size according to the AWS performance guide
pub const INITIAL_READ_SIZE: usize = 8 * 1024 * 1024;

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
pub type MessageId = Vec<LayoutPartId>;
pub type Message = (MessageId, ByteRange);

#[derive(Debug)]
pub enum ReadResult {
    ReadMore(Vec<Message>),
    Batch(Array),
}

#[derive(Debug)]
pub enum RangeResult {
    ReadMore(Vec<Message>),
    Rows(Option<RowSelector>),
}

pub trait LayoutReader: Debug + Send {
    /// Produce sets of row ranges to read from underlying layouts.
    ///
    /// Empty RangeResult indicates layout is done producing ranges
    fn next_range(&mut self) -> VortexResult<RangeResult>;

    /// Reads the data from the underlying layout
    ///
    /// The layout can either return a batch data, i.e. an Array or ask for more layout messages to
    /// be read. When requesting messages to be read the caller should populate the messages in the cache
    /// and then call back into this function.
    ///
    /// The layout is finished reading when it returns None
    fn read_next(&mut self, selector: RowSelector) -> VortexResult<Option<ReadResult>>;

    /// Advance readers to global row offset
    fn advance(&mut self, up_to_row: usize) -> VortexResult<Vec<Message>>;
}
