use std::collections::BTreeSet;
use std::fmt::Debug;

use vortex_array::ArrayData;
use vortex_error::VortexResult;

pub mod builder;
mod cache;
mod context;
mod expr_project;
mod filtering;
pub mod layouts;
mod mask;
pub mod projection;
mod recordbatchreader;
mod splits;
mod stream;

pub use builder::initial_read::*;
pub use builder::VortexReadBuilder;
pub use cache::*;
pub use context::*;
pub use filtering::RowFilter;
pub use projection::Projection;
pub use recordbatchreader::{AsyncRuntime, VortexRecordBatchReader};
pub use stream::VortexFileArrayStream;
use vortex_expr::ExprRef;
use vortex_ipc::stream_writer::ByteRange;

pub use crate::read::mask::RowMask;

// Recommended read-size according to the AWS performance guide
pub const INITIAL_READ_SIZE: usize = 8 * 1024 * 1024;

/// Operation to apply to data returned by the layout
#[derive(Debug, Clone)]
pub struct Scan {
    expr: Option<ExprRef>,
}

impl Scan {
    pub fn new(expr: Option<ExprRef>) -> Self {
        Self { expr }
    }
}

/// Unique identifier for a message within a layout
pub type LayoutPartId = u16;
/// Path through layout tree to given message
pub type MessageId = Vec<LayoutPartId>;
/// A unique locator for a message, including its ID and byte range containing
/// the message contents.
#[derive(Debug, Clone)]
pub struct MessageLocator(pub MessageId, pub ByteRange);

#[derive(Debug)]
pub enum BatchRead {
    ReadMore(Vec<MessageLocator>),
    Batch(ArrayData),
}

/// A reader for a layout, a serialized sequence of Vortex arrays.
///
/// Some layouts are _horizontally divisble_: they can read a sub-sequence of rows independently of
/// other sub-sequences. A layout advertises its sub-divisions in its [add_splits][Self::add_splits]
/// method. Any layout which is or contains a chunked layout is horizontally divisble.
///
/// The [read_selection][Self::read_selection] method accepts and applies a [RowMask], reading only
/// the sub-divisions which contain the selected (i.e. masked) rows.
pub trait LayoutReader: Debug + Send {
    /// Register all horizontal row boundaries of this layout.
    ///
    /// Layout should register all indivisible absolute row boundaries of the data stored in itself and its children.
    /// `row_offset` gives the relative row position of this layout to the beginning of the file.
    fn add_splits(&self, row_offset: usize, splits: &mut BTreeSet<usize>) -> VortexResult<()>;

    /// Reads the data from the underlying layout within given selection
    ///
    /// Layout is required to return all data for given selection in one batch.  Layout can either
    /// return a batch of data (i.e., an Array) or ask for more layout messages to be read. When
    /// requesting messages to be read the caller should populate the message cache used when
    /// creating the invoked instance of this trait and then call back into this function.
    ///
    /// The layout is finished producing data for selection when it returns None
    fn read_selection(&self, selector: &RowMask) -> VortexResult<Option<BatchRead>>;
}
