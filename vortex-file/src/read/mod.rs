use std::collections::BTreeSet;
use std::fmt::Debug;

use vortex_array::ArrayData;
use vortex_error::VortexResult;

mod buffered;
pub mod builder;
mod cache;
mod context;
mod expr_project;
mod filtering;
pub mod layouts;
mod mask;
pub mod metadata;
pub mod projection;
mod reader;
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
use vortex_buffer::Buffer;
use vortex_expr::ExprRef;

use crate::byte_range::ByteRange;
pub use crate::read::mask::RowMask;

// Recommended read-size according to the AWS performance guide
// FIXME(ngates): this is dumb
pub const INITIAL_READ_SIZE: usize = 8 * 1024 * 1024;

/// Operation to apply to data returned by the layout
#[derive(Debug, Clone)]
pub struct Scan {
    expr: Option<ExprRef>,
}

impl Scan {
    pub fn empty() -> Self {
        Self { expr: None }
    }

    pub fn new(expr: ExprRef) -> Self {
        Self { expr: Some(expr) }
    }
}

impl From<Option<ExprRef>> for Scan {
    fn from(expr: Option<ExprRef>) -> Self {
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
/// A message that has had its bytes materialized onto the heap.
#[derive(Debug, Clone)]
pub struct Message(pub MessageId, pub Buffer);

/// A polling interface for reading a value from a [`LayoutReader`].
#[derive(Debug)]
pub enum PollRead<T> {
    ReadMore(Vec<MessageLocator>),
    Value(T),
}

/// Result type for an attempt to prune rows from a [`LayoutReader`].
///
/// The default value is `CannotPrune` so that layouts which do not implement pruning can default
/// to performing full scans of their data.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prune {
    /// It is unsafe for the layout to prune the requested row range.
    #[default]
    CannotPrune,
    /// It is safe for the layout to prune the requested row range.
    CanPrune,
}

/// A reader for a layout, a serialized sequence of Vortex arrays.
///
/// Some layouts are _horizontally divisible_: they can read a sub-sequence of rows independently of
/// other sub-sequences. A layout advertises its subdivisions in its [add_splits][Self::add_splits]
/// method. Any layout which is or contains a chunked layout is horizontally divisible.
///
/// The [poll_read][Self::poll_read] method accepts and applies a [RowMask], reading only
/// the subdivisions which contain the selected rows.
///
/// # State management
///
/// Layout readers are **synchronous** and **stateful**. A request to read a given row range may
/// trigger a request for more messages, which will be handled by the caller, placing the messages
/// back into the message cache for this layout as a result.
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
    fn poll_read(&self, selector: &RowMask) -> VortexResult<Option<PollRead<ArrayData>>>;

    /// Reads the metadata of the layout, if it exists.
    ///
    /// `LayoutReader`s can override the default behavior, which is to return no metadata.
    fn poll_metadata(&self) -> VortexResult<Option<PollRead<Vec<Option<ArrayData>>>>> {
        Ok(None)
    }

    /// Introspect to determine if we can prune the given [begin, end) row range.
    ///
    /// `LayoutReader`s can opt out of the default implementation, which is to not prune.
    fn poll_prune(&self, _begin: usize, _end: usize) -> VortexResult<PollRead<Prune>> {
        Ok(PollRead::Value(Prune::CannotPrune))
    }
}
