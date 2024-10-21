use std::fmt::Debug;
use std::sync::Arc;

use arrow_buffer::BooleanBuffer;
use vortex::array::BoolArray;
use vortex::validity::Validity;
use vortex::{Array, IntoArray as _, IntoArrayVariant as _};
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
    Selector(RowSelector),
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
    /// Range ending at the end of the layout length indicates layout is done producing ranges
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

pub fn null_as_false(array: BoolArray) -> VortexResult<Array> {
    Ok(match array.validity() {
        Validity::NonNullable => array.into_array(),
        Validity::AllValid => {
            BoolArray::try_new(array.boolean_buffer(), Validity::NonNullable)?.into_array()
        }
        Validity::AllInvalid => BoolArray::from(BooleanBuffer::new_unset(array.len())).into_array(),
        Validity::Array(v) => {
            let bool_buffer = &array.boolean_buffer() & &v.into_bool()?.boolean_buffer();
            BoolArray::from(bool_buffer).into_array()
        }
    })
}

#[cfg(test)]
mod tests {
    use vortex::array::BoolArray;
    use vortex::validity::Validity;
    use vortex::IntoArrayVariant;

    use super::*;

    #[test]
    fn coerces_nulls() {
        let bool_array = BoolArray::from_vec(
            vec![true, true, false, false],
            Validity::Array(BoolArray::from(vec![true, false, true, false]).into()),
        );
        let non_null_array = null_as_false(bool_array).unwrap().into_bool().unwrap();
        assert_eq!(
            non_null_array.boolean_buffer().iter().collect::<Vec<_>>(),
            vec![true, false, false, false]
        );
    }
}
