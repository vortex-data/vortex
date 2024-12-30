mod chunked;
mod columnar;
mod flat;
#[cfg(test)]
mod test_read;

use std::sync::Arc;

pub use chunked::ChunkedLayout;
pub use columnar::ColumnarLayout;
pub use flat::FlatLayout;

use crate::LayoutReader;

// TODO(aduffy): make this container more useful
#[derive(Debug)]
pub struct RangedLayoutReader((usize, usize), Arc<dyn LayoutReader>);
