mod chunked;
mod columnar;
mod flat;
#[cfg(test)]
mod test_read;

pub use chunked::ChunkedLayout;
pub use columnar::ColumnarLayout;
pub use flat::FlatLayout;

use crate::LayoutReader;

type RangedLayoutReader = ((usize, usize), Box<dyn LayoutReader>);
