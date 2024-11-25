mod chunked;
mod columnar;
mod flat;
mod inline_dtype;
#[cfg(test)]
mod test_read;

pub use chunked::ChunkedLayout;
pub use columnar::ColumnarLayout;
pub use flat::FlatLayout;
pub use inline_dtype::InlineDTypeLayout;

use crate::LayoutReader;

type RangedLayoutReader = ((usize, usize), Box<dyn LayoutReader>);
