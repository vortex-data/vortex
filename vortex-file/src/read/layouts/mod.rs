mod chunked;
mod columnar;
mod flat;
mod inline_dtype;
#[cfg(test)]
mod test_read;

use std::sync::RwLock;

pub use chunked::ChunkedLayout;
pub use columnar::ColumnarLayout;
pub use flat::FlatLayout;
pub use inline_dtype::InlineDTypeLayout;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::ArrayData;

use crate::LayoutReader;

type RangedLayoutReader = ((usize, usize), Box<dyn LayoutReader>);
type InProgressRanges = RwLock<HashMap<(usize, usize), Vec<Option<ArrayData>>>>;
