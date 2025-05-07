use crate::arrays::ChunkedEncoding;
use crate::vtable::ComputeVTable;

mod cast;
mod compare;
mod elementwise;
mod fill_null;
mod filter;
mod invert;
mod is_constant;
mod is_sorted;
mod mask;
mod min_max;
mod sum;
mod take;

impl ComputeVTable for ChunkedEncoding {}
