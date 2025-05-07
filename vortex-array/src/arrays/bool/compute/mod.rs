use crate::arrays::BoolEncoding;
use crate::vtable::ComputeVTable;

mod cast;
mod fill_null;
pub mod filter;
mod flatten;
mod invert;
mod is_constant;
mod is_sorted;
mod mask;
mod min_max;
mod sum;
mod take;

impl ComputeVTable for BoolEncoding {}
