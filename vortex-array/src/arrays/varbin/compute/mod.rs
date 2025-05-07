pub use min_max::compute_min_max;

use crate::arrays::VarBinEncoding;
use crate::vtable::ComputeVTable;

mod cast;
mod compare;
mod filter;
mod is_constant;
mod is_sorted;
mod mask;
mod min_max;
mod take;

impl ComputeVTable for VarBinEncoding {}
