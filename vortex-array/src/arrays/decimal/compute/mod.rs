mod between;
mod filter;
mod is_constant;
mod is_sorted;
mod min_max;
mod sum;
mod take;

use crate::arrays::DecimalEncoding;
use crate::vtable::ComputeVTable;

impl ComputeVTable for DecimalEncoding {}
