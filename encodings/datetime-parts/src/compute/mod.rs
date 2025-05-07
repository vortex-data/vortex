mod cast;
mod compare;
mod filter;
mod is_constant;
mod take;

use vortex_array::vtable::ComputeVTable;

use crate::DateTimePartsEncoding;

impl ComputeVTable for DateTimePartsEncoding {}
