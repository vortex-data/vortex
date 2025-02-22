use vortex_array::vtable::SerdeVTable;

use crate::{FoRArray, FoREncoding};

impl SerdeVTable<&FoRArray> for FoREncoding {}
