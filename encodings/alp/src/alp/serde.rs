use vortex_array::vtable::SerdeVTable;

use crate::{ALPArray, ALPEncoding};

impl SerdeVTable<&ALPArray> for ALPEncoding {}
