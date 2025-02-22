use vortex_array::vtable::SerdeVTable;

use crate::{BitPackedArray, BitPackedEncoding};

impl SerdeVTable<&BitPackedArray> for BitPackedEncoding {}
