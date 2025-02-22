use vortex_array::vtable::SerdeVTable;

use crate::{ByteBoolArray, ByteBoolEncoding};

impl SerdeVTable<&ByteBoolArray> for ByteBoolEncoding {}
