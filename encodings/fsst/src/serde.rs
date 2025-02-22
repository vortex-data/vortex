use vortex_array::vtable::SerdeVTable;

use crate::{FSSTArray, FSSTEncoding};

impl SerdeVTable<&FSSTArray> for FSSTEncoding {}
