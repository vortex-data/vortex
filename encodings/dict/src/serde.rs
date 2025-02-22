use vortex_array::vtable::SerdeVTable;

use crate::{DictArray, DictEncoding};

impl SerdeVTable<&DictArray> for DictEncoding {}
