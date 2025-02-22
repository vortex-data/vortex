use vortex_array::vtable::SerdeVTable;

use crate::{DeltaArray, DeltaEncoding};

impl SerdeVTable<&DeltaArray> for DeltaEncoding {}
