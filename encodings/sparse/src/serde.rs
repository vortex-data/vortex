use vortex_array::vtable::SerdeVTable;

use crate::{SparseArray, SparseEncoding};

impl SerdeVTable<&SparseArray> for SparseEncoding {}
