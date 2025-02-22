use crate::arrays::{NullArray, NullEncoding};
use crate::vtable::SerdeVTable;

impl SerdeVTable<&NullArray> for NullEncoding {}
