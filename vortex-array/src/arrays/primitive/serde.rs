use crate::arrays::{PrimitiveArray, PrimitiveEncoding};
use crate::vtable::SerdeVTable;

impl SerdeVTable<&PrimitiveArray> for PrimitiveEncoding {}
