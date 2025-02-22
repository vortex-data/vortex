use crate::arrays::{StructArray, StructEncoding};
use crate::vtable::SerdeVTable;

impl SerdeVTable<StructArray> for StructEncoding {}
