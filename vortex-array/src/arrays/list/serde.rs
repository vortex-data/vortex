use crate::arrays::{ListArray, ListEncoding};
use crate::vtable::SerdeVTable;

impl SerdeVTable<&ListArray> for ListEncoding {}
