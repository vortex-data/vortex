use crate::arrays::{ConstantArray, ConstantEncoding};
use crate::vtable::SerdeVTable;

impl SerdeVTable<ConstantArray> for ConstantEncoding {}
