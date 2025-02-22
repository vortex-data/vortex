use crate::arrays::{VarBinArray, VarBinEncoding};
use crate::vtable::SerdeVTable;

impl SerdeVTable<VarBinArray> for VarBinEncoding {}
