use crate::arrays::{VarBinViewArray, VarBinViewEncoding};
use crate::vtable::SerdeVTable;

impl SerdeVTable<VarBinViewArray> for VarBinViewEncoding {}
