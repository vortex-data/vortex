use crate::array::varbin::VarBinArray;
use crate::array::VarBinEncoding;
use crate::variants::{BinaryArrayTrait, Utf8ArrayTrait, VariantsVTable};

impl VariantsVTable<VarBinArray> for VarBinEncoding {
    fn as_utf8_array<'a>(&self, array: &'a VarBinArray) -> Option<&'a dyn Utf8ArrayTrait> {
        Some(array)
    }

    fn as_binary_array<'a>(&self, array: &'a VarBinArray) -> Option<&'a dyn BinaryArrayTrait> {
        Some(array)
    }
}

impl Utf8ArrayTrait for VarBinArray {}

impl BinaryArrayTrait for VarBinArray {}
