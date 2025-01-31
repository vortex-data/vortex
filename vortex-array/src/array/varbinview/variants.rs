use crate::array::varbinview::VarBinViewArray;
use crate::array::VarBinViewEncoding;
use crate::variants::{BinaryArrayTrait, Utf8ArrayTrait};
use crate::vtable::VariantsVTable;

impl VariantsVTable<VarBinViewArray> for VarBinViewEncoding {
    fn as_utf8_array<'a>(&self, array: &'a VarBinViewArray) -> Option<&'a dyn Utf8ArrayTrait> {
        Some(array)
    }

    fn as_binary_array<'a>(&self, array: &'a VarBinViewArray) -> Option<&'a dyn BinaryArrayTrait> {
        Some(array)
    }
}

impl Utf8ArrayTrait for VarBinViewArray {}

impl BinaryArrayTrait for VarBinViewArray {}
