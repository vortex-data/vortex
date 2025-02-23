use crate::arrays::varbinview::VarBinViewArray;
use crate::arrays::VarBinViewEncoding;
use crate::variants::{BinaryArrayTrait, Utf8ArrayTrait};
use crate::ArrayVariantsImpl;

impl ArrayVariantsImpl for VarBinViewArray {
    fn _as_utf8_typed(&self) -> Option<&dyn Utf8ArrayTrait> {
        Some(self)
    }

    fn _as_binary_typed(&self) -> Option<&dyn BinaryArrayTrait> {
        Some(self)
    }
}

impl Utf8ArrayTrait for VarBinViewArray {}

impl BinaryArrayTrait for VarBinViewArray {}
