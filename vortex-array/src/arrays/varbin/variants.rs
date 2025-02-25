use crate::ArrayVariantsImpl;
use crate::arrays::varbin::VarBinArray;
use crate::variants::{BinaryArrayTrait, Utf8ArrayTrait};

impl ArrayVariantsImpl for VarBinArray {
    fn _as_utf8_typed(&self) -> Option<&dyn Utf8ArrayTrait> {
        Some(self)
    }

    fn _as_binary_typed(&self) -> Option<&dyn BinaryArrayTrait> {
        Some(self)
    }
}

impl Utf8ArrayTrait for VarBinArray {}

impl BinaryArrayTrait for VarBinArray {}
