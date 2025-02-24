use vortex_array::ArrayVariantsImpl;
use vortex_array::variants::{
    BinaryArrayTrait, BoolArrayTrait, PrimitiveArrayTrait, Utf8ArrayTrait,
};

use crate::DictArray;

impl ArrayVariantsImpl for DictArray {
    fn _as_bool_typed(&self) -> Option<&dyn BoolArrayTrait> {
        Some(self)
    }

    fn _as_primitive_typed(&self) -> Option<&dyn PrimitiveArrayTrait> {
        Some(self)
    }

    fn _as_utf8_typed(&self) -> Option<&dyn Utf8ArrayTrait> {
        Some(self)
    }

    fn _as_binary_typed(&self) -> Option<&dyn BinaryArrayTrait> {
        Some(self)
    }
}

impl BoolArrayTrait for DictArray {}

impl PrimitiveArrayTrait for DictArray {}

impl Utf8ArrayTrait for DictArray {}

impl BinaryArrayTrait for DictArray {}
