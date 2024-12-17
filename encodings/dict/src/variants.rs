use vortex_array::variants::{
    BinaryArrayTrait, BoolArrayTrait, PrimitiveArrayTrait, Utf8ArrayTrait, VariantsVTable,
};

use crate::{DictArray, DictEncoding};

impl VariantsVTable<DictArray> for DictEncoding {
    fn as_bool_array<'a>(&self, array: &'a DictArray) -> Option<&'a dyn BoolArrayTrait> {
        Some(array)
    }

    fn as_primitive_array<'a>(&self, array: &'a DictArray) -> Option<&'a dyn PrimitiveArrayTrait> {
        Some(array)
    }

    fn as_utf8_array<'a>(&self, array: &'a DictArray) -> Option<&'a dyn Utf8ArrayTrait> {
        Some(array)
    }

    fn as_binary_array<'a>(&self, array: &'a DictArray) -> Option<&'a dyn BinaryArrayTrait> {
        Some(array)
    }
}

impl BoolArrayTrait for DictArray {}

impl PrimitiveArrayTrait for DictArray {}

impl Utf8ArrayTrait for DictArray {}

impl BinaryArrayTrait for DictArray {}
