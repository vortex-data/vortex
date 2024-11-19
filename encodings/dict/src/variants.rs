use vortex_array::variants::{
    ArrayVariants, BinaryArrayTrait, BoolArrayTrait, PrimitiveArrayTrait, Utf8ArrayTrait,
};
use vortex_array::{ArrayDType, ArrayData};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::DictArray;

impl ArrayVariants for DictArray {
    fn as_bool_array(&self) -> Option<&dyn BoolArrayTrait> {
        matches!(self.dtype(), DType::Bool(..)).then_some(self)
    }

    fn as_primitive_array(&self) -> Option<&dyn PrimitiveArrayTrait> {
        matches!(self.dtype(), DType::Primitive(..)).then_some(self)
    }

    fn as_utf8_array(&self) -> Option<&dyn Utf8ArrayTrait> {
        matches!(self.dtype(), DType::Utf8(..)).then_some(self)
    }

    fn as_binary_array(&self) -> Option<&dyn BinaryArrayTrait> {
        matches!(self.dtype(), DType::Binary(..)).then_some(self)
    }
}

impl BoolArrayTrait for DictArray {
    fn invert(&self) -> VortexResult<ArrayData> {
        todo!()
    }
}

impl PrimitiveArrayTrait for DictArray {}

impl Utf8ArrayTrait for DictArray {}

impl BinaryArrayTrait for DictArray {}
