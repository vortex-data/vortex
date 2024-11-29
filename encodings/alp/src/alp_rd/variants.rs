use vortex_array::variants::{PrimitiveArrayTrait, VariantsVTable};

use crate::{ALPRDArray, ALPRDEncoding};

impl VariantsVTable<ALPRDArray> for ALPRDEncoding {
    fn as_primitive_array<'a>(&self, array: &'a ALPRDArray) -> Option<&'a dyn PrimitiveArrayTrait> {
        Some(array)
    }
}

impl PrimitiveArrayTrait for ALPRDArray {}
