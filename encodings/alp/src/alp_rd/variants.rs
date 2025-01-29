use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::vtable::VariantsVTable;

use crate::{ALPRDArray, ALPRDEncoding};

impl VariantsVTable<ALPRDArray> for ALPRDEncoding {
    fn as_primitive_array<'a>(&self, array: &'a ALPRDArray) -> Option<&'a dyn PrimitiveArrayTrait> {
        Some(array)
    }
}

impl PrimitiveArrayTrait for ALPRDArray {}
