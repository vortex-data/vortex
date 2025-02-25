use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::ArrayVariantsImpl;

use crate::ALPRDArray;

impl ArrayVariantsImpl for ALPRDArray {
    fn _as_primitive_typed(&self) -> Option<&dyn PrimitiveArrayTrait> {
        Some(self)
    }
}

impl PrimitiveArrayTrait for ALPRDArray {}
