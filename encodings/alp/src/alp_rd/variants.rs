use vortex_array::variants::{ArrayVariants, PrimitiveArrayTrait};

use crate::ALPRDArray;

impl ArrayVariants for ALPRDArray {
    fn as_primitive_array(&self) -> Option<&dyn PrimitiveArrayTrait> {
        Some(self)
    }

    fn as_primitive_array_unchecked(&self) -> &dyn PrimitiveArrayTrait {
        self
    }
}

impl PrimitiveArrayTrait for ALPRDArray {}
