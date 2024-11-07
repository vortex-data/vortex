use std::any::Any;

use vortex_array::array::ConstantArray;
use vortex_array::{Array, IntoArray};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{unbox_any, VortexExpr};

#[derive(Debug, PartialEq)]
pub struct Literal {
    value: Scalar,
}

impl Literal {
    pub fn new(value: Scalar) -> Self {
        Self { value }
    }

    pub fn value(&self) -> &Scalar {
        &self.value
    }
}

impl VortexExpr for Literal {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, batch: &Array) -> VortexResult<Array> {
        Ok(ConstantArray::new(self.value.clone(), batch.len()).into_array())
    }
}

impl PartialEq<dyn Any> for Literal {
    fn eq(&self, other: &dyn Any) -> bool {
        unbox_any(other)
            .downcast_ref::<Self>()
            .map(|x| x == self)
            .unwrap_or(false)
    }
}
