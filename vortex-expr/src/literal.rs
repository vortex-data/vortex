use std::any::Any;
use std::fmt::Display;
use std::sync::Arc;

use vortex_array::array::ConstantArray;
use vortex_array::{ArrayData, IntoArrayData};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{ExprRef, VortexExpr};

#[derive(Debug, PartialEq, Eq)]
pub struct Literal {
    value: Scalar,
}

impl Literal {
    pub fn new_expr(value: Scalar) -> ExprRef {
        Arc::new(Self { value })
    }

    pub fn value(&self) -> &Scalar {
        &self.value
    }
}

impl Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl VortexExpr for Literal {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, batch: &ArrayData) -> VortexResult<ArrayData> {
        Ok(ConstantArray::new(self.value.clone(), batch.len()).into_array())
    }
}

/// Create a new `Literal` expression from a type that coerces to `Scalar`.
///
///
/// ## Example usage
///
/// ```
/// use vortex_array::array::PrimitiveArray;
/// use vortex_dtype::Nullability;
/// use vortex_expr::{lit, Literal};
/// use vortex_scalar::Scalar;
///
/// let number = lit(34i32);
///
/// let literal = number.as_any()
///     .downcast_ref::<Literal>()
///     .unwrap();
/// assert_eq!(literal.value(), &Scalar::primitive(34i32, Nullability::NonNullable));
/// ```
pub fn lit(value: impl Into<Scalar>) -> ExprRef {
    Literal::new_expr(value.into())
}
