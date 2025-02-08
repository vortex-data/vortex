use std::any::Any;
use std::fmt::Display;
use std::sync::Arc;

use vortex_array::array::ConstantArray;
use vortex_array::{Array, IntoArray};
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{ExprRef, VortexExpr};

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Literal {
    value: Scalar,
}

impl Literal {
    pub fn new_expr(value: impl Into<Scalar>) -> ExprRef {
        Arc::new(Self {
            value: value.into(),
        })
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

    fn unchecked_evaluate(&self, batch: &Array) -> VortexResult<Array> {
        Ok(ConstantArray::new(self.value.clone(), batch.len()).into_array())
    }

    fn children(&self) -> Vec<&ExprRef> {
        vec![]
    }

    fn replacing_children(self: Arc<Self>, children: Vec<ExprRef>) -> ExprRef {
        assert_eq!(children.len(), 0);
        self
    }

    fn return_dtype(&self, _scope_dtype: &DType) -> VortexResult<DType> {
        Ok(self.value.dtype().clone())
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::{DType, Nullability, PType, StructDType};
    use vortex_scalar::Scalar;

    use crate::{lit, test_harness};

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();

        assert_eq!(
            lit(10).return_dtype(&dtype).unwrap(),
            DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_eq!(
            lit(0_u8).return_dtype(&dtype).unwrap(),
            DType::Primitive(PType::U8, Nullability::NonNullable)
        );
        assert_eq!(
            lit(0.0_f32).return_dtype(&dtype).unwrap(),
            DType::Primitive(PType::F32, Nullability::NonNullable)
        );
        assert_eq!(
            lit(i64::MAX).return_dtype(&dtype).unwrap(),
            DType::Primitive(PType::I64, Nullability::NonNullable)
        );
        assert_eq!(
            lit(true).return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
        assert_eq!(
            lit(Scalar::null(DType::Bool(Nullability::Nullable)))
                .return_dtype(&dtype)
                .unwrap(),
            DType::Bool(Nullability::Nullable)
        );

        let sdtype = DType::Struct(
            Arc::new(StructDType::new(
                Arc::from([Arc::from("dog"), Arc::from("cat")]),
                vec![
                    DType::Primitive(PType::U32, Nullability::NonNullable),
                    DType::Utf8(Nullability::NonNullable),
                ],
            )),
            Nullability::NonNullable,
        );
        assert_eq!(
            lit(Scalar::struct_(
                sdtype.clone(),
                vec![Scalar::from(32_u32), Scalar::from("rufus".to_string())]
            ))
            .return_dtype(&dtype)
            .unwrap(),
            sdtype
        );
    }
}
