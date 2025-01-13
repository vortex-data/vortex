use std::any::Any;
use std::fmt::Display;
use std::hash::Hash;
use std::sync::Arc;

use vortex_array::{ArrayDType, ArrayData};
use vortex_dtype::Field;
use vortex_error::{vortex_err, VortexResult};

use crate::{ExprRef, VortexExpr};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Column {
    field: Field,
}

impl Column {
    pub fn new_expr(field: impl Into<Field>) -> ExprRef {
        Arc::new(Self {
            field: field.into(),
        })
    }

    pub fn field(&self) -> &Field {
        &self.field
    }
}

pub fn col(field: impl Into<Field>) -> ExprRef {
    Column::new_expr(field)
}

impl From<String> for Column {
    fn from(value: String) -> Self {
        Column {
            field: value.into(),
        }
    }
}

impl From<usize> for Column {
    fn from(value: usize) -> Self {
        Column {
            field: value.into(),
        }
    }
}

impl Display for Column {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.field)
    }
}

impl VortexExpr for Column {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn unchecked_evaluate(&self, batch: &ArrayData) -> VortexResult<ArrayData> {
        batch
            .clone()
            .as_struct_array()
            .ok_or_else(|| {
                vortex_err!(
                    "Array must be a struct array, however it was a {}",
                    batch.dtype()
                )
            })?
            .field(&self.field)?
            .ok_or_else(|| vortex_err!("Array doesn't contain child array {}", self.field))
    }

    fn children(&self) -> Vec<&ExprRef> {
        vec![]
    }

    fn replacing_children(self: Arc<Self>, children: Vec<ExprRef>) -> ExprRef {
        assert_eq!(children.len(), 0);
        self
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::array::{BoolArray, PrimitiveArray, StructArray};
    use vortex_array::compute::scalar_at;
    use vortex_array::validity::{ArrayValidity as _, Validity};
    use vortex_array::{ArrayDType as _, IntoArrayData as _};
    use vortex_dtype::{DType, FieldNames, Nullability, PType};
    use vortex_scalar::Scalar;

    use crate::{col, test_harness};

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();
        assert_eq!(
            col("a").return_dtype(&dtype).unwrap(),
            DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_eq!(
            col(1).return_dtype(&dtype).unwrap(),
            DType::Primitive(PType::U16, Nullability::Nullable)
        );
    }

    #[test]
    fn evaluate_with_nulls() {
        let a = PrimitiveArray::from_option_iter([Some(0_i32), None, None, Some(3), Some(4)])
            .into_array();
        let array = StructArray::try_new(
            FieldNames::from(["a".into()]),
            vec![a],
            5,
            Validity::Array(BoolArray::from_iter([true, false, true, false, true]).into_array()),
        )
        .unwrap()
        .into_array();

        let a_result = col("a").evaluate(&array).unwrap();

        assert_eq!(
            a_result.dtype(),
            &DType::Primitive(PType::I32, Nullability::Nullable)
        );
        assert_eq!(scalar_at(&a_result, 0).unwrap(), Scalar::from(Some(0_i32)));
        assert!(!a_result.is_valid(1));
        assert!(!a_result.is_valid(2));
        assert!(!a_result.is_valid(3));
        assert_eq!(scalar_at(&a_result, 4).unwrap(), Scalar::from(Some(4_i32)));
    }
}
