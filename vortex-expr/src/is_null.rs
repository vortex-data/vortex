use std::any::Any;
use std::fmt::Display;
use std::ops::Not;
use std::sync::Arc;

use vortex_array::arrays::{BoolArray, ConstantArray};
use vortex_array::{Array, ArrayRef, IntoArray};
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;

use crate::{ExprRef, VortexExpr};

#[derive(Debug, Eq, Hash)]
#[allow(clippy::derived_hash_with_manual_eq)]
pub struct IsNull {
    child: ExprRef,
}

impl IsNull {
    pub fn new_expr(child: ExprRef) -> ExprRef {
        Arc::new(Self { child })
    }
}

impl PartialEq for IsNull {
    fn eq(&self, other: &Self) -> bool {
        self.child.eq(&other.child)
    }
}

impl Display for IsNull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "is_null({})", self.child)
    }
}

#[cfg(feature = "proto")]
pub(crate) mod proto {
    use vortex_error::{VortexResult, vortex_bail};
    use vortex_proto::expr::kind;
    use vortex_proto::expr::kind::Kind;

    use crate::is_null::IsNull;
    use crate::{ExprDeserialize, ExprRef, ExprSerializable, Id};

    pub(crate) struct IsNullSerde;

    impl Id for IsNullSerde {
        fn id(&self) -> &'static str {
            "is_null"
        }
    }

    impl ExprDeserialize for IsNullSerde {
        fn deserialize(&self, kind: &Kind, children: Vec<ExprRef>) -> VortexResult<ExprRef> {
            let Kind::IsNull(kind::IsNull {}) = kind else {
                vortex_bail!("wrong kind {:?}, want is_null", kind)
            };

            Ok(IsNull::new_expr(children[0].clone()))
        }
    }

    impl ExprSerializable for IsNull {
        fn id(&self) -> &'static str {
            IsNullSerde.id()
        }

        fn serialize_kind(&self) -> VortexResult<Kind> {
            Ok(Kind::IsNull(kind::IsNull {}))
        }
    }
}

impl VortexExpr for IsNull {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn unchecked_evaluate(&self, batch: &dyn Array) -> VortexResult<ArrayRef> {
        let array = self.child.evaluate(batch)?;
        match array.validity_mask()? {
            Mask::AllTrue(len) => Ok(ConstantArray::new(false, len).into_array()),
            Mask::AllFalse(len) => Ok(ConstantArray::new(true, len).into_array()),
            Mask::Values(mask) => Ok(BoolArray::from(mask.boolean_buffer().not()).into_array()),
        }
    }

    fn children(&self) -> Vec<&ExprRef> {
        vec![&self.child]
    }

    fn replacing_children(self: Arc<Self>, mut children: Vec<ExprRef>) -> ExprRef {
        Self::new_expr(
            children
                .pop()
                .vortex_expect("IsNull::replacing_children should have one child"),
        )
    }

    fn return_dtype(&self, _scope_dtype: &DType) -> VortexResult<DType> {
        Ok(DType::Bool(Nullability::NonNullable))
    }
}

pub fn is_null(child: ExprRef) -> ExprRef {
    IsNull::new_expr(child)
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::{PrimitiveArray, StructArray};
    use vortex_dtype::{DType, Nullability};
    use vortex_scalar::Scalar;

    use crate::is_null::is_null;
    use crate::{get_item, ident, test_harness};

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();
        assert_eq!(
            is_null(ident()).return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
    }

    #[test]
    fn replace_children() {
        let expr = is_null(ident());
        let _ = expr.replacing_children(vec![ident()]);
    }

    #[test]
    fn evaluate_mask() {
        let test_array =
            PrimitiveArray::from_option_iter(vec![Some(1), None, Some(2), None, Some(3)])
                .into_array();
        let expected = [false, true, false, true, false];

        let result = is_null(ident()).unchecked_evaluate(&test_array).unwrap();

        assert_eq!(result.len(), test_array.len());
        assert_eq!(result.dtype(), &DType::Bool(Nullability::NonNullable));

        for (i, expected_value) in expected.iter().enumerate() {
            assert_eq!(
                result.scalar_at(i).unwrap(),
                Scalar::bool(*expected_value, Nullability::NonNullable)
            );
        }
    }

    #[test]
    fn evaluate_all_false() {
        let test_array = PrimitiveArray::from_iter(vec![1, 2, 3, 4, 5]).into_array();

        let result = is_null(ident()).unchecked_evaluate(&test_array).unwrap();

        assert_eq!(result.len(), test_array.len());
        assert_eq!(
            result.as_constant().unwrap(),
            Scalar::bool(false, Nullability::NonNullable)
        );
    }

    #[test]
    fn evaluate_all_true() {
        let test_array =
            PrimitiveArray::from_option_iter(vec![None::<i32>, None, None, None, None])
                .into_array();

        let result = is_null(ident()).unchecked_evaluate(&test_array).unwrap();

        assert_eq!(result.len(), test_array.len());
        assert_eq!(
            result.as_constant().unwrap(),
            Scalar::bool(true, Nullability::NonNullable)
        );
    }

    #[test]
    fn evaluate_struct() {
        let test_array = StructArray::from_fields(&[(
            "a",
            PrimitiveArray::from_option_iter(vec![Some(1), None, Some(2), None, Some(3)])
                .into_array(),
        )])
        .unwrap()
        .into_array();
        let expected = [false, true, false, true, false];

        let result = is_null(get_item("a", ident()))
            .unchecked_evaluate(&test_array)
            .unwrap();

        assert_eq!(result.len(), test_array.len());
        assert_eq!(result.dtype(), &DType::Bool(Nullability::NonNullable));

        for (i, expected_value) in expected.iter().enumerate() {
            assert_eq!(
                result.scalar_at(i).unwrap(),
                Scalar::bool(*expected_value, Nullability::NonNullable)
            );
        }
    }
}
