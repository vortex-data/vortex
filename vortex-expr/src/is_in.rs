use std::any::Any;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;

use vortex_array::compute::list_contains;
use vortex_array::{Array, ArrayRef};
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{ExprRef, VortexExpr};

#[derive(Debug, Clone, Eq, Hash)]
#[allow(clippy::derived_hash_with_manual_eq)]
pub struct IsIn {
    value: Scalar,
    child: ExprRef,
}

impl IsIn {
    pub fn new_expr(value: impl Into<Scalar>, child: ExprRef) -> ExprRef {
        Arc::new(Self {
            value: value.into(),
            child,
        })
    }

    pub fn value(&self) -> &Scalar {
        &self.value
    }
}

pub fn is_in(value: impl Into<Scalar>, child: ExprRef) -> ExprRef {
    IsIn::new_expr(value, child)
}

impl Display for IsIn {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} in {}", &self.value, &self.child)
    }
}

#[cfg(feature = "proto")]
pub(crate) mod proto {
    use vortex_error::{VortexResult, vortex_bail, vortex_err};
    use vortex_proto::expr::kind;
    use vortex_proto::expr::kind::Kind;
    use vortex_scalar::Scalar;

    use crate::is_in::IsIn;
    use crate::{ExprDeserialize, ExprRef, ExprSerializable, Id};

    pub(crate) struct IsInSerde;

    impl Id for IsInSerde {
        fn id(&self) -> &'static str {
            "is_in"
        }
    }

    impl ExprDeserialize for IsInSerde {
        fn deserialize(&self, kind: &Kind, children: Vec<ExprRef>) -> VortexResult<ExprRef> {
            let Kind::IsIn(kind::IsIn { value }) = kind else {
                vortex_bail!("wrong kind {:?}, want is_in", kind)
            };
            let scalar: Scalar = value
                .as_ref()
                .ok_or_else(|| vortex_err!("empty literal scalar"))?
                .try_into()?;

            Ok(IsIn::new_expr(scalar, children[0].clone()))
        }
    }

    impl ExprSerializable for IsIn {
        fn id(&self) -> &'static str {
            IsInSerde.id()
        }

        fn serialize_kind(&self) -> VortexResult<Kind> {
            Ok(Kind::IsIn(kind::IsIn {
                value: Some((&self.value).into()),
            }))
        }
    }
}

impl VortexExpr for IsIn {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn unchecked_evaluate(&self, batch: &dyn Array) -> VortexResult<ArrayRef> {
        list_contains(self.child.evaluate(batch)?.as_ref(), self.value.clone())
    }

    fn children(&self) -> Vec<&ExprRef> {
        vec![&self.child]
    }

    fn replacing_children(self: Arc<Self>, children: Vec<ExprRef>) -> ExprRef {
        assert_eq!(children.len(), 1);
        Self::new_expr(self.value().clone(), children[0].clone())
    }

    fn return_dtype(&self, scope_dtype: &DType) -> VortexResult<DType> {
        // If input is nullable, the output is nullable.
        Ok(DType::Bool(scope_dtype.nullability()))
    }
}

impl PartialEq for IsIn {
    fn eq(&self, other: &IsIn) -> bool {
        self.value == other.value && self.child.eq(&other.child)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::{BoolArray, BooleanBuffer, ListArray, PrimitiveArray};
    use vortex_array::validity::Validity;
    use vortex_array::{Array, ArrayRef, IntoArray};
    use vortex_dtype::Nullability;
    use vortex_scalar::Scalar;

    use crate::ident;
    use crate::is_in::is_in;

    fn test_array() -> ArrayRef {
        ListArray::try_new(
            PrimitiveArray::from_iter(vec![1, 1, 2, 2, 2, 2, 2, 3, 3, 3]).into_array(),
            PrimitiveArray::from_iter(vec![0, 5, 10]).into_array(),
            Validity::AllValid,
        )
        .unwrap()
        .into_array()
    }

    #[test]
    pub fn test_is_in_one() {
        let arr = test_array();

        let expr = is_in(1, ident());
        let item = expr.evaluate(arr.as_ref()).unwrap();

        assert_eq!(
            item.scalar_at(0).unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
        assert_eq!(
            item.scalar_at(1).unwrap(),
            Scalar::bool(false, Nullability::Nullable)
        );
    }

    #[test]
    pub fn test_is_in_both() {
        let arr = test_array();

        let expr = is_in(2, ident());
        let item = expr.evaluate(arr.as_ref()).unwrap();

        assert_eq!(
            item.scalar_at(0).unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
        assert_eq!(
            item.scalar_at(1).unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
    }

    #[test]
    pub fn test_is_in_none() {
        let arr = test_array();

        let expr = is_in(4, ident());
        let item = expr.evaluate(arr.as_ref()).unwrap();

        assert_eq!(
            item.scalar_at(0).unwrap(),
            Scalar::bool(false, Nullability::Nullable)
        );
        assert_eq!(
            item.scalar_at(1).unwrap(),
            Scalar::bool(false, Nullability::Nullable)
        );
    }

    #[test]
    pub fn test_is_in_empty() {
        let arr = ListArray::try_new(
            PrimitiveArray::from_iter(vec![1, 1, 2, 2, 2]).into_array(),
            PrimitiveArray::from_iter(vec![0, 5, 5]).into_array(),
            Validity::AllValid,
        )
        .unwrap()
        .into_array();

        let expr = is_in(2, ident());
        let item = expr.evaluate(arr.as_ref()).unwrap();

        assert_eq!(
            item.scalar_at(0).unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
        assert_eq!(
            item.scalar_at(1).unwrap(),
            Scalar::bool(false, Nullability::Nullable)
        );
    }

    #[test]
    pub fn test_is_in_nullable() {
        let arr = ListArray::try_new(
            PrimitiveArray::from_iter(vec![1, 1, 2, 2, 2]).into_array(),
            PrimitiveArray::from_iter(vec![0, 5, 5]).into_array(),
            Validity::Array(BoolArray::from(BooleanBuffer::from(vec![true, false])).into_array()),
        )
        .unwrap()
        .into_array();

        let expr = is_in(2, ident());
        let item = expr.evaluate(arr.as_ref()).unwrap();

        assert_eq!(
            item.scalar_at(0).unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
        assert!(!item.is_valid(1).unwrap());
    }
}
