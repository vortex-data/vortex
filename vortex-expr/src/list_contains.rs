use std::any::Any;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;

use vortex_array::compute::list_contains as compute_list_contains;
use vortex_array::{Array, ArrayRef};
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{ExprRef, VortexExpr};

#[derive(Debug, Clone, Eq, Hash)]
#[allow(clippy::derived_hash_with_manual_eq)]
pub struct ListContains {
    list: ExprRef,
    value: Scalar,
}

impl ListContains {
    pub fn new_expr(list: ExprRef, value: impl Into<Scalar>) -> ExprRef {
        Arc::new(Self {
            list,
            value: value.into(),
        })
    }

    pub fn value(&self) -> &Scalar {
        &self.value
    }
}

pub fn list_contains(list: ExprRef, value: impl Into<Scalar>) -> ExprRef {
    ListContains::new_expr(list, value)
}

impl Display for ListContains {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.contains({})", &self.list, &self.value)
    }
}

#[cfg(feature = "proto")]
pub(crate) mod proto {
    use vortex_error::{VortexResult, vortex_bail, vortex_err};
    use vortex_proto::expr::kind;
    use vortex_proto::expr::kind::Kind;
    use vortex_scalar::Scalar;

    use crate::list_contains::ListContains;
    use crate::{ExprDeserialize, ExprRef, ExprSerializable, Id};

    pub(crate) struct ListContainsSerde;

    impl Id for ListContainsSerde {
        fn id(&self) -> &'static str {
            "list_contains"
        }
    }

    impl ExprDeserialize for ListContainsSerde {
        fn deserialize(&self, kind: &Kind, children: Vec<ExprRef>) -> VortexResult<ExprRef> {
            let Kind::ListContains(kind::ListContains { value }) = kind else {
                vortex_bail!("wrong kind {:?}, want list_contains", kind)
            };
            let scalar: Scalar = value
                .as_ref()
                .ok_or_else(|| vortex_err!("empty literal scalar"))?
                .try_into()?;

            Ok(ListContains::new_expr(children[0].clone(), scalar))
        }
    }

    impl ExprSerializable for ListContains {
        fn id(&self) -> &'static str {
            ListContainsSerde.id()
        }

        fn serialize_kind(&self) -> VortexResult<Kind> {
            Ok(Kind::ListContains(kind::ListContains {
                value: Some((&self.value).into()),
            }))
        }
    }
}

impl VortexExpr for ListContains {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn unchecked_evaluate(&self, batch: &dyn Array) -> VortexResult<ArrayRef> {
        compute_list_contains(self.list.evaluate(batch)?.as_ref(), self.value.clone())
    }

    fn children(&self) -> Vec<&ExprRef> {
        vec![&self.list]
    }

    fn replacing_children(self: Arc<Self>, children: Vec<ExprRef>) -> ExprRef {
        assert_eq!(children.len(), 1);
        Self::new_expr(children[0].clone(), self.value().clone())
    }

    fn return_dtype(&self, scope_dtype: &DType) -> VortexResult<DType> {
        // If input is nullable, the output is nullable.
        Ok(DType::Bool(scope_dtype.nullability()))
    }
}

impl PartialEq for ListContains {
    fn eq(&self, other: &ListContains) -> bool {
        self.value == other.value && self.list.eq(&other.list)
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
    use crate::list_contains::list_contains;

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
    pub fn test_one() {
        let arr = test_array();

        let expr = list_contains(ident(), 1);
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
    pub fn test_all() {
        let arr = test_array();

        let expr = list_contains(ident(), 2);
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
    pub fn test_none() {
        let arr = test_array();

        let expr = list_contains(ident(), 4);
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
    pub fn test_empty() {
        let arr = ListArray::try_new(
            PrimitiveArray::from_iter(vec![1, 1, 2, 2, 2]).into_array(),
            PrimitiveArray::from_iter(vec![0, 5, 5]).into_array(),
            Validity::AllValid,
        )
        .unwrap()
        .into_array();

        let expr = list_contains(ident(), 2);
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
    pub fn test_nullable() {
        let arr = ListArray::try_new(
            PrimitiveArray::from_iter(vec![1, 1, 2, 2, 2]).into_array(),
            PrimitiveArray::from_iter(vec![0, 5, 5]).into_array(),
            Validity::Array(BoolArray::from(BooleanBuffer::from(vec![true, false])).into_array()),
        )
        .unwrap()
        .into_array();

        let expr = list_contains(ident(), 2);
        let item = expr.evaluate(arr.as_ref()).unwrap();

        assert_eq!(
            item.scalar_at(0).unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
        assert!(!item.is_valid(1).unwrap());
    }
}
