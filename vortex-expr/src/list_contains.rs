use std::any::Any;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::compute::list_contains as compute_list_contains;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::{AnalysisExpr, ExprRef, Scope, ScopeDType, VortexExpr};

#[derive(Debug, Clone, Eq, Hash)]
#[allow(clippy::derived_hash_with_manual_eq)]
pub struct ListContains {
    list: ExprRef,
    value: ExprRef,
}

impl ListContains {
    pub fn new_expr(list: ExprRef, value: ExprRef) -> ExprRef {
        Arc::new(Self { list, value })
    }

    pub fn value(&self) -> &ExprRef {
        &self.value
    }
}

pub fn list_contains(list: ExprRef, value: ExprRef) -> ExprRef {
    ListContains::new_expr(list, value)
}

impl Display for ListContains {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.contains({})", &self.list, &self.value)
    }
}

#[cfg(feature = "proto")]
pub(crate) mod proto {
    use vortex_error::{VortexResult, vortex_bail};
    use vortex_proto::expr::kind;
    use vortex_proto::expr::kind::Kind;

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
            let Kind::ListContains(kind::ListContains {}) = kind else {
                vortex_bail!("wrong kind {:?}, want list_contains", kind)
            };

            Ok(ListContains::new_expr(
                children[0].clone(),
                children[1].clone(),
            ))
        }
    }

    impl ExprSerializable for ListContains {
        fn id(&self) -> &'static str {
            ListContainsSerde.id()
        }

        fn serialize_kind(&self) -> VortexResult<Kind> {
            Ok(Kind::ListContains(kind::ListContains {}))
        }
    }
}

impl AnalysisExpr for ListContains {}

impl VortexExpr for ListContains {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn unchecked_evaluate(&self, scope: &Scope) -> VortexResult<ArrayRef> {
        let Some(scalar) = self.value.evaluate(scope)?.as_constant() else {
            todo!("not implemented list contains of a value array")
        };
        compute_list_contains(self.list.evaluate(scope)?.as_ref(), scalar)
    }

    fn children(&self) -> Vec<&ExprRef> {
        vec![&self.list, &self.value]
    }

    fn replacing_children(self: Arc<Self>, children: Vec<ExprRef>) -> ExprRef {
        assert_eq!(children.len(), 2);
        Self::new_expr(children[0].clone(), children[1].clone())
    }

    fn return_dtype(&self, scope_dtype: &ScopeDType) -> VortexResult<DType> {
        Ok(DType::Bool(
            self.list.return_dtype(scope_dtype)?.nullability()
                | self.value.return_dtype(scope_dtype)?.nullability(),
        ))
    }
}

impl PartialEq for ListContains {
    fn eq(&self, other: &ListContains) -> bool {
        self.value.eq(&other.value) && self.list.eq(&other.list)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::{BoolArray, BooleanBuffer, ListArray, PrimitiveArray};
    use vortex_array::validity::Validity;
    use vortex_array::{Array, ArrayRef, IntoArray};
    use vortex_dtype::{FieldNames, Nullability, StructFields};
    use vortex_scalar::Scalar;

    use crate::list_contains::list_contains;
    use crate::{Arc, DType, Scope, ScopeDType, get_item, lit, root};

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

        let expr = list_contains(root(), lit(1));
        let item = expr.evaluate(&Scope::new(arr)).unwrap();

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

        let expr = list_contains(root(), lit(2));
        let item = expr.evaluate(&Scope::new(arr)).unwrap();

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

        let expr = list_contains(root(), lit(4));
        let item = expr.evaluate(&Scope::new(arr)).unwrap();

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

        let expr = list_contains(root(), lit(2));
        let item = expr.evaluate(&Scope::new(arr)).unwrap();

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

        let expr = list_contains(root(), lit(2));
        let item = expr.evaluate(&Scope::new(arr)).unwrap();

        assert_eq!(
            item.scalar_at(0).unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
        assert!(!item.is_valid(1).unwrap());
    }

    #[test]
    pub fn test_return_type() {
        let scope = ScopeDType::new(DType::Struct(
            Arc::new(StructFields::new(
                FieldNames::from(["array".into()]),
                vec![DType::List(
                    Arc::new(DType::Primitive(
                        vortex_dtype::PType::I32,
                        Nullability::NonNullable,
                    )),
                    Nullability::Nullable,
                )],
            )),
            Nullability::NonNullable,
        ));

        let expr = list_contains(get_item("array", root()), lit(2));

        // Expect nullable, although scope is non-nullable
        assert_eq!(
            expr.return_dtype(&scope).unwrap(),
            DType::Bool(Nullability::Nullable)
        );
    }
}
