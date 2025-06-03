use std::any::Any;
use std::fmt::Display;
use std::sync::Arc;

use vortex_array::compute::cast as compute_cast;
use vortex_array::{Array, ArrayRef};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};

use crate::{ExprRef, VortexExpr};

#[derive(Debug, Eq, Hash)]
#[allow(clippy::derived_hash_with_manual_eq)]
pub struct Cast {
    target: DType,
    child: ExprRef,
}

impl Cast {
    pub fn new_expr(child: ExprRef, target: DType) -> ExprRef {
        Arc::new(Self { target, child })
    }
}

impl PartialEq for Cast {
    fn eq(&self, other: &Self) -> bool {
        self.target.eq(&other.target) && self.child.eq(&other.child)
    }
}

impl Display for Cast {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cast({}, {})", self.child, self.target)
    }
}

#[cfg(feature = "proto")]
pub(crate) mod proto {
    use vortex_dtype::DType;
    use vortex_error::{VortexResult, vortex_bail, vortex_err};
    use vortex_proto::expr::kind;
    use vortex_proto::expr::kind::Kind;

    use crate::cast::Cast;
    use crate::{ExprDeserialize, ExprRef, ExprSerializable, Id};

    pub(crate) struct CastSerde;

    impl Id for CastSerde {
        fn id(&self) -> &'static str {
            "cast"
        }
    }

    impl ExprDeserialize for CastSerde {
        fn deserialize(&self, kind: &Kind, children: Vec<ExprRef>) -> VortexResult<ExprRef> {
            let Kind::Cast(kind::Cast { target }) = kind else {
                vortex_bail!("wrong kind {:?}, want cast", kind)
            };
            let target: DType = target
                .as_ref()
                .ok_or_else(|| vortex_err!("empty target dtype"))?
                .try_into()?;

            Ok(Cast::new_expr(children[0].clone(), target))
        }
    }

    impl ExprSerializable for Cast {
        fn id(&self) -> &'static str {
            CastSerde.id()
        }

        fn serialize_kind(&self) -> VortexResult<Kind> {
            Ok(Kind::Cast(kind::Cast {
                target: Some((&self.target).into()),
            }))
        }
    }
}

impl VortexExpr for Cast {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn unchecked_evaluate(&self, batch: &dyn Array) -> VortexResult<ArrayRef> {
        let array = self.child.evaluate(batch)?;
        compute_cast(&array, &self.target)
    }

    fn children(&self) -> Vec<&ExprRef> {
        vec![&self.child]
    }

    fn replacing_children(self: Arc<Self>, mut children: Vec<ExprRef>) -> ExprRef {
        Self::new_expr(
            children
                .pop()
                .vortex_expect("Cast::replacing_children should have one child"),
            self.target.clone(),
        )
    }

    fn return_dtype(&self, _scope_dtype: &DType) -> VortexResult<DType> {
        Ok(self.target.clone())
    }
}

pub fn cast(child: ExprRef, target: DType) -> ExprRef {
    Cast::new_expr(child, target)
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::StructArray;
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::{ExprRef, cast, get_item, ident, test_harness};

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();
        assert_eq!(
            cast(ident(), DType::Bool(Nullability::NonNullable))
                .return_dtype(&dtype)
                .unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
    }

    #[test]
    fn replace_children() {
        let expr = cast(ident(), DType::Bool(Nullability::Nullable));
        let _ = expr.replacing_children(vec![ident()]);
    }

    #[test]
    fn evaluate() {
        let test_array = StructArray::from_fields(&[
            ("a", buffer![0i32, 1, 2].into_array()),
            ("b", buffer![4i64, 5, 6].into_array()),
        ])
        .unwrap()
        .into_array();

        let expr: ExprRef = cast(
            get_item("a", ident()),
            DType::Primitive(PType::I64, Nullability::NonNullable),
        );
        let result = expr.unchecked_evaluate(&test_array).unwrap();

        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );
    }
}
