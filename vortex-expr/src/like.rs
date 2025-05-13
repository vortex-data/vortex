use std::any::Any;
use std::fmt::Display;
use std::hash::Hash;
use std::sync::Arc;

use vortex_array::compute::{LikeOptions, like};
use vortex_array::{Array, ArrayRef};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::{ExprRef, VortexExpr};

#[derive(Debug, Eq, Hash)]
#[allow(clippy::derived_hash_with_manual_eq)]
pub struct Like {
    child: ExprRef,
    pattern: ExprRef,
    negated: bool,
    case_insensitive: bool,
}

impl Like {
    pub fn new_expr(
        child: ExprRef,
        pattern: ExprRef,
        negated: bool,
        case_insensitive: bool,
    ) -> ExprRef {
        Arc::new(Self {
            child,
            pattern,
            negated,
            case_insensitive,
        })
    }

    pub fn child(&self) -> &ExprRef {
        &self.child
    }

    pub fn pattern(&self) -> &ExprRef {
        &self.pattern
    }

    pub fn negated(&self) -> bool {
        self.negated
    }

    pub fn case_insensitive(&self) -> bool {
        self.case_insensitive
    }
}

impl Display for Like {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} LIKE {}", self.child(), self.pattern())
    }
}

#[cfg(feature = "proto")]
pub(crate) mod proto {
    use vortex_error::{VortexResult, vortex_bail};
    use vortex_proto::expr::kind;
    use vortex_proto::expr::kind::Kind;

    use crate::{ExprDeserialize, ExprRef, ExprSerializable, Id, Like};

    pub(crate) struct LikeSerde;

    impl Id for LikeSerde {
        fn id(&self) -> &'static str {
            "like"
        }
    }

    impl ExprSerializable for Like {
        fn id(&self) -> &'static str {
            LikeSerde.id()
        }

        fn serialize_kind(&self) -> VortexResult<Kind> {
            Ok(Kind::Like(kind::Like {
                negated: self.negated,
                case_insensitive: self.case_insensitive,
            }))
        }
    }

    impl ExprDeserialize for LikeSerde {
        fn deserialize(&self, kind: &Kind, children: Vec<ExprRef>) -> VortexResult<ExprRef> {
            let Kind::Like(like) = kind else {
                vortex_bail!("wrong kind {:?}, want like", kind)
            };

            Ok(Like::new_expr(
                children[0].clone(),
                children[1].clone(),
                like.negated,
                like.case_insensitive,
            ))
        }
    }
}

impl VortexExpr for Like {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn unchecked_evaluate(&self, batch: &dyn Array) -> VortexResult<ArrayRef> {
        let child = self.child().evaluate(batch)?;
        let pattern = self.pattern().evaluate(&child)?;
        like(
            &child,
            &pattern,
            LikeOptions {
                negated: self.negated,
                case_insensitive: self.case_insensitive,
            },
        )
    }

    fn children(&self) -> Vec<&ExprRef> {
        vec![&self.child, &self.pattern]
    }

    fn replacing_children(self: Arc<Self>, children: Vec<ExprRef>) -> ExprRef {
        assert_eq!(children.len(), 2);
        Like::new_expr(
            children[0].clone(),
            children[1].clone(),
            self.negated,
            self.case_insensitive,
        )
    }

    fn return_dtype(&self, scope_dtype: &DType) -> VortexResult<DType> {
        let input = self.child().return_dtype(scope_dtype)?;
        let pattern = self.pattern().return_dtype(scope_dtype)?;
        Ok(DType::Bool(
            (input.is_nullable() || pattern.is_nullable()).into(),
        ))
    }
}

impl PartialEq for Like {
    fn eq(&self, other: &Like) -> bool {
        other.case_insensitive == self.case_insensitive
            && other.negated == self.negated
            && other.pattern.eq(&self.pattern)
            && other.child.eq(&self.child)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::ToCanonical;
    use vortex_array::arrays::BoolArray;
    use vortex_dtype::{DType, Nullability};

    use crate::{Like, ident, lit, not};

    #[test]
    fn invert_booleans() {
        let not_expr = not(ident());
        let bools = BoolArray::from_iter([false, true, false, false, true, true]);
        assert_eq!(
            not_expr
                .evaluate(bools.as_ref())
                .unwrap()
                .to_bool()
                .unwrap()
                .boolean_buffer()
                .iter()
                .collect::<Vec<_>>(),
            vec![true, false, true, true, false, false]
        );
    }

    #[test]
    fn dtype() {
        let dtype = DType::Utf8(Nullability::NonNullable);
        let like_expr = Like::new_expr(ident(), lit("%test%"), false, false);
        assert_eq!(
            like_expr.return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
    }
}
