use std::any::Any;
use std::fmt::Display;
use std::hash::Hash;
use std::sync::Arc;

use vortex_array::compute::{like, LikeOptions};
use vortex_array::Array;
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

impl VortexExpr for Like {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn unchecked_evaluate(&self, batch: &Array) -> VortexResult<Array> {
        let child = self.child().evaluate(batch)?;
        let pattern = self.pattern().evaluate(&child)?;
        like(
            child,
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
    use vortex_array::arrays::BoolArray;
    use vortex_array::IntoArrayVariant;
    use vortex_dtype::{DType, Nullability};

    use crate::{ident, lit, not, Like};

    #[test]
    fn invert_booleans() {
        let not_expr = not(ident());
        let bools = BoolArray::from_iter([false, true, false, false, true, true]);
        assert_eq!(
            not_expr
                .evaluate(bools.as_ref())
                .unwrap()
                .into_bool()
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
