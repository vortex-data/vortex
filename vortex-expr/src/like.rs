use std::any::Any;
use std::fmt::Display;
use std::sync::Arc;

use vortex_array::aliases::hash_set::HashSet;
use vortex_array::compute::{like, LikeOptions};
use vortex_array::ArrayData;
use vortex_dtype::Field;
use vortex_error::VortexResult;

use crate::{unbox_any, ExprRef, VortexExpr};

#[derive(Debug)]
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

    fn evaluate(&self, batch: &ArrayData) -> VortexResult<ArrayData> {
        let child = self.child().evaluate(batch)?;
        let pattern = self.pattern().evaluate(batch)?;
        like(
            &child,
            &pattern,
            LikeOptions {
                negated: self.negated,
                case_insensitive: self.case_insensitive,
            },
        )
    }

    fn collect_references<'a>(&'a self, references: &mut HashSet<&'a Field>) {
        self.child().collect_references(references);
        self.pattern().collect_references(references);
    }
}

impl PartialEq<dyn Any> for Like {
    fn eq(&self, other: &dyn Any) -> bool {
        unbox_any(other)
            .downcast_ref::<Self>()
            .map(|x| x.child.eq(&self.child) && x.pattern.eq(&self.pattern))
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::array::BoolArray;
    use vortex_array::IntoArrayVariant;

    use crate::{Identity, Not};

    #[test]
    fn invert_booleans() {
        let not_expr = Not::new_expr(Arc::new(Identity));
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
}
