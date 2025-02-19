use std::any::Any;
use std::fmt::Display;
use std::hash::Hash;
use std::sync::Arc;

use vortex_array::compute::invert;
use vortex_array::Array;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::{ExprRef, VortexExpr};

#[derive(Debug, Eq, Hash)]
// We cannot auto derive PartialEq because ExprRef, since its a Arc<..> and derive doesn't work
#[allow(clippy::derived_hash_with_manual_eq)]
pub struct Not {
    child: ExprRef,
}

impl Not {
    pub fn new_expr(child: ExprRef) -> ExprRef {
        Arc::new(Self { child })
    }

    pub fn child(&self) -> &ExprRef {
        &self.child
    }
}

impl Display for Not {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "!")?;
        self.child.fmt(f)
    }
}

impl VortexExpr for Not {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn unchecked_evaluate(&self, batch: &Array) -> VortexResult<Array> {
        let child_result = self.child.evaluate(batch)?;
        invert(&child_result)
    }

    fn children(&self) -> Vec<&ExprRef> {
        vec![&self.child]
    }

    fn replacing_children(self: Arc<Self>, mut children: Vec<ExprRef>) -> ExprRef {
        assert_eq!(children.len(), 0);
        Self::new_expr(children.remove(0))
    }

    fn return_dtype(&self, scope_dtype: &DType) -> VortexResult<DType> {
        self.child.return_dtype(scope_dtype)
    }
}

impl PartialEq for Not {
    fn eq(&self, other: &Not) -> bool {
        other.child.eq(&self.child)
    }
}

pub fn not(operand: ExprRef) -> ExprRef {
    Not::new_expr(operand)
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::BoolArray;
    use vortex_array::IntoArrayVariant;
    use vortex_dtype::{DType, Nullability};

    use crate::{col, ident, not, test_harness};

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
        let not_expr = not(ident());
        assert_eq!(
            not_expr
                .return_dtype(&DType::Bool(Nullability::NonNullable))
                .unwrap(),
            DType::Bool(Nullability::NonNullable)
        );

        let dtype = test_harness::struct_dtype();
        assert_eq!(
            not(col("bool1")).return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
    }
}
