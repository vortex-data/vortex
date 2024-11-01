use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use vortex_array::aliases::hash_set::HashSet;

mod binary;
mod column;
pub mod datafusion;
mod identity;
mod literal;
mod operators;
mod select;

pub use binary::*;
pub use column::*;
pub use identity::*;
pub use literal::*;
pub use operators::*;
pub use select::*;
use vortex_array::Array;
use vortex_dtype::field::Field;
use vortex_error::{VortexExpect, VortexResult};

/// Represents logical operation on [`Array`]s
pub trait VortexExpr: Debug + Send + Sync + PartialEq<dyn Any> {
    /// Convert expression reference to reference of [`Any`] type
    fn as_any(&self) -> &dyn Any;

    /// Compute result of expression on given batch producing a new batch
    fn evaluate(&self, batch: &Array) -> VortexResult<Array>;

    /// Accumulate all field references from this expression and its children in the provided set
    fn collect_references<'a>(&'a self, _references: &mut HashSet<&'a Field>) {}

    /// Accumulate all field references from this expression and its children in a new set
    fn references(&self) -> HashSet<&Field> {
        let mut refs = HashSet::new();
        self.collect_references(&mut refs);
        refs
    }
}

/// Splits top level and operations into separate expressions
pub fn split_conjunction(expr: &Arc<dyn VortexExpr>) -> Vec<Arc<dyn VortexExpr>> {
    let mut conjunctions = vec![];
    split_inner(expr, &mut conjunctions);
    conjunctions
}

fn split_inner(expr: &Arc<dyn VortexExpr>, exprs: &mut Vec<Arc<dyn VortexExpr>>) {
    match expr.as_any().downcast_ref::<BinaryExpr>() {
        Some(bexp) if bexp.op() == Operator::And => {
            split_inner(bexp.lhs(), exprs);
            split_inner(bexp.rhs(), exprs);
        }
        Some(_) | None => {
            exprs.push(expr.clone());
        }
    }
}

// Taken from apache-datafusion, necessary since you can't require VortexExpr implement PartialEq<dyn VortexExpr>
pub fn unbox_any(any: &dyn Any) -> &dyn Any {
    if any.is::<Arc<dyn VortexExpr>>() {
        any.downcast_ref::<Arc<dyn VortexExpr>>()
            .vortex_expect("any.is::<Arc<dyn VortexExpr>> returned true but downcast_ref failed")
            .as_any()
    } else if any.is::<Box<dyn VortexExpr>>() {
        any.downcast_ref::<Box<dyn VortexExpr>>()
            .vortex_expect("any.is::<Box<dyn VortexExpr>> returned true but downcast_ref failed")
            .as_any()
    } else {
        any
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::field::Field;

    use super::*;

    #[test]
    fn basic_expr_split_test() {
        let lhs = Arc::new(Column::new(Field::Name("a".to_string()))) as _;
        let rhs = Arc::new(Literal::new(1.into())) as _;
        let expr = Arc::new(BinaryExpr::new(lhs, Operator::Eq, rhs)) as _;
        let conjunction = split_conjunction(&expr);
        assert_eq!(conjunction.len(), 1);
    }

    #[test]
    fn basic_conjunction_split_test() {
        let lhs = Arc::new(Column::new(Field::Name("a".to_string()))) as _;
        let rhs = Arc::new(Literal::new(1.into())) as _;
        let expr = Arc::new(BinaryExpr::new(lhs, Operator::And, rhs)) as _;
        let conjunction = split_conjunction(&expr);
        assert_eq!(conjunction.len(), 2, "Conjunction is {conjunction:?}");
    }
}
