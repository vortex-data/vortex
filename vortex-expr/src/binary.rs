use std::any::Any;
use std::fmt::Display;
use std::sync::Arc;

use vortex_array::aliases::hash_set::HashSet;
use vortex_array::compute::{and_kleene, compare, or_kleene, Operator as ArrayOperator};
use vortex_array::ArrayData;
use vortex_dtype::field::Field;
use vortex_error::VortexResult;

use crate::{unbox_any, ExprRef, Operator, VortexExpr};

#[derive(Debug, Clone)]
pub struct BinaryExpr {
    lhs: ExprRef,
    operator: Operator,
    rhs: ExprRef,
}

impl BinaryExpr {
    pub fn new_expr(lhs: ExprRef, operator: Operator, rhs: ExprRef) -> ExprRef {
        Arc::new(Self { lhs, operator, rhs })
    }

    pub fn lhs(&self) -> &ExprRef {
        &self.lhs
    }

    pub fn rhs(&self) -> &ExprRef {
        &self.rhs
    }

    pub fn op(&self) -> Operator {
        self.operator
    }
}

impl Display for BinaryExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} {} {})", self.lhs, self.operator, self.rhs)
    }
}

impl VortexExpr for BinaryExpr {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, batch: &ArrayData) -> VortexResult<ArrayData> {
        let lhs = self.lhs.evaluate(batch)?;
        let rhs = self.rhs.evaluate(batch)?;

        match self.operator {
            Operator::Eq => compare(lhs, rhs, ArrayOperator::Eq),
            Operator::NotEq => compare(lhs, rhs, ArrayOperator::NotEq),
            Operator::Lt => compare(lhs, rhs, ArrayOperator::Lt),
            Operator::Lte => compare(lhs, rhs, ArrayOperator::Lte),
            Operator::Gt => compare(lhs, rhs, ArrayOperator::Gt),
            Operator::Gte => compare(lhs, rhs, ArrayOperator::Gte),
            Operator::And => and_kleene(lhs, rhs),
            Operator::Or => or_kleene(lhs, rhs),
        }
    }

    fn collect_references<'a>(&'a self, references: &mut HashSet<&'a Field>) {
        self.lhs.collect_references(references);
        self.rhs.collect_references(references);
    }
}

impl PartialEq<dyn Any> for BinaryExpr {
    fn eq(&self, other: &dyn Any) -> bool {
        unbox_any(other)
            .downcast_ref::<Self>()
            .map(|x| x.operator == self.operator && x.lhs.eq(&self.lhs) && x.rhs.eq(&self.rhs))
            .unwrap_or(false)
    }
}
