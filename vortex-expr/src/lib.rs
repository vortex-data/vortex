use std::any::Any;
use std::fmt::{Debug, Display};
use std::sync::Arc;

use vortex_array::aliases::hash_set::HashSet;

mod binary;
mod column;
pub mod datafusion;
mod identity;
mod literal;
mod not;
mod operators;
mod select;

pub use binary::*;
pub use column::*;
pub use identity::*;
pub use literal::*;
pub use not::*;
pub use operators::*;
pub use select::*;
use vortex_array::ArrayData;
use vortex_dtype::field::Field;
use vortex_error::{VortexExpect, VortexResult};

pub type ExprRef = Arc<dyn VortexExpr>;

/// Represents logical operation on [`ArrayData`]s
pub trait VortexExpr: Debug + Send + Sync + PartialEq<dyn Any> + Display {
    /// Convert expression reference to reference of [`Any`] type
    fn as_any(&self) -> &dyn Any;

    /// Compute result of expression on given batch producing a new batch
    fn evaluate(&self, batch: &ArrayData) -> VortexResult<ArrayData>;

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
pub fn split_conjunction(expr: &ExprRef) -> Vec<ExprRef> {
    let mut conjunctions = vec![];
    split_inner(expr, &mut conjunctions);
    conjunctions
}

fn split_inner(expr: &ExprRef, exprs: &mut Vec<ExprRef>) {
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
    if any.is::<ExprRef>() {
        any.downcast_ref::<ExprRef>()
            .vortex_expect("any.is::<ExprRef> returned true but downcast_ref failed")
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
    use vortex_dtype::{DType, Nullability, PType, StructDType};
    use vortex_scalar::Scalar;

    use super::*;

    #[test]
    fn basic_expr_split_test() {
        let lhs = Column::new_expr(Field::Name("a".to_string()));
        let rhs = Literal::new_expr(1.into());
        let expr = BinaryExpr::new_expr(lhs, Operator::Eq, rhs);
        let conjunction = split_conjunction(&expr);
        assert_eq!(conjunction.len(), 1);
    }

    #[test]
    fn basic_conjunction_split_test() {
        let lhs = Column::new_expr(Field::Name("a".to_string()));
        let rhs = Literal::new_expr(1.into());
        let expr = BinaryExpr::new_expr(lhs, Operator::And, rhs);
        let conjunction = split_conjunction(&expr);
        assert_eq!(conjunction.len(), 2, "Conjunction is {conjunction:?}");
    }

    #[test]
    fn expr_display() {
        assert_eq!(
            Column::new_expr(Field::Name("a".to_string())).to_string(),
            "$a"
        );
        assert_eq!(Column::new_expr(Field::Index(1)).to_string(), "[1]");
        assert_eq!(Identity.to_string(), "[]");
        assert_eq!(Identity.to_string(), "[]");

        let col1: Arc<dyn VortexExpr> = Column::new_expr(Field::Name("col1".to_string()));
        let col2: Arc<dyn VortexExpr> = Column::new_expr(Field::Name("col2".to_string()));
        assert_eq!(
            BinaryExpr::new_expr(col1.clone(), Operator::And, col2.clone()).to_string(),
            "($col1 and $col2)"
        );
        assert_eq!(
            BinaryExpr::new_expr(col1.clone(), Operator::Or, col2.clone()).to_string(),
            "($col1 or $col2)"
        );
        assert_eq!(
            BinaryExpr::new_expr(col1.clone(), Operator::Eq, col2.clone()).to_string(),
            "($col1 = $col2)"
        );
        assert_eq!(
            BinaryExpr::new_expr(col1.clone(), Operator::NotEq, col2.clone()).to_string(),
            "($col1 != $col2)"
        );
        assert_eq!(
            BinaryExpr::new_expr(col1.clone(), Operator::Gt, col2.clone()).to_string(),
            "($col1 > $col2)"
        );
        assert_eq!(
            BinaryExpr::new_expr(col1.clone(), Operator::Gte, col2.clone()).to_string(),
            "($col1 >= $col2)"
        );
        assert_eq!(
            BinaryExpr::new_expr(col1.clone(), Operator::Lt, col2.clone()).to_string(),
            "($col1 < $col2)"
        );
        assert_eq!(
            BinaryExpr::new_expr(col1.clone(), Operator::Lte, col2.clone()).to_string(),
            "($col1 <= $col2)"
        );

        assert_eq!(
            BinaryExpr::new_expr(
                BinaryExpr::new_expr(col1.clone(), Operator::Lt, col2.clone()),
                Operator::Or,
                BinaryExpr::new_expr(col1.clone(), Operator::NotEq, col2.clone())
            )
            .to_string(),
            "(($col1 < $col2) or ($col1 != $col2))"
        );

        assert_eq!(Not::new_expr(col1.clone()).to_string(), "!$col1");

        assert_eq!(
            Select::include(vec![Field::Name("col1".to_string())]).to_string(),
            "Include($col1)"
        );
        assert_eq!(
            Select::include(vec![
                Field::Name("col1".to_string()),
                Field::Name("col2".to_string())
            ])
            .to_string(),
            "Include($col1,$col2)"
        );
        assert_eq!(
            Select::exclude(vec![
                Field::Name("col1".to_string()),
                Field::Name("col2".to_string()),
                Field::Index(1),
            ])
            .to_string(),
            "Exclude($col1,$col2,[1])"
        );

        assert_eq!(Literal::new_expr(Scalar::from(0_u8)).to_string(), "0_u8");
        assert_eq!(
            Literal::new_expr(Scalar::from(0.0_f32)).to_string(),
            "0_f32"
        );
        assert_eq!(
            Literal::new_expr(Scalar::from(i64::MAX)).to_string(),
            "9223372036854775807_i64"
        );
        assert_eq!(Literal::new_expr(Scalar::from(true)).to_string(), "true");
        assert_eq!(
            Literal::new_expr(Scalar::null(DType::Bool(Nullability::Nullable))).to_string(),
            "null"
        );

        assert_eq!(
            Literal::new_expr(Scalar::struct_(
                DType::Struct(
                    StructDType::new(
                        Arc::from([Arc::from("dog"), Arc::from("cat")]),
                        vec![
                            DType::Primitive(PType::U32, Nullability::NonNullable),
                            DType::Utf8(Nullability::NonNullable)
                        ],
                    ),
                    Nullability::NonNullable
                ),
                vec![Scalar::from(32_u32), Scalar::from("rufus".to_string())]
            ))
            .to_string(),
            "{dog:32_u32,cat:rufus}"
        );
    }
}
