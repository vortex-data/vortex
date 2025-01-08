use vortex_error::VortexResult;

use super::nnf::nnf;
use crate::traversal::{Node as _, NodeVisitor, TraversalOrder};
use crate::{BinaryExpr, ExprRef, Operator};

/// Return an equivalent expression in Conjunctive Normal Form (CNF).
///
/// A CNF expression is a vector of vectors. The outer vector is a conjunction. The inner vectors
/// are disjunctions. Neither [Operator::And] nor [Operator::Or] may appear in the
/// disjunctions. Moreover, each disjunction in a CNF expression must be in Negative Normal Form.
///
/// # Examples
///
/// All the NNF examples also apply to CNF, for example double negation is removed entirely:
///
/// ```
/// use vortex_expr::{not, col};
/// use vortex_expr::forms::cnf::cnf;
///
/// let double_negation = not(not(col("a")));
/// let cnfed = cnf(double_negation).unwrap();
/// assert_eq!(cnfed, vec![vec![col("a")]]);
/// ```
///
/// Unlike NNF, CNF, lifts conjunctions to the top-level and distributions disjunctions such that
/// there is at most one disjunction for each conjunction operand:
///
/// ```
/// use vortex_expr::{not, col, or, and};
/// use vortex_expr::forms::cnf::cnf;
///
/// assert_eq!(
///     cnf(or(or(and(col("a"), col("b")), col("c")), col("d"))).unwrap(),
///     vec![vec![col("a"), col("c"), col("d")], vec![col("b"), col("c"), col("d")]]
/// );
/// ```
///
/// Vortex extends the CNF definition to any Boolean-valued expression, even ones with non-Boolean
/// parameters:
///
/// ```
/// use vortex_expr::{not, col, or, and, gt_eq, lit, not_eq, lt, eq};
/// use vortex_expr::forms::cnf::cnf;
/// use itertools::Itertools;
///
/// assert_eq!(
///     cnf(
///         or(
///             and(
///                 gt_eq(col("earnings"), lit(50_000)),
///                 not_eq(col("role"), lit("Manager"))
///             ),
///             col("special_flag")
///         ),
///     ).unwrap(),
///     vec![
///         vec![
///             gt_eq(col("earnings"), lit(50_000)),
///             col("special_flag")
///         ],
///         vec![
///             not_eq(col("role"), lit("Manager")),
///             col("special_flag")
///         ]
///     ]
/// );
///
/// assert_eq!(
///     cnf(
///         or(
///             or(
///                 and(
///                     gt_eq(col("earnings"), lit(50_000)),
///                     not_eq(col("role"), lit("Manager"))
///                 ),
///                 col("special_flag")
///             ),
///             and(
///                 lt(col("tenure"), lit(5)),
///                 eq(col("role"), lit("Engineer"))
///             ),
///         )
///     ).unwrap(),
///     vec![
///         vec![
///             gt_eq(col("earnings"), lit(50_000)),
///             col("special_flag"),
///             lt(col("tenure"), lit(5)),
///             eq(col("role"), lit("Engineer"))
///         ],
///         vec![
///             not_eq(col("role"), lit("Manager")),
///             col("special_flag"),
///             lt(col("tenure"), lit(5)),
///             eq(col("role"), lit("Engineer"))
///         ]
///     ]
/// );
/// ```
///
pub fn cnf(expr: ExprRef) -> VortexResult<Vec<Vec<ExprRef>>> {
    let nnf = nnf(expr)?;

    let mut visitor = CNFVisitor::default();
    nnf.accept(&mut visitor)?;
    Ok(visitor.finish())
}

#[derive(Default)]
struct CNFVisitor {
    conjuncts_of_disjuncts: Vec<Vec<ExprRef>>,
}

impl CNFVisitor {
    fn finish(self) -> Vec<Vec<ExprRef>> {
        self.conjuncts_of_disjuncts
    }
}

impl NodeVisitor<'_> for CNFVisitor {
    type NodeTy = ExprRef;

    fn visit_down(&mut self, node: &ExprRef) -> VortexResult<TraversalOrder> {
        if let Some(binary_expr) = node.as_any().downcast_ref::<BinaryExpr>() {
            match binary_expr.op() {
                Operator::And => return Ok(TraversalOrder::Continue),
                Operator::Or => {
                    let mut visitor = CNFVisitor::default();
                    binary_expr.lhs().accept(&mut visitor)?;
                    let mut lhs = visitor.finish();

                    let mut visitor = CNFVisitor::default();
                    binary_expr.rhs().accept(&mut visitor)?;
                    let rhs = visitor.finish();

                    for l in lhs.iter_mut() {
                        for r in rhs.iter() {
                            l.extend(r.clone())
                        }
                    }

                    self.conjuncts_of_disjuncts.extend(lhs);

                    return Ok(TraversalOrder::Skip);
                }
                _ => {}
            }
        }
        // Anything other than And and Or are terminals from the perspective of CNF
        self.conjuncts_of_disjuncts.push(vec![node.clone()]);
        Ok(TraversalOrder::Skip)
    }
}
