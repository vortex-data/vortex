use itertools::Itertools;
use vortex_error::{VortexExpect, VortexResult};

use super::nnf::nnf;
use crate::traversal::{Node as _, NodeVisitor, TraversalOrder};
use crate::{BinaryExpr, ExprRef, Operator, lit, or};

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
/// let cnfed = cnf(double_negation);
/// assert_eq!(cnfed, vec![col("a")]);
/// ```
///
/// Unlike NNF, CNF, lifts conjunctions to the top-level and distributions disjunctions such that
/// there is at most one disjunction for each conjunction operand:
///
///
/// ```rust
/// use vortex_expr::{not, col, or, and};
/// use vortex_expr::forms::cnf::cnf;
///
/// assert_eq!(
///     cnf(
///         or(
///             and(col("a"), col("b")),
///             col("c"),
///         )
///     ),
///     vec![
///         or(col("a"), col("c")),
///         or(col("b"), col("c")),
///     ]
/// );
/// ```
///
pub fn cnf(expr: ExprRef) -> Vec<ExprRef> {
    if expr == lit(true) {
        // True in CNF
        return vec![];
    }
    let nnf = nnf(expr);

    let mut visitor = CNFVisitor::default();
    nnf.accept(&mut visitor).vortex_expect("cannot fail");
    visitor
        .finish()
        .into_iter()
        .filter_map(|disjunction| disjunction.into_iter().reduce(or))
        .collect_vec()
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
                    let lhs_conjuncts = visitor.finish();

                    let mut visitor = CNFVisitor::default();
                    binary_expr.rhs().accept(&mut visitor)?;
                    let rhs_conjuncts = visitor.finish();

                    self.conjuncts_of_disjuncts
                        .extend(lhs_conjuncts.iter().flat_map(|lhs_disjunct| {
                            rhs_conjuncts.iter().map(|rhs_disjunct| {
                                let mut lhs_copy = lhs_disjunct.clone();
                                lhs_copy.extend(rhs_disjunct.iter().cloned());
                                lhs_copy
                            })
                        }));

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

#[cfg(test)]
mod tests {

    use vortex_expr::forms::cnf::cnf;
    use vortex_expr::{and, col, eq, gt_eq, lit, lt, not_eq, or};

    #[test]
    fn test_cnf_simple() {
        assert_eq!(
            cnf(or(or(and(col("a"), col("b")), col("c")), col("d"))),
            vec![
                or(or(col("a"), col("c")), col("d")),
                or(or(col("b"), col("c")), col("d"))
            ]
        );
    }

    #[test]
    fn test_with_lit() {
        assert_eq!(
            cnf(or(
                and(
                    gt_eq(col("earnings"), lit(50_000)),
                    not_eq(col("role"), lit("Manager"))
                ),
                col("special_flag")
            ),),
            vec![
                or(gt_eq(col("earnings"), lit(50_000)), col("special_flag")),
                or(not_eq(col("role"), lit("Manager")), col("special_flag"))
            ]
        );
    }

    #[test]
    fn test_cnf() {
        assert_eq!(
            cnf(or(
                or(
                    and(
                        gt_eq(col("earnings"), lit(50_000)),
                        not_eq(col("role"), lit("Manager"))
                    ),
                    col("special_flag")
                ),
                and(lt(col("tenure"), lit(5)), eq(col("role"), lit("Engineer"))),
            )),
            vec![
                or(
                    or(gt_eq(col("earnings"), lit(50_000)), col("special_flag")),
                    lt(col("tenure"), lit(5))
                ),
                or(
                    or(gt_eq(col("earnings"), lit(50_000)), col("special_flag")),
                    eq(col("role"), lit("Engineer"))
                ),
                or(
                    or(not_eq(col("role"), lit("Manager")), col("special_flag")),
                    lt(col("tenure"), lit(5))
                ),
                or(
                    or(not_eq(col("role"), lit("Manager")), col("special_flag")),
                    eq(col("role"), lit("Engineer"))
                )
            ]
        );
    }
}
