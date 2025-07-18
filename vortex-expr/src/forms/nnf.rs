// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::traversal::{FoldDown, FoldUp, FolderMut, Node as _, NodeVisitor, TraversalOrder};
use crate::{
    BinaryExpr, BinaryVTable, ExprRef, GetItemVTable, IntoExpr, LiteralVTable, NotVTable, Operator,
    not,
};

/// Return an equivalent expression in Negative Normal Form ([NNF](https://en.wikipedia.org/wiki/Negation_normal_form)).
///
/// In NNF, [crate::NotExpr] expressions may only contain terminal nodes such as [Literal](crate::LiteralExpr) or
/// [GetItem](crate::GetItemExpr). They *may not* contain [crate::BinaryExpr], [crate::NotExpr], etc.
///
/// # Examples
///
/// Double negation is removed entirely:
///
/// ```
/// use vortex_expr::{not, col};
/// use vortex_expr::forms::nnf::nnf;
///
/// let double_negation = not(not(col("a")));
/// let nnfed = nnf(double_negation);
/// assert_eq!(&nnfed, &col("a"));
/// ```
///
/// Triple negation becomes single negation:
///
/// ```
/// use vortex_expr::{not, col};
/// use vortex_expr::forms::nnf::nnf;
///
/// let triple_negation = not(not(not(col("a"))));
/// let nnfed = nnf(triple_negation);
/// assert_eq!(&nnfed, &not(col("a")));
/// ```
///
/// Negation at a high-level is pushed to the leaves, likely increasing the total number nodes:
///
/// ```
/// use vortex_expr::{not, col, and, or};
/// use vortex_expr::forms::nnf::nnf;
///
/// assert_eq!(
///     &nnf(not(and(col("a"), col("b")))),
///     &or(not(col("a")), not(col("b")))
/// );
/// ```
///
/// In Vortex, NNF is extended beyond simple Boolean operators to any Boolean-valued operator:
///
/// ```
/// use vortex_expr::{not, col, and, or, lt, lit, gt_eq};
/// use vortex_expr::forms::nnf::nnf;
///
/// assert_eq!(
///     &nnf(not(and(gt_eq(col("a"), lit(3)), col("b")))),
///     &or(lt(col("a"), lit(3)), not(col("b")))
/// );
/// ```
pub fn nnf(expr: ExprRef) -> ExprRef {
    let mut visitor = NNFVisitor::default();

    expr.transform_with_context(&mut visitor, false)
        .vortex_expect("cannot fail")
        .result()
}

/// Verifies whether the expression is in Negative Normal Form ([NNF](https://en.wikipedia.org/wiki/Negation_normal_form)).
///
/// Note that NNF isn't canonical, different expressions might be logically equivalent but different.
pub fn is_nnf(expr: &ExprRef) -> bool {
    let mut visitor = NNFValidationVisitor::default();
    expr.accept(&mut visitor).vortex_expect("never fails");
    visitor.is_nnf
}

#[derive(Default)]
struct NNFVisitor {}

impl FolderMut for NNFVisitor {
    type NodeTy = ExprRef;
    type Out = ExprRef;
    type Context = bool;

    fn visit_down(
        &mut self,
        node: &ExprRef,
        negating: bool,
    ) -> VortexResult<FoldDown<ExprRef, bool>> {
        if node.is::<NotVTable>() {
            return Ok(FoldDown::Continue(!negating));
        }

        Ok(FoldDown::Continue(negating))
    }

    fn visit_up(
        &mut self,
        node: ExprRef,
        negating: bool,
        mut new_children: Vec<ExprRef>,
    ) -> VortexResult<FoldUp<ExprRef>> {
        let new_node = if node.is::<NotVTable>() {
            debug_assert_eq!(new_children.len(), 1);
            new_children.remove(0)
        } else if let Some(binary_expr) = node.as_opt::<BinaryVTable>() {
            if !negating {
                node.with_children(new_children)?
            } else {
                let new_op = match binary_expr.op() {
                    Operator::Eq => Operator::NotEq,
                    Operator::NotEq => Operator::Eq,
                    Operator::Gt => Operator::Lte,
                    Operator::Gte => Operator::Lt,
                    Operator::Lt => Operator::Gte,
                    Operator::Lte => Operator::Gt,
                    Operator::And => Operator::Or,
                    Operator::Or => Operator::And,
                    Operator::Add => {
                        vortex_bail!("nnf: type mismatch: cannot negate addition")
                    }
                };
                let (lhs, rhs) = match binary_expr.op() {
                    Operator::Or | Operator::And => {
                        let mut negated_children = new_children;
                        debug_assert_eq!(negated_children.len(), 2);
                        let rhs = negated_children.remove(1);
                        let lhs = negated_children.remove(0);
                        (lhs, rhs)
                    }
                    _ => (binary_expr.lhs().clone(), binary_expr.rhs().clone()),
                };
                BinaryExpr::new(lhs, new_op, rhs).into_expr()
            }
        } else if negating {
            not(node)
        } else {
            node
        };

        Ok(FoldUp::Continue(new_node))
    }
}

struct NNFValidationVisitor {
    is_nnf: bool,
}

impl Default for NNFValidationVisitor {
    fn default() -> Self {
        Self { is_nnf: true }
    }
}

impl<'a> NodeVisitor<'a> for NNFValidationVisitor {
    type NodeTy = ExprRef;

    fn visit_down(&mut self, node: &'a Self::NodeTy) -> VortexResult<TraversalOrder> {
        if let Some(not_expr) = node.as_opt::<NotVTable>() {
            let is_var =
                not_expr.child().is::<GetItemVTable>() || not_expr.child().is::<LiteralVTable>();
            self.is_nnf &= is_var;
            if !is_var {
                return Ok(TraversalOrder::Stop);
            }
        }

        Ok(TraversalOrder::Continue)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::{and, col, gt_eq, lit, lt, or};

    #[rstest]
    #[case(
        and(not(and(lit(true), lit(true))), and(lit(true), lit(true))),
        and(or(not(lit(true)), not(lit(true))), and(lit(true), lit(true)),)
    )]
    #[case(not(not(col("a"))), col("a"))]
    #[case(not(not(not(col("a")))), not(col("a")))]
    #[case(not(and(col("a"), col("b"))), or(not(col("a")), not(col("b"))))]
    #[case(
        not(and(gt_eq(col("a"), lit(3)), col("b"))),
        or(lt(col("a"), lit(3)), not(col("b")))
    )]
    fn basic_nnf_test(#[case] input: ExprRef, #[case] expected: ExprRef) {
        let output = nnf(input.clone());

        assert_eq!(
            &output, &expected,
            "\nOriginal expr: {input}]\nRewritten expr: {output}\nexpected expr:{expected}"
        );
        assert!(is_nnf(&output));
    }
}
