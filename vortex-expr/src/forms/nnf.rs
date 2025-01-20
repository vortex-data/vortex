use vortex_error::VortexResult;

use crate::traversal::{FoldDown, FoldUp, FolderMut, Node as _};
use crate::{not, BinaryExpr, ExprRef, Not, Operator};

/// Return an equivalent expression in Negative Normal Form (NNF).
///
/// In NNF, [Not] expressions may only contain terminal nodes such as [Literal](crate::Literal) or
/// [GetItem](crate::GetItem). They *may not* contain [BinaryExpr], [Not], etc.
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
/// let nnfed = nnf(double_negation).unwrap();
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
/// let nnfed = nnf(triple_negation).unwrap();
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
///     &nnf(not(and(col("a"), col("b")))).unwrap(),
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
///     &nnf(not(and(gt_eq(col("a"), lit(3)), col("b")))).unwrap(),
///     &or(lt(col("a"), lit(3)), not(col("b")))
/// );
/// ```
pub fn nnf(expr: ExprRef) -> VortexResult<ExprRef> {
    let mut visitor = NNFVisitor::default();
    Ok(expr.transform_with_context(&mut visitor, false)?.result())
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
        let node_any = node.as_any();
        if node_any.is::<Not>() {
            return Ok(FoldDown::Continue(!negating));
        } else if let Some(binary_expr) = node_any.downcast_ref::<BinaryExpr>() {
            match binary_expr.op() {
                Operator::And | Operator::Or => {
                    return Ok(FoldDown::Continue(negating));
                }
                _ => {}
            }
        }

        Ok(FoldDown::Continue(negating))
    }

    fn visit_up(
        &mut self,
        node: ExprRef,
        negating: bool,
        mut new_children: Vec<ExprRef>,
    ) -> VortexResult<FoldUp<ExprRef>> {
        let node_any = node.as_any();

        let new_node = if node_any.is::<Not>() {
            debug_assert_eq!(new_children.len(), 1);
            new_children.remove(0)
        } else if let Some(binary_expr) = node_any.downcast_ref::<BinaryExpr>() {
            if !negating {
                node
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
                BinaryExpr::new_expr(lhs, new_op, rhs)
            }
        } else if negating {
            not(node)
        } else {
            node
        };

        Ok(FoldUp::Continue(new_node))
    }
}
