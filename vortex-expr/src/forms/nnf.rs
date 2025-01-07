use vortex_error::{vortex_bail, vortex_err, VortexResult};

use crate::traversal::{Node as _, NodeVisitor, TraversalOrder};
use crate::{not, BinaryExpr, Column, ExprRef, Literal, Not, Operator};

/// Return an equivalent expression in Negative Normal Form (NNF).
///
/// In NNF, [Not] expressions may only contain terminal nodes such as [Literal] or [Column]. They
/// *may not* contain [BinaryExpr], [Not], etc.
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
    expr.accept(&mut visitor)?;
    visitor.finish()
}

struct NNFVisitor {
    negating: Vec<bool>,
    stack: Vec<Vec<ExprRef>>,
}

impl Default for NNFVisitor {
    fn default() -> Self {
        NNFVisitor {
            negating: vec![false],
            stack: vec![vec![]],
        }
    }
}

impl NNFVisitor {
    fn finish(mut self) -> VortexResult<ExprRef> {
        if self.negating.len() != 1 || self.stack.len() != 1 {
            vortex_bail!("finish called before traversal completed");
        }

        let mut frame = self.stack.remove(0);

        if frame.len() != 1 {
            vortex_bail!("final frame contains more than one expr: {:?}", frame);
        }

        Ok(frame.remove(0))
    }
}

impl NodeVisitor<'_> for NNFVisitor {
    type NodeTy = ExprRef;

    fn visit_down(&mut self, node: &ExprRef) -> VortexResult<TraversalOrder> {
        let negating = self
            .negating
            .last()
            .ok_or_else(|| vortex_err!("negating must be non-empty"))?;
        self.stack.push(vec![]);

        let node_any = node.as_any();
        if node_any.downcast_ref::<Not>().is_some() {
            self.negating.push(!negating)
        } else if node_any.downcast_ref::<Literal>().is_some()
            || node_any.downcast_ref::<Column>().is_some()
        {
            // do nothing
        } else if let Some(binary_expr) = node_any.downcast_ref::<BinaryExpr>() {
            match binary_expr.op() {
                Operator::And | Operator::Or => self.negating.push(*negating),
                Operator::Eq
                | Operator::NotEq
                | Operator::Gt
                | Operator::Gte
                | Operator::Lt
                | Operator::Lte => self.negating.push(false),
            }
        } else {
            todo!("{:?}", node)
        }

        Ok(TraversalOrder::Continue)
    }

    fn visit_up(&mut self, node: &ExprRef) -> VortexResult<TraversalOrder> {
        let mut children = self
            .stack
            .pop()
            .ok_or_else(|| vortex_err!("stack must at least have current frame"))?;

        let node_any = node.as_any();
        let new_expr = if node_any.downcast_ref::<Not>().is_some() {
            debug_assert_eq!(children.len(), 1);
            self.negating.pop();
            children.remove(0)
        } else if node_any.downcast_ref::<Literal>().is_some()
            || node_any.downcast_ref::<Column>().is_some()
        {
            debug_assert_eq!(children.len(), 0);

            let negating = self
                .negating
                .last()
                .ok_or_else(|| vortex_err!("negating must be non-empty"))?;
            if *negating {
                not(node.clone())
            } else {
                node.clone()
            }
        } else if let Some(binary_expr) = node_any.downcast_ref::<BinaryExpr>() {
            debug_assert_eq!(children.len(), 2);

            self.negating
                .pop()
                .ok_or_else(|| vortex_err!("negating must be non-empty"))?;

            let negating = self
                .negating
                .last()
                .ok_or_else(|| vortex_err!("negating must be non-empty"))?;

            let rhs = children.remove(1);
            let lhs = children.remove(0);
            let operator = if *negating {
                binary_expr.op().inverse()
            } else {
                binary_expr.op()
            };

            BinaryExpr::new_expr(lhs, operator, rhs)
        } else {
            todo!("{:?}", node)
        };

        self.stack
            .last_mut()
            .ok_or_else(|| vortex_err!("stack is always non-empty"))?
            .push(new_expr);

        Ok(TraversalOrder::Continue)
    }
}
