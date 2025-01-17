mod references;
mod visitor;

use itertools::Itertools;
pub use references::ReferenceCollector;
use vortex_error::VortexResult;

use crate::ExprRef;

/// Define a data fusion inspired traversal pattern for visiting nodes in a `Node`,
/// for now only VortexExpr.
///
/// This traversal is a pre-order traversal.
/// There are control traversal controls `TraversalOrder`:
/// - `Skip`: Skip visiting the children of the current node.
/// - `Stop`: Stop visiting any more nodes in the traversal.
/// - `Continue`: Continue with the traversal as expected.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraversalOrder {
    // In a top-down traversal, skip visiting the children of the current node.
    // In the bottom-up phase of the traversal this does nothing (for now).
    Skip,

    // Stop visiting any more nodes in the traversal.
    Stop,

    // Continue with the traversal as expected.
    Continue,
}

#[derive(Debug, Clone)]
pub struct TransformResult<T> {
    pub result: T,
    order: TraversalOrder,
    changed: bool,
}

impl<T> TransformResult<T> {
    pub fn yes(result: T) -> Self {
        Self {
            result,
            order: TraversalOrder::Continue,
            changed: true,
        }
    }

    pub fn no(result: T) -> Self {
        Self {
            result,
            order: TraversalOrder::Continue,
            changed: false,
        }
    }
}

pub trait NodeVisitor<'a> {
    type NodeTy: Node;

    fn visit_down(&mut self, _node: &'a Self::NodeTy) -> VortexResult<TraversalOrder> {
        Ok(TraversalOrder::Continue)
    }

    fn visit_up(&mut self, _node: &'a Self::NodeTy) -> VortexResult<TraversalOrder> {
        Ok(TraversalOrder::Continue)
    }
}

pub trait MutNodeVisitor {
    type NodeTy: Node;

    fn visit_down(&mut self, _node: &Self::NodeTy) -> VortexResult<TraversalOrder> {
        Ok(TraversalOrder::Continue)
    }

    fn visit_up(&mut self, _node: Self::NodeTy) -> VortexResult<TransformResult<Self::NodeTy>>;
}

pub enum FoldDown<Out, Context> {
    /// Abort the entire traversal and immediately return the result.
    Abort(Out),
    /// Skip visiting children of the current node and return the result to the parent's `fold_up`.
    SkipChildren(Out),
    /// Continue visiting the `fold_down` of the children of the current node.
    Continue(Context),
}

#[derive(Debug)]
pub enum FoldUp<Out> {
    /// Abort the entire traversal and immediately return the result.
    Abort(Out),
    /// Continue visiting the `fold_up` of the parent node.
    Continue(Out),
}

impl<Out> FoldUp<Out> {
    pub fn result(self) -> Out {
        match self {
            FoldUp::Abort(out) => out,
            FoldUp::Continue(out) => out,
        }
    }
}

pub trait Folder<'a> {
    type NodeTy: Node;
    type Out;
    type Context: Clone;

    fn visit_down(
        &mut self,
        _node: &'a Self::NodeTy,
        context: Self::Context,
    ) -> VortexResult<FoldDown<Self::Out, Self::Context>> {
        Ok(FoldDown::Continue(context))
    }

    fn visit_up(
        &mut self,
        node: &'a Self::NodeTy,
        context: Self::Context,
        children: Vec<Self::Out>,
    ) -> VortexResult<FoldUp<Self::Out>>;
}

pub trait FolderMut {
    type NodeTy: Node;
    type Out;
    type Context: Clone;

    fn visit_down(
        &mut self,
        _node: &Self::NodeTy,
        context: Self::Context,
    ) -> VortexResult<FoldDown<Self::Out, Self::Context>> {
        Ok(FoldDown::Continue(context))
    }

    fn visit_up(
        &mut self,
        node: Self::NodeTy,
        context: Self::Context,
        children: Vec<Self::Out>,
    ) -> VortexResult<FoldUp<Self::Out>>;
}

pub trait Node: Sized {
    fn accept<'a, V: NodeVisitor<'a, NodeTy = Self>>(
        &'a self,
        _visitor: &mut V,
    ) -> VortexResult<TraversalOrder>;

    fn accept_with_context<'a, V: Folder<'a, NodeTy = Self>>(
        &'a self,
        visitor: &mut V,
        context: V::Context,
    ) -> VortexResult<FoldUp<V::Out>>;

    fn transform<V: MutNodeVisitor<NodeTy = Self>>(
        self,
        _visitor: &mut V,
    ) -> VortexResult<TransformResult<Self>>;

    fn transform_with_context<V: FolderMut<NodeTy = Self>>(
        self,
        _visitor: &mut V,
        _context: V::Context,
    ) -> VortexResult<FoldUp<V::Out>>;
}

impl Node for ExprRef {
    // A pre-order traversal.
    fn accept<'a, V: NodeVisitor<'a, NodeTy = ExprRef>>(
        &'a self,
        visitor: &mut V,
    ) -> VortexResult<TraversalOrder> {
        let mut ord = visitor.visit_down(self)?;
        if ord == TraversalOrder::Stop {
            return Ok(TraversalOrder::Stop);
        }
        if ord == TraversalOrder::Skip {
            return Ok(TraversalOrder::Continue);
        }
        for child in self.children() {
            if ord != TraversalOrder::Continue {
                return Ok(ord);
            }
            ord = child.accept(visitor)?;
        }
        if ord == TraversalOrder::Stop {
            return Ok(TraversalOrder::Stop);
        }
        visitor.visit_up(self)
    }

    fn accept_with_context<'a, V: Folder<'a, NodeTy = Self>>(
        &'a self,
        visitor: &mut V,
        context: V::Context,
    ) -> VortexResult<FoldUp<V::Out>> {
        let children = match visitor.visit_down(self, context.clone())? {
            FoldDown::Abort(out) => return Ok(FoldUp::Abort(out)),
            FoldDown::SkipChildren(out) => return Ok(FoldUp::Continue(out)),
            FoldDown::Continue(child_context) => {
                let mut new_children = Vec::with_capacity(self.children().len());
                for child in self.children() {
                    match child.accept_with_context(visitor, child_context.clone())? {
                        FoldUp::Abort(out) => return Ok(FoldUp::Abort(out)),
                        FoldUp::Continue(out) => new_children.push(out),
                    }
                }
                new_children
            }
        };

        visitor.visit_up(self, context, children)
    }

    // A pre-order transform, with an option to ignore sub-tress (using visit_down).
    fn transform<V: MutNodeVisitor<NodeTy = Self>>(
        self,
        visitor: &mut V,
    ) -> VortexResult<TransformResult<Self>> {
        let mut ord = visitor.visit_down(&self)?;
        if ord == TraversalOrder::Stop {
            return Ok(TransformResult {
                result: self,
                order: TraversalOrder::Stop,
                changed: false,
            });
        }
        let (children, ord, changed) = if ord == TraversalOrder::Continue {
            let mut new_children = Vec::with_capacity(self.children().len());
            let mut changed = false;
            for child in self.children() {
                match ord {
                    TraversalOrder::Continue | TraversalOrder::Skip => {
                        let TransformResult {
                            result: new_child,
                            order: child_order,
                            changed: child_changed,
                        } = child.clone().transform(visitor)?;
                        new_children.push(new_child);
                        ord = child_order;
                        changed |= child_changed;
                    }
                    TraversalOrder::Stop => new_children.push(child.clone()),
                }
            }
            (new_children, ord, changed)
        } else {
            (
                self.children().into_iter().cloned().collect_vec(),
                ord,
                false,
            )
        };

        if ord == TraversalOrder::Continue {
            let up = visitor.visit_up(self.replacing_children(children))?;
            Ok(TransformResult::yes(up.result))
        } else {
            Ok(TransformResult {
                result: self.replacing_children(children),
                order: ord,
                changed,
            })
        }
    }

    fn transform_with_context<V: FolderMut<NodeTy = Self>>(
        self,
        visitor: &mut V,
        context: V::Context,
    ) -> VortexResult<FoldUp<V::Out>> {
        let children = match visitor.visit_down(&self, context.clone())? {
            FoldDown::Abort(out) => return Ok(FoldUp::Abort(out)),
            FoldDown::SkipChildren(out) => return Ok(FoldUp::Continue(out)),
            FoldDown::Continue(child_context) => {
                let mut new_children = Vec::with_capacity(self.children().len());
                for child in self.children() {
                    match child
                        .clone()
                        .transform_with_context(visitor, child_context.clone())?
                    {
                        FoldUp::Abort(out) => return Ok(FoldUp::Abort(out)),
                        FoldUp::Continue(out) => new_children.push(out),
                    }
                }
                new_children
            }
        };

        visitor.visit_up(self, context, children)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::aliases::hash_set::HashSet;
    use vortex_error::VortexResult;

    use crate::traversal::visitor::pre_order_visit_down;
    use crate::traversal::{MutNodeVisitor, Node, NodeVisitor, TransformResult, TraversalOrder};
    use crate::{
        col, BinaryExpr, ExprRef, FieldName, GetItem, Literal, Operator, VortexExpr, VortexExprExt,
    };

    #[derive(Default)]
    pub struct ExprLitCollector<'a>(pub Vec<&'a ExprRef>);

    impl<'a> NodeVisitor<'a> for ExprLitCollector<'a> {
        type NodeTy = ExprRef;

        fn visit_down(&mut self, node: &'a ExprRef) -> VortexResult<TraversalOrder> {
            if node.as_any().downcast_ref::<Literal>().is_some() {
                self.0.push(node)
            }
            Ok(TraversalOrder::Continue)
        }

        fn visit_up(&mut self, _node: &'a ExprRef) -> VortexResult<TraversalOrder> {
            Ok(TraversalOrder::Continue)
        }
    }

    #[derive(Default)]
    pub struct ExprColToLit(i32);

    impl MutNodeVisitor for ExprColToLit {
        type NodeTy = ExprRef;

        fn visit_up(&mut self, node: Self::NodeTy) -> VortexResult<TransformResult<Self::NodeTy>> {
            let col = node.as_any().downcast_ref::<GetItem>();
            if col.is_some() {
                let id = self.0;
                self.0 += 1;
                Ok(TransformResult::yes(Literal::new_expr(id)))
            } else {
                Ok(TransformResult::no(node))
            }
        }
    }

    #[test]
    fn expr_deep_visitor_test() {
        let col1: Arc<dyn VortexExpr> = col("col1");
        let lit1 = Literal::new_expr(1);
        let expr = BinaryExpr::new_expr(col1.clone(), Operator::Eq, lit1.clone());
        let lit2 = Literal::new_expr(2);
        let expr = BinaryExpr::new_expr(expr, Operator::And, lit2);
        let mut printer = ExprLitCollector::default();
        expr.accept(&mut printer).unwrap();
        assert_eq!(printer.0.len(), 2);
    }

    #[test]
    fn expr_deep_mut_visitor_test() {
        let col1: Arc<dyn VortexExpr> = col("col1");
        let col2: Arc<dyn VortexExpr> = col("col2");
        let expr = BinaryExpr::new_expr(col1.clone(), Operator::Eq, col2.clone());
        let lit2 = Literal::new_expr(2);
        let expr = BinaryExpr::new_expr(expr, Operator::And, lit2);
        let mut printer = ExprColToLit::default();
        let new = expr.transform(&mut printer).unwrap();
        assert!(new.changed);

        let expr = new.result;

        let mut printer = ExprLitCollector::default();
        expr.accept(&mut printer).unwrap();
        assert_eq!(printer.0.len(), 3);
    }

    #[test]
    fn expr_skip_test() {
        let col1: Arc<dyn VortexExpr> = col("col1");
        let col2: Arc<dyn VortexExpr> = col("col2");
        let expr1 = BinaryExpr::new_expr(col1.clone(), Operator::Eq, col2.clone());
        let col3: Arc<dyn VortexExpr> = col("col3");
        let col4: Arc<dyn VortexExpr> = col("col4");
        let expr2 = BinaryExpr::new_expr(col3.clone(), Operator::NotEq, col4.clone());
        let expr = BinaryExpr::new_expr(expr1, Operator::And, expr2);

        let mut nodes = Vec::new();
        expr.accept(&mut pre_order_visit_down(|node: &ExprRef| {
            if node.as_any().downcast_ref::<GetItem>().is_some() {
                nodes.push(node)
            }
            if let Some(bin) = node.as_any().downcast_ref::<BinaryExpr>() {
                if bin.op() == Operator::Eq {
                    return Ok(TraversalOrder::Skip);
                }
            }
            Ok(TraversalOrder::Continue)
        }))
        .unwrap();

        assert_eq!(
            nodes
                .into_iter()
                .map(|x| x.references())
                .fold(HashSet::new(), |acc, x| acc.union(&x).cloned().collect()),
            HashSet::from_iter(vec![FieldName::from("col3"), FieldName::from("col4")])
        );
    }

    #[test]
    fn expr_stop_test() {
        let col1: Arc<dyn VortexExpr> = col("col1");
        let col2: Arc<dyn VortexExpr> = col("col2");
        let expr1 = BinaryExpr::new_expr(col1.clone(), Operator::Eq, col2.clone());
        let col3: Arc<dyn VortexExpr> = col("col3");
        let col4: Arc<dyn VortexExpr> = col("col4");
        let expr2 = BinaryExpr::new_expr(col3.clone(), Operator::NotEq, col4.clone());
        let expr = BinaryExpr::new_expr(expr1, Operator::And, expr2);

        let mut nodes = Vec::new();
        expr.accept(&mut pre_order_visit_down(|node: &ExprRef| {
            if node.as_any().downcast_ref::<GetItem>().is_some() {
                nodes.push(node)
            }
            if let Some(bin) = node.as_any().downcast_ref::<BinaryExpr>() {
                if bin.op() == Operator::Eq {
                    return Ok(TraversalOrder::Stop);
                }
            }
            Ok(TraversalOrder::Continue)
        }))
        .unwrap();

        assert_eq!(
            nodes
                .into_iter()
                .map(|x| x.references())
                .fold(HashSet::new(), |acc, x| acc.union(&x).cloned().collect()),
            HashSet::from_iter(vec![])
        );
    }
}
