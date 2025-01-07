mod references;

use itertools::Itertools;
pub use references::ReferenceCollector;
use vortex_error::VortexResult;

use crate::ExprRef;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraversalOrder {
    // In a top down traversal, skip visiting the children of the current node.
    // In a bottom up traversal,  for now this does nothing.
    Skip,

    // Stop visiting any more nodes in the traversal.
    Stop,

    // Continue with the traversal as expected.
    Continue,
}

#[derive(Debug, Clone)]
pub struct TransformResult<T> {
    result: T,
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

pub trait Node: Sized {
    fn accept<'a, V: NodeVisitor<'a, NodeTy = Self>>(
        &'a self,
        _visitor: &mut V,
    ) -> VortexResult<TraversalOrder>;

    fn transform<V: MutNodeVisitor<NodeTy = Self>>(
        self,
        _visitor: &mut V,
    ) -> VortexResult<TransformResult<Self>>;
}

impl Node for ExprRef {
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
            let up = visitor.visit_up(self)?;
            Ok(TransformResult::yes(up.result.replacing_children(children)))
        } else {
            Ok(TransformResult {
                result: self.replacing_children(children),
                order: ord,
                changed,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::aliases::hash_set::HashSet;
    use vortex_dtype::Field;
    use vortex_error::VortexResult;

    use crate::traversal::{MutNodeVisitor, Node, NodeVisitor, TransformResult, TraversalOrder};
    use crate::{BinaryExpr, Column, ExprRef, Literal, Operator, VortexExpr, VortexExprExt};

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
            let col = node.as_any().downcast_ref::<Column>();
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
        let col1: Arc<dyn VortexExpr> = Column::new_expr("col1");
        let lit1 = Literal::new_expr(1);
        let expr = BinaryExpr::new_expr(col1.clone(), Operator::Eq, lit1.clone());
        let lit2 = Literal::new_expr(2);
        let expr = BinaryExpr::new_expr(expr, Operator::And, lit2);
        let mut printer = ColumnCollectorSkip::default();
        expr.accept(&mut printer).unwrap();
        assert_eq!(printer.0.len(), 2);
    }

    #[test]
    fn expr_deep_mut_visitor_test() {
        let col1: Arc<dyn VortexExpr> = Column::new_expr("col1");
        let col2: Arc<dyn VortexExpr> = Column::new_expr("col2");
        let expr = BinaryExpr::new_expr(col1.clone(), Operator::Eq, col2.clone());
        let lit2 = Literal::new_expr(2);
        let expr = BinaryExpr::new_expr(expr, Operator::And, lit2);
        let mut printer = ExprColToLit::default();
        let new = expr.transform(&mut printer).unwrap();
        assert!(new.changed);

        let expr = new.result;

        let mut printer = ColumnCollectorSkip::default();
        expr.accept(&mut printer).unwrap();
        assert_eq!(printer.0.len(), 3);
    }

    #[derive(Default)]
    pub struct ColumnCollectorSkip<'a>(pub Vec<&'a ExprRef>);

    impl<'a> NodeVisitor<'a> for ColumnCollectorSkip<'a> {
        type NodeTy = ExprRef;

        fn visit_down(&mut self, node: &'a ExprRef) -> VortexResult<TraversalOrder> {
            if node.as_any().downcast_ref::<Column>().is_some() {
                self.0.push(node)
            }
            if let Some(bin) = node.as_any().downcast_ref::<BinaryExpr>() {
                if bin.op() == Operator::Eq {
                    return Ok(TraversalOrder::Skip);
                }
            }
            Ok(TraversalOrder::Continue)
        }
    }

    #[test]
    fn expr_skip_test() {
        let col1: Arc<dyn VortexExpr> = Column::new_expr("col1");
        let col2: Arc<dyn VortexExpr> = Column::new_expr("col2");
        let expr1 = BinaryExpr::new_expr(col1.clone(), Operator::Eq, col2.clone());
        let col3: Arc<dyn VortexExpr> = Column::new_expr("col3");
        let col4: Arc<dyn VortexExpr> = Column::new_expr("col4");
        let expr2 = BinaryExpr::new_expr(col3.clone(), Operator::NotEq, col4.clone());
        let expr = BinaryExpr::new_expr(expr1, Operator::And, expr2);
        let mut printer = ColumnCollectorSkip::default();
        expr.accept(&mut printer).unwrap();
        assert_eq!(
            printer
                .0
                .into_iter()
                .map(|x| x.references())
                .fold(HashSet::new(), |acc, x| acc.union(&x).cloned().collect()),
            HashSet::from_iter(vec![&Field::from("col3"), &Field::from("col4")])
        );
    }

    #[derive(Default)]
    pub struct ColumnCollectorStop<'a>(pub Vec<&'a ExprRef>);

    impl<'a> NodeVisitor<'a> for ColumnCollectorStop<'a> {
        type NodeTy = ExprRef;

        fn visit_down(&mut self, node: &'a ExprRef) -> VortexResult<TraversalOrder> {
            if node.as_any().downcast_ref::<Column>().is_some() {
                self.0.push(node)
            }
            if let Some(bin) = node.as_any().downcast_ref::<BinaryExpr>() {
                if bin.op() == Operator::Eq {
                    return Ok(TraversalOrder::Stop);
                }
            }
            Ok(TraversalOrder::Continue)
        }
    }

    #[test]
    fn expr_stop_test() {
        let col1: Arc<dyn VortexExpr> = Column::new_expr("col1");
        let col2: Arc<dyn VortexExpr> = Column::new_expr("col2");
        let expr1 = BinaryExpr::new_expr(col1.clone(), Operator::Eq, col2.clone());
        let col3: Arc<dyn VortexExpr> = Column::new_expr("col3");
        let col4: Arc<dyn VortexExpr> = Column::new_expr("col4");
        let expr2 = BinaryExpr::new_expr(col3.clone(), Operator::NotEq, col4.clone());
        let expr = BinaryExpr::new_expr(expr1, Operator::And, expr2);
        let mut printer = ColumnCollectorStop::default();
        expr.accept(&mut printer).unwrap();
        assert_eq!(
            printer
                .0
                .into_iter()
                .map(|x| x.references())
                .fold(HashSet::new(), |acc, x| acc.union(&x).cloned().collect()),
            HashSet::from_iter(vec![])
        );
    }
}
