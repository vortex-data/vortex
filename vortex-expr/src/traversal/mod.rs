mod references;

use itertools::Itertools;
pub use references::ReferenceCollector;
use vortex_error::VortexResult;

use crate::ExprRef;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisitationOrder {
    Skip,

    Stop,
    Continue,
}

#[derive(Debug, Clone)]
pub struct TransformResult<T> {
    result: T,
    order: VisitationOrder,
    changed: bool,
}

pub trait NodeVisitor<'a> {
    type NodeTy: Node;

    fn visit_down(&mut self, _node: &'a Self::NodeTy) -> VortexResult<VisitationOrder> {
        Ok(VisitationOrder::Continue)
    }

    fn visit_up(&mut self, _node: &'a Self::NodeTy) -> VortexResult<VisitationOrder> {
        Ok(VisitationOrder::Continue)
    }
}

pub trait MutNodeVisitor {
    type NodeTy: Node;

    fn visit_down(&mut self, _node: &Self::NodeTy) -> VortexResult<VisitationOrder> {
        Ok(VisitationOrder::Continue)
    }

    fn visit_up(&mut self, _node: Self::NodeTy) -> VortexResult<TransformResult<Self::NodeTy>>;
}

pub trait Node: Sized {
    fn accept<'a, V: NodeVisitor<'a, NodeTy = Self>>(
        &'a self,
        _visitor: &mut V,
    ) -> VortexResult<VisitationOrder>;

    fn transform<V: MutNodeVisitor<NodeTy = Self>>(
        self,
        _visitor: &mut V,
    ) -> VortexResult<TransformResult<Self>>;
}

impl Node for ExprRef {
    fn accept<'a, V: NodeVisitor<'a, NodeTy = ExprRef>>(
        &'a self,
        visitor: &mut V,
    ) -> VortexResult<VisitationOrder> {
        let mut ord = visitor.visit_down(self)?;
        if ord == VisitationOrder::Stop {
            return Ok(VisitationOrder::Stop);
        }
        if ord == VisitationOrder::Skip {
            return Ok(VisitationOrder::Continue);
        }
        for child in self.children() {
            if ord != VisitationOrder::Continue {
                return Ok(ord);
            }
            ord = child.accept(visitor)?;
        }
        if ord == VisitationOrder::Stop {
            return Ok(VisitationOrder::Stop);
        }
        visitor.visit_up(self)
    }

    fn transform<V: MutNodeVisitor<NodeTy = Self>>(
        self,
        visitor: &mut V,
    ) -> VortexResult<TransformResult<Self>> {
        let mut ord = visitor.visit_down(&self)?;
        if ord == VisitationOrder::Stop {
            return Ok(TransformResult {
                result: self,
                order: VisitationOrder::Stop,
                changed: false,
            });
        }
        let (children, ord, changed) = if ord == VisitationOrder::Continue {
            let mut new_children = Vec::with_capacity(self.children().len());
            let mut changed = false;
            for child in self.children() {
                match ord {
                    VisitationOrder::Continue | VisitationOrder::Skip => {
                        let TransformResult {
                            result: new_child,
                            order: child_order,
                            changed: child_changed,
                        } = child.clone().transform(visitor)?;
                        new_children.push(new_child);
                        ord = child_order;
                        changed |= child_changed;
                    }
                    VisitationOrder::Stop => new_children.push(child.clone()),
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

        if ord == VisitationOrder::Continue {
            let up = visitor.visit_up(self)?;
            Ok(TransformResult {
                result: up.result.replacing_children(children),
                order: up.order,
                changed: true,
            })
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

    use vortex_error::VortexResult;

    use crate::traversal::{MutNodeVisitor, Node, NodeVisitor, TransformResult, VisitationOrder};
    use crate::{BinaryExpr, Column, ExprRef, Literal, Operator, VortexExpr};

    #[derive(Default)]
    pub struct ExprCollector<'a>(Vec<&'a ExprRef>);

    impl<'a> ExprCollector<'a> {
        pub fn into_children(self) -> Vec<&'a ExprRef> {
            self.0
        }
    }

    impl<'a> NodeVisitor<'a> for ExprCollector<'a> {
        type NodeTy = ExprRef;

        fn visit_down(&mut self, node: &'a ExprRef) -> VortexResult<VisitationOrder> {
            if node.as_any().downcast_ref::<Literal>().is_some() {
                self.0.push(node)
            }
            Ok(VisitationOrder::Continue)
        }

        fn visit_up(&mut self, _node: &'a ExprRef) -> VortexResult<VisitationOrder> {
            Ok(VisitationOrder::Continue)
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
                Ok(TransformResult {
                    result: Literal::new_expr(id),
                    order: VisitationOrder::Continue,
                    changed: true,
                })
            } else {
                Ok(TransformResult {
                    result: node,
                    order: VisitationOrder::Continue,
                    changed: false,
                })
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
        let mut printer = ExprCollector::default();
        expr.accept(&mut printer).unwrap();
        assert_eq!(printer.into_children().len(), 2);
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

        let mut printer = ExprCollector::default();
        expr.accept(&mut printer).unwrap();
        assert_eq!(printer.into_children().len(), 3);
    }
}
