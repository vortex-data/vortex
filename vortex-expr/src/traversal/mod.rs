use vortex_error::VortexResult;

use crate::{Column, ExprRef};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisitationOrder {
    Skip,
    Stop,
    Continue,
}

pub trait NodeVisitor {
    type NodeTy: Node;

    fn visit_down(&self, _node: &Self::NodeTy) -> VortexResult<VisitationOrder>;

    fn visit_up(&self, _node: &Self::NodeTy) -> VortexResult<VisitationOrder>;
}

// trait MutNodeVisitor {
//     fn visit_down(&mut self, _node: &dyn Node) -> VortexResult<()>;
//
//     fn visit_up(&mut self, _node: &dyn Node) -> VortexResult<()>;
// }

pub trait Node {
    fn accept<V: NodeVisitor<NodeTy = Self>>(
        &self,
        _visitor: &mut V,
    ) -> VortexResult<VisitationOrder>;
}

impl Node for ExprRef {
    fn accept<V: NodeVisitor<NodeTy = ExprRef>>(
        &self,
        visitor: &mut V,
    ) -> VortexResult<VisitationOrder> {
        let ord = visitor.visit_down(self)?;
        if ord == VisitationOrder::Stop {
            return Ok(VisitationOrder::Stop);
        }
        if ord == VisitationOrder::Skip {
            return Ok(VisitationOrder::Continue);
        }
        let ord = self
            .children()
            .iter()
            .try_fold(VisitationOrder::Continue, |acc, child| {
                let ord = child.accept(visitor)?;
                if ord != VisitationOrder::Continue {
                    VortexResult::Ok(ord)
                } else {
                    Ok(acc)
                }
            })?;
        if ord == VisitationOrder::Stop {
            return Ok(VisitationOrder::Stop);
        }
        visitor.visit_up(self)
    }
}

pub struct ExprPrinter();

impl NodeVisitor for ExprPrinter {
    type NodeTy = ExprRef;

    fn visit_down(&self, node: &ExprRef) -> VortexResult<VisitationOrder> {
        println!("Visiting down: {:?}", node);
        if node.as_any().downcast_ref::<Column>().is_some() {
            Ok(VisitationOrder::Skip)
        } else {
            Ok(VisitationOrder::Continue)
        }
    }

    fn visit_up(&self, node: &ExprRef) -> VortexResult<VisitationOrder> {
        println!("Visiting up: {:?}", node);
        Ok(VisitationOrder::Continue)
    }
}

// pub trait Tree {
//     fn children<F, O>(&self, _f: F) -> impl Iterator<Item = T>
//     where
//         F: FnMut(&Self) -> O,
//     {
//         iter::empty()
//     }
// }

// impl<T> Node for T
// where
//     T: Tree,
// {
//     fn accept(&self, visitor: &mut dyn NodeVisitor) -> VortexResult<VisitationOrder> {
//         visitor.visit_down(self)?;
//         for o in self.children(|child| child.accept(visitor)) {
//             if o? == VisitationOrder::Stop {
//                 return Ok(VisitationOrder::Stop);
//             }
//         }
//         visitor.visit_up(self)
//     }
// }

// trait VisitationStepper {
//     fn children<F>(&self, f: F) -> VortexResult<VisitationOrder>;
// }
