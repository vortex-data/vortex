// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::traversal::Node;

/// Use to indicate the control flow of the fold on the downwards pass.
/// `Stop` indicates that the fold should stop.
/// `Skip` indicates that the fold should skip the children of the current node.
/// `Continue` indicates that the fold should continue.
#[derive(Debug)]
pub enum FoldDown<R> {
    Continue,
    Stop(R),
    Skip(R),
}

/// Use to indicate the control flow of the fold on the downwards pass.
/// In the case of Continue, the context is passed on to the children nodes.
/// Other cases are the same as `FoldDown`.
#[derive(Debug)]
pub enum FoldDownContext<C, R> {
    Continue(C),
    Stop(R),
    Skip(R),
}

/// Use to indicate the control flow of the fold on the upwards pass.
/// `Stop` indicates that the fold should stop at the current position and return the result.
#[derive(Debug)]
pub enum FoldUp<R> {
    Continue(R),
    Stop(R),
}

impl<R> FoldUp<R> {
    pub fn value(self) -> R {
        match self {
            Self::Continue(r) => r,
            Self::Stop(r) => r,
        }
    }
}

/// Use to implement the folding a tree like structure in a pre-order traversal.
///
/// At each point on the way down, the `visit_down` method is called. If it returns `Skip`,
/// the children of the current node are skipped. If it returns `Stop`, the fold is stopped.
/// If it returns `Continue`, the children of the current node are visited.
///
/// At each point on the way up, the `visit_up` method is called. If it returns `Stop`,
/// the fold stops.
///
/// On the way up the folded children are passed to the `visit_up` method along with the current node.
///
/// Note: this trait is not safe to use for graphs with a cycle.
pub trait NodeFolderContext {
    type NodeTy: Node;
    type Result;
    type Context;

    /// visit_down is called when a node is first encountered, in a pre-order traversal.
    /// If the node's children are to be skipped, return Skip.
    /// If the node should stop traversal, return Stop.
    /// Otherwise, return Continue.
    fn visit_down(
        &mut self,
        _ctx: &Self::Context,
        _node: &Self::NodeTy,
    ) -> VortexResult<FoldDownContext<Self::Context, Self::Result>>;

    /// visit_up is called when a node is last encountered, in a pre-order traversal.
    /// If the node should stop traversal, return Stop.
    /// Otherwise, return Continue.
    fn visit_up(
        &mut self,
        _node: Self::NodeTy,
        _context: &Self::Context,
        _children: Vec<Self::Result>,
    ) -> VortexResult<FoldUp<Self::Result>>;
}

/// This trait is used to implement a fold (see `NodeFolderContext`), but without a context.
pub trait NodeFolder {
    type NodeTy: Node;
    type Result;

    /// visit_down is called when a node is first encountered, in a pre-order traversal.
    /// If the node's children are to be skipped, return Skip.
    /// If the node should stop traversal, return Stop.
    /// Otherwise, return Continue.
    fn visit_down(&mut self, _node: &Self::NodeTy) -> VortexResult<FoldDown<Self::Result>> {
        Ok(FoldDown::Continue)
    }

    /// visit_up is called when a node is last encountered, in a pre-order traversal.
    /// If the node should stop traversal, return Stop.
    /// Otherwise, return Continue.
    fn visit_up(
        &mut self,
        _node: Self::NodeTy,
        _children: Vec<Self::Result>,
    ) -> VortexResult<FoldUp<Self::Result>>;
}

pub(crate) struct NodeFolderContextWrapper<'a, T>
where
    T: NodeFolder,
{
    pub inner: &'a mut T,
}

impl<T: NodeFolder> NodeFolderContext for NodeFolderContextWrapper<'_, T> {
    type NodeTy = T::NodeTy;
    type Result = T::Result;
    type Context = ();

    fn visit_down(
        &mut self,
        _ctx: &Self::Context,
        _node: &Self::NodeTy,
    ) -> VortexResult<FoldDownContext<Self::Context, Self::Result>> {
        match self.inner.visit_down(_node)? {
            FoldDown::Continue => Ok(FoldDownContext::Continue(())),
            FoldDown::Stop(r) => Ok(FoldDownContext::Stop(r)),
            FoldDown::Skip(r) => Ok(FoldDownContext::Skip(r)),
        }
    }

    fn visit_up(
        &mut self,
        _node: Self::NodeTy,
        _context: &Self::Context,
        _children: Vec<Self::Result>,
    ) -> VortexResult<FoldUp<Self::Result>> {
        self.inner.visit_up(_node, _children)
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::{VortexExpect, vortex_bail};

    use super::*;
    use crate::Expression;
    use crate::exprs::binary::{Binary, checked_add, gt};
    use crate::exprs::literal::{Literal, lit};
    use crate::exprs::operators::Operator;
    use crate::traversal::NodeExt;

    struct AddFold;
    impl NodeFolder for AddFold {
        type NodeTy = Expression;
        type Result = i32;

        fn visit_down(&mut self, node: &'_ Self::NodeTy) -> VortexResult<FoldDown<Self::Result>> {
            if let Some(lit) = node.as_opt::<Literal>() {
                let v = lit
                    .data()
                    .as_primitive()
                    .typed_value::<i32>()
                    .vortex_expect("i32");

                if v == 5 {
                    return Ok(FoldDown::Stop(5));
                }
            }

            if let Some(binary) = node.as_opt::<Binary>()
                && binary.operator() == Operator::Gt
            {
                return Ok(FoldDown::Skip(0));
            }

            Ok(FoldDown::Continue)
        }

        fn visit_up(
            &mut self,
            node: Self::NodeTy,
            children: Vec<Self::Result>,
        ) -> VortexResult<FoldUp<Self::Result>> {
            if let Some(lit) = node.as_opt::<Literal>() {
                let v = lit
                    .data()
                    .as_primitive()
                    .typed_value::<i32>()
                    .vortex_expect("i32");
                Ok(FoldUp::Continue(v))
            } else if let Some(binary) = node.as_opt::<Binary>() {
                if binary.operator() == Operator::Add {
                    Ok(FoldUp::Continue(children[0] + children[1]))
                } else {
                    vortex_bail!("not a valid operator")
                }
            } else {
                vortex_bail!("not a valid type")
            }
        }
    }

    #[test]
    fn test_fold() {
        let expr = checked_add(checked_add(lit(1), lit(2)), lit(3));

        let mut folder = AddFold;
        let result = expr.fold(&mut folder).unwrap().value();
        assert_eq!(result, 6);
    }

    #[test]
    fn test_stop_value() {
        let expr = checked_add(checked_add(lit(1), lit(5)), lit(3));

        let mut folder = AddFold;
        let result = expr.fold(&mut folder).unwrap().value();
        assert_eq!(result, 5);
    }

    #[test]
    fn test_skip_value() {
        let expr = checked_add(gt(lit(1), lit(2)), lit(3));

        let mut folder = AddFold;
        let result = expr.fold(&mut folder).unwrap().value();
        assert_eq!(result, 3);
    }

    #[test]
    fn test_control_flow_value() {
        let expr = checked_add(gt(lit(1), lit(5)), lit(3));

        let mut folder = AddFold;
        let result = expr.fold(&mut folder).unwrap().value();
        assert_eq!(result, 3);
    }
}
