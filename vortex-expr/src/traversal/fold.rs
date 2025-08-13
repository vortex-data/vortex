// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::traversal::{FoldDown, FoldUp, Node};

pub trait NodeFolder {
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
    ) -> VortexResult<FoldDown<Self::Context, Self::Result>>;

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

#[cfg(test)]
mod tests {
    use vortex_error::VortexExpect;

    use super::*;
    use crate::traversal::FoldDown::Skip;
    use crate::traversal::NodeExt;
    use crate::{
        BinaryVTable, ExprRef, LiteralVTable, Operator, checked_add, gt, lit, vortex_bail,
    };

    struct AddFold;
    impl NodeFolder for AddFold {
        type NodeTy = ExprRef;
        type Result = i32;
        type Context = ();

        fn visit_down(
            &mut self,
            _: &Self::Context,
            node: &'_ Self::NodeTy,
        ) -> VortexResult<FoldDown<Self::Context, Self::Result>> {
            if let Some(lit) = node.as_opt::<LiteralVTable>() {
                let v = lit
                    .value()
                    .as_primitive()
                    .typed_value::<i32>()
                    .vortex_expect("i32");

                if v == 5 {
                    return Ok(FoldDown::Stop(5));
                }
            }

            if let Some(binary) = node.as_opt::<BinaryVTable>()
                && binary.op() == Operator::Gt {
                    return Ok(Skip(0));
                }

            Ok(FoldDown::Continue(()))
        }

        fn visit_up(
            &mut self,
            node: Self::NodeTy,
            (): &Self::Context,
            children: Vec<Self::Result>,
        ) -> VortexResult<FoldUp<Self::Result>> {
            if let Some(lit) = node.as_opt::<LiteralVTable>() {
                let v = lit
                    .value()
                    .as_primitive()
                    .typed_value::<i32>()
                    .vortex_expect("i32");
                Ok(FoldUp::Continue(v))
            } else if let Some(binary) = node.as_opt::<BinaryVTable>() {
                if binary.op() == Operator::Add {
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
        let result = expr.fold(&(), &mut folder).unwrap().value();
        assert_eq!(result, 6);
    }

    #[test]
    fn test_stop_value() {
        let expr = checked_add(checked_add(lit(1), lit(5)), lit(3));

        let mut folder = AddFold;
        let result = expr.fold(&(), &mut folder).unwrap().value();
        assert_eq!(result, 5);
    }

    #[test]
    fn test_skip_value() {
        let expr = checked_add(gt(lit(1), lit(2)), lit(3));

        let mut folder = AddFold;
        let result = expr.fold(&(), &mut folder).unwrap().value();
        assert_eq!(result, 3);
    }

    #[test]
    fn test_control_flow_value() {
        let expr = checked_add(gt(lit(1), lit(5)), lit(3));

        let mut folder = AddFold;
        let result = expr.fold(&(), &mut folder).unwrap().value();
        assert_eq!(result, 3);
    }
}
