use vortex_error::VortexResult;

use crate::traversal::{FoldDown, FoldUp, Node, Transformed};

pub trait NodeFolder<'a> {
    type NodeTy: Node;
    type Result;
    // type Ctx;

    fn visit_down(&mut self, _node: &'a Self::NodeTy) -> VortexResult<(FoldDown<Self::Result>)> {
        Ok(FoldDown::Continue)
    }

    fn visit_up(
        &mut self,
        _node: Self::NodeTy,
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
    impl NodeFolder<'_> for AddFold {
        type NodeTy = ExprRef;
        type Result = i32;

        fn visit_down(&mut self, node: &'_ Self::NodeTy) -> VortexResult<(FoldDown<Self::Result>)> {
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

            if let Some(binary) = node.as_opt::<BinaryVTable>() {
                if binary.op() == Operator::Gt {
                    return Ok(Skip(0));
                }
            }

            Ok(FoldDown::Continue)
        }

        fn visit_up(
            &mut self,
            node: Self::NodeTy,
            children: Vec<Self::Result>,
        ) -> VortexResult<Transformed<Self::Result>> {
            if let Some(lit) = node.as_opt::<LiteralVTable>() {
                let v = lit
                    .value()
                    .as_primitive()
                    .typed_value::<i32>()
                    .vortex_expect("i32");
                Ok(Transformed::yes(v))
            } else if let Some(binary) = node.as_opt::<BinaryVTable>() {
                if binary.op() == Operator::Add {
                    Ok(Transformed::yes(children[0] + children[1]))
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
        let result = expr.fold(&mut folder).unwrap();
        assert_eq!(result.value, 6);
    }

    #[test]
    fn test_stop_value() {
        let expr = checked_add(checked_add(lit(1), lit(5)), lit(3));

        let mut folder = AddFold;
        let result = expr.fold(&mut folder).unwrap();
        assert_eq!(result.value, 5);
    }

    #[test]
    fn test_skip_value() {
        let expr = checked_add(gt(lit(1), lit(2)), lit(3));

        let mut folder = AddFold;
        let result = expr.fold(&mut folder).unwrap();
        assert_eq!(result.value, 3);
    }

    #[test]
    fn test_control_flow_value() {
        let expr = checked_add(gt(lit(1), lit(5)), lit(3));

        let mut folder = AddFold;
        let result = expr.fold(&mut folder).unwrap();
        assert_eq!(result.value, 3);
    }
}
