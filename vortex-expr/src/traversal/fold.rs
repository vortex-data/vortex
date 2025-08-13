use vortex_error::VortexResult;

use crate::traversal::{Node, Transformed};

pub trait NodeFolder {
    type NodeTy: Node;
    type Result;
    // type Ctx;

    // fn visit_down(&mut self, _node: &'a Self::NodeTy) -> VortexResult<TraversalOrder> {
    //     Ok(TraversalOrder::Continue)
    // }

    fn visit_up(
        &mut self,
        _node: Self::NodeTy,
        _children: Vec<Self::Result>,
    ) -> VortexResult<Transformed<Self::Result>>;
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexExpect;

    use super::*;
    use crate::traversal::NodeExt;
    use crate::{BinaryVTable, ExprRef, LiteralVTable, Operator, checked_add, lit, vortex_bail};

    struct AddFold;
    impl NodeFolder for AddFold {
        type NodeTy = ExprRef;
        type Result = i32;

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
    fn test() {
        let expr = checked_add(checked_add(lit(1), lit(2)), lit(3));

        let mut folder = AddFold;
        let result = expr.fold(&mut folder).unwrap();
        assert_eq!(result.value, 6);
    }
}
