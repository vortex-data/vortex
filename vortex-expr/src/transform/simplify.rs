use vortex_error::VortexResult;

use crate::traversal::{FoldChildren, FoldUp, FolderMut, Node};
use crate::{ExprRef, GetItem, Pack};

pub struct Simplify;

impl Simplify {
    pub fn simplify(e: ExprRef) -> VortexResult<ExprRef> {
        let mut folder = Simplify;
        e.transform_with_context(&mut folder, ())
            .map(|e| e.result())
    }
}

impl FolderMut for Simplify {
    type NodeTy = ExprRef;
    type Out = ExprRef;
    type Context = ();

    fn visit_up(
        &mut self,
        node: Self::NodeTy,
        _context: Self::Context,
        children: FoldChildren<Self::Out>,
    ) -> VortexResult<FoldUp<Self::Out>> {
        if let Some(get_item) = node.as_any().downcast_ref::<GetItem>() {
            if let Some(pack) = get_item.child().as_any().downcast_ref::<Pack>() {
                let expr = pack.field(get_item.field())?;
                return Ok(FoldUp::Continue(expr));
            }
        }
        Ok(FoldUp::Continue(
            node.replacing_children(children.contained_children()),
        ))
    }
}
