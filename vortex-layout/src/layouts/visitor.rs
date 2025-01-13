use vortex_error::VortexResult;
use vortex_expr::traversal::{FoldUp, Folder, FolderMut, Node, NodeVisitor, TraversalOrder};

use crate::LayoutEncoding;

impl Node for LayoutEncoding {
    fn accept<'a, V: NodeVisitor<'a, NodeTy = Self>>(
        &'a self,
        _visitor: &mut V,
    ) -> VortexResult<TraversalOrder> {
        todo!()
    }

    fn accept_with_context<'a, V: Folder<'a, NodeTy = Self>>(
        &'a self,
        visitor: &mut V,
        context: V::Context,
    ) -> VortexResult<FoldUp<V::Out>> {
        todo!()
    }
}
