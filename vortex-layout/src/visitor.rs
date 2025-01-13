use std::fmt;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_expr::traversal::{DynNode, Node, NodeVisitor, TraversalOrder};

use crate::LayoutReader;

impl DynNode for dyn LayoutReader {
    fn arc_children(&self) -> VortexResult<Vec<&Arc<Self>>> {
        self.children()
    }
}

pub struct LayoutVisitor<'a, 'b> {
    display: &'a mut Formatter<'b>,
}

impl NodeVisitor<'_> for LayoutVisitor<'_, '_> {
    type NodeTy = Arc<dyn LayoutReader + 'static>;

    fn visit_down(&mut self, node: &Self::NodeTy) -> VortexResult<TraversalOrder> {
        node.layout().fmt(self.display)?;
        self.display.write_str("\n")?;
        Ok(TraversalOrder::Continue)
    }
}

pub struct LayoutReaderDebug(pub Arc<dyn LayoutReader + 'static>);

impl Debug for LayoutReaderDebug {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "LayoutReader")?;
        let mut vis = LayoutVisitor { display: f };
        self.0.accept(&mut vis).unwrap();
        Ok(())
    }
}
