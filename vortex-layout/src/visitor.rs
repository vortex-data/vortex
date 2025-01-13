use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_expr::traversal::DynNode;

use crate::LayoutReader;

impl DynNode for dyn LayoutReader {
    fn arc_children(&self) -> VortexResult<Vec<&Arc<Self>>> {
        self.children()
    }
}
