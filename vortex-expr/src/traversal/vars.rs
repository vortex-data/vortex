use vortex_error::VortexResult;
use vortex_utils::aliases::hash_set::HashSet;

use crate::traversal::{NodeVisitor, TraversalOrder};
use crate::{ExprRef, Identifier, VarVTable};

#[derive(Default)]
pub struct VarsCollector {
    ids: HashSet<Identifier>,
}

impl VarsCollector {
    pub fn new() -> Self {
        Self {
            ids: HashSet::new(),
        }
    }

    pub fn into_vars(self) -> HashSet<Identifier> {
        self.ids
    }
}

impl NodeVisitor<'_> for VarsCollector {
    type NodeTy = ExprRef;

    fn visit_up(&mut self, node: &ExprRef) -> VortexResult<TraversalOrder> {
        if let Some(var) = node.as_opt::<VarVTable>() {
            self.ids.insert(var.var().clone());
        }
        Ok(TraversalOrder::Continue)
    }
}
