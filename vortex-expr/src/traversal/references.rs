use vortex_array::aliases::hash_set::HashSet;
use vortex_dtype::FieldName;
use vortex_error::VortexResult;

use crate::traversal::{NodeVisitor, TraversalOrder};
use crate::{ExprRef, GetItem, Select};

pub struct ReferenceCollector {
    fields: HashSet<FieldName>,
}

impl ReferenceCollector {
    pub fn new() -> Self {
        Self {
            fields: HashSet::new(),
        }
    }

    pub fn with_set(set: HashSet<FieldName>) -> Self {
        Self { fields: set }
    }

    pub fn into_fields(self) -> HashSet<FieldName> {
        self.fields
    }
}

impl NodeVisitor<'_> for ReferenceCollector {
    type NodeTy = ExprRef;

    fn visit_up(&mut self, node: &ExprRef) -> VortexResult<TraversalOrder> {
        if let Some(get_item) = node.as_any().downcast_ref::<GetItem>() {
            self.fields.insert(get_item.field().clone());
        }
        if let Some(sel) = node.as_any().downcast_ref::<Select>() {
            self.fields.extend(sel.fields().fields().iter().cloned());
        }
        Ok(TraversalOrder::Continue)
    }
}
