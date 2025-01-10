use vortex_array::aliases::hash_set::HashSet;
use vortex_dtype::Field;
use vortex_error::VortexResult;

use crate::traversal::{NodeVisitor, TraversalOrder};
use crate::{Column, ExprRef, Select};

pub struct ReferenceCollector<'a> {
    fields: HashSet<&'a Field>,
}

impl<'a> ReferenceCollector<'a> {
    pub fn new() -> Self {
        Self {
            fields: HashSet::new(),
        }
    }

    pub fn with_set(set: HashSet<&'a Field>) -> Self {
        Self { fields: set }
    }

    pub fn into_fields(self) -> HashSet<&'a Field> {
        self.fields
    }
}

impl<'a> NodeVisitor<'a> for ReferenceCollector<'a> {
    type NodeTy = ExprRef;

    fn visit_up(&mut self, node: &'a ExprRef) -> VortexResult<TraversalOrder> {
        if let Some(col) = node.as_any().downcast_ref::<Column>() {
            self.fields.insert(col.field());
        }
        if let Some(sel) = node.as_any().downcast_ref::<Select>() {
            self.fields.extend(sel.fields().fields());
        }
        Ok(TraversalOrder::Continue)
    }
}
