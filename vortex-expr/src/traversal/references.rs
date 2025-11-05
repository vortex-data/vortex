// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::FieldName;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_set::HashSet;

use crate::Expression;
use crate::exprs::get_item::GetItem;
use crate::exprs::select::Select;
use crate::traversal::{NodeVisitor, TraversalOrder};

#[derive(Default)]
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
    type NodeTy = Expression;

    fn visit_up(&mut self, node: &Expression) -> VortexResult<TraversalOrder> {
        if let Some(get_item) = node.as_opt::<GetItem>() {
            self.fields.insert(get_item.data().clone());
        }
        if let Some(sel) = node.as_opt::<Select>() {
            self.fields.extend(sel.data().field_names().iter().cloned());
        }
        Ok(TraversalOrder::Continue)
    }
}
