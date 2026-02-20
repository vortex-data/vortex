// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_utils::aliases::hash_set::HashSet;

use crate::dtype::FieldName;
use crate::expr::Expression;
use crate::expr::exprs::get_item::GetItem;
use crate::expr::exprs::select::Select;
use crate::expr::traversal::NodeVisitor;
use crate::expr::traversal::TraversalOrder;

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
        if let Some(field_name) = node.as_opt::<GetItem>() {
            self.fields.insert(field_name.clone());
        }
        if let Some(field_selection) = node.as_opt::<Select>() {
            self.fields
                .extend(field_selection.field_names().iter().cloned());
        }
        Ok(TraversalOrder::Continue)
    }
}
