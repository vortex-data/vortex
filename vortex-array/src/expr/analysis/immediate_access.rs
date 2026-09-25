// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;

use crate::dtype::FieldName;
use crate::dtype::StructFields;
use crate::expr::BoundExpression;
use crate::expr::analysis::AnnotationFn;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::select::Select;

/// Returns the free top-level fields for bound expression nodes.
pub fn make_bound_free_field_annotator(
    scope: &StructFields,
) -> impl AnnotationFn<BoundExpression, Annotation = FieldName> {
    move |expr: &BoundExpression| {
        let Some(scalar_fn) = expr.as_scalar() else {
            return scope.names().iter().cloned().collect();
        };

        if let Some(selection) = scalar_fn.as_opt::<Select>() {
            if expr.children()[0].is_root() {
                return selection
                    .normalize_to_included_fields(scope.names())
                    .vortex_expect("Select fields must be valid for scope")
                    .into_iter()
                    .collect();
            }
        } else if let Some(field_name) = scalar_fn.as_opt::<GetItem>()
            && expr.children()[0].is_root()
        {
            return vec![field_name.clone()];
        }

        vec![]
    }
}
