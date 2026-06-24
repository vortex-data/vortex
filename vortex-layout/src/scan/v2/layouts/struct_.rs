// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scan2 vtable support for struct layouts: plans field access expressions.
//!
//! A struct node treats field access as scalar expression pushdown:
//! `get_item(field, root())` pushes to the field child, and `select(...)`
//! becomes a virtual struct node assembled from pushed child nodes.

use std::fmt;
use std::sync::Arc;

use parking_lot::Mutex;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
use vortex_array::expr::get_item;
use vortex_array::expr::is_root;
use vortex_array::expr::root;
use vortex_array::expr::transform::replace;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::pack::Pack;
use vortex_array::scalar_fn::fns::root::Root;
use vortex_array::scalar_fn::fns::select::Select;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::layout_v2::Layout;
use crate::layout_v2::LayoutRef;
use crate::layouts_v2::struct_::Struct;
use crate::scan::plan::ApplyScanPlan;
use crate::scan::plan::MaskScanPlan;
use crate::scan::plan::PrepareCtx;
use crate::scan::plan::PreparedReadRef;
use crate::scan::plan::PushCtx;
use crate::scan::plan::ScanPlan;
use crate::scan::plan::ScanPlanRef;
use crate::scan::plan::ScanStateRef;
use crate::scan::plan::StateCtx;
use crate::scan::plan::StructValueScanPlan;
use crate::scan::plan::literal_scan_plan;
use crate::scan::plan::request::ScanRequest;
use crate::scan::v2::referenced_fields;
use crate::scan::v2::struct_fields;

pub(crate) fn new_scan_plan(
    layout: Layout<Struct>,
    _req: &mut ScanRequest,
    session: &VortexSession,
) -> VortexResult<ScanPlanRef> {
    let validity = layout
        .dtype()
        .is_nullable()
        .then(|| {
            layout
                .child(0)?
                .new_scan_plan(&mut ScanRequest::empty(), session)
        })
        .transpose()?;
    let fields = struct_fields(layout.dtype())?;
    let children = Mutex::new(vec![None; fields.nfields()]);
    let field_child_offset = usize::from(layout.dtype().is_nullable());
    Ok(Arc::new(StructScanPlan {
        layout: layout.to_layout(),
        session: session.clone(),
        fields,
        children,
        field_child_offset,
        validity,
    }))
}

/// Plans struct field expressions through child scan plans.
pub struct StructScanPlan {
    layout: LayoutRef,
    session: VortexSession,
    fields: StructFields,
    children: Mutex<Vec<Option<ScanPlanRef>>>,
    field_child_offset: usize,
    validity: Option<ScanPlanRef>,
}

impl ScanPlan for StructScanPlan {
    fn init_state(&self, _cx: &mut StateCtx<'_>) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(()))
    }

    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanPlanRef>> {
        let scope = &self.fields;
        if let Some(literal) = literal_scan_plan(expr, self.layout.row_count()) {
            return Ok(Some(literal));
        }
        if is_root(expr) {
            return self.push_struct(scope.names().clone(), cx).map(Some);
        }
        if let Some(name) = root_field(expr) {
            let child = self.child_field(name)?;
            return Ok(self.apply_validity(child.try_push_expr(&root(), cx)?));
        }
        if let Some(selection) = expr.as_opt::<Select>()
            && expr.child(0).is::<Root>()
        {
            let names = selection.normalize_to_included_fields(scope.names())?;
            return self.push_struct(names, cx).map(Some);
        }
        if let Some(pack) = expr.as_opt::<Pack>()
            && is_direct_field_projection(expr, &pack.names)
        {
            return self.push_struct(pack.names.clone(), cx).map(Some);
        }
        let fields = referenced_fields(expr, scope);
        if let [name] = fields.as_slice()
            && can_push_as_single_field(expr, name)
        {
            let scoped = replace(expr.clone(), &get_item(name.clone(), root()), root());
            let child = self.child_field(name)?;
            return Ok(self.apply_validity(child.try_push_expr(&scoped, cx)?));
        }
        let input = self.push_struct(fields.clone().into(), cx)?;
        Ok(Some(Arc::new(ApplyScanPlan::new(input, expr.clone()))))
    }

    fn prepare_read(
        self: Arc<Self>,
        _cx: &mut PrepareCtx,
    ) -> VortexResult<Option<PreparedReadRef>> {
        Ok(None)
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "struct({})", self.layout.nchildren())
    }
}

impl StructScanPlan {
    /// Apply this struct's validity to a pushed single-field node.
    ///
    /// The single-field fast paths route straight to a child node, bypassing
    /// the parent struct's validity. When the struct is nullable we wrap the
    /// child in a [`MaskScanPlan`] so the parent's null mask is applied to the
    /// child result, mirroring the v1 struct reader's `array.mask(validity)`.
    fn apply_validity(&self, pushed: Option<ScanPlanRef>) -> Option<ScanPlanRef> {
        match (pushed, &self.validity) {
            (Some(node), Some(validity)) => {
                Some(Arc::new(MaskScanPlan::new(node, Arc::clone(validity))))
            }
            (pushed, _) => pushed,
        }
    }

    fn child_field(&self, name: &FieldName) -> VortexResult<ScanPlanRef> {
        let Some(field_idx) = self.fields.find(name) else {
            vortex_bail!("field {name} not found in struct layout")
        };
        self.child_field_by_index(field_idx, name)
    }

    fn child_field_by_index(
        &self,
        field_idx: usize,
        name: &FieldName,
    ) -> VortexResult<ScanPlanRef> {
        let mut children = self.children.lock();
        let Some(slot) = children.get_mut(field_idx) else {
            vortex_bail!("field {name} not found in struct layout")
        };
        if let Some(hit) = slot {
            return Ok(Arc::clone(hit));
        }

        let child_idx = field_idx + self.field_child_offset;
        let child = self
            .layout
            .child(child_idx)?
            .new_scan_plan(&mut ScanRequest::empty(), &self.session)?;
        *slot = Some(Arc::clone(&child));
        Ok(child)
    }

    fn push_struct(&self, names: FieldNames, cx: &mut PushCtx) -> VortexResult<ScanPlanRef> {
        let field_indices = if names == self.fields.names() {
            (0..names.len()).collect::<Vec<_>>()
        } else {
            names
                .iter()
                .map(|name| {
                    self.fields
                        .find(name)
                        .ok_or_else(|| vortex_error::vortex_err!("field {name} not found"))
                })
                .collect::<VortexResult<Vec<_>>>()?
        };
        let fields = names
            .iter()
            .zip(field_indices)
            .map(|(name, field_idx)| {
                let child = self.child_field_by_index(field_idx, name)?;
                child
                    .try_push_expr(&root(), cx)?
                    .ok_or_else(|| vortex_error::vortex_err!("field {name} did not push root"))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(Arc::new(StructValueScanPlan::new(
            names,
            fields,
            self.validity.clone(),
        )))
    }
}

fn root_field(expr: &Expression) -> Option<&FieldName> {
    let name = expr.as_opt::<GetItem>()?;
    expr.child(0).is::<Root>().then_some(name)
}

fn is_direct_field_projection(expr: &Expression, names: &FieldNames) -> bool {
    if names.len() != expr.children().len() {
        return false;
    }

    names
        .iter()
        .zip(expr.children().iter())
        .all(|(name, child)| root_field(child).is_some_and(|field| field == name))
}

fn can_push_as_single_field(expr: &Expression, name: &FieldName) -> bool {
    if let Some(field) = root_field(expr) {
        return field == name;
    }
    if expr.is::<Root>() {
        return false;
    }
    expr.children()
        .iter()
        .all(|child| can_push_as_single_field(child, name))
}

#[cfg(test)]
mod tests {
    use vortex_array::dtype::Nullability;
    use vortex_array::expr::pack;

    use super::*;

    #[test]
    fn pack_of_root_field_is_direct_projection() {
        let expr = pack(
            [("labels", get_item("labels", root()))],
            Nullability::NonNullable,
        );
        let pack = expr.as_opt::<Pack>().expect("pack expression");

        assert!(is_direct_field_projection(&expr, &pack.names));
        assert!(can_push_as_single_field(&expr, &FieldName::from("labels")));
    }

    #[test]
    fn pack_of_root_is_not_child_field_projection() {
        let expr = pack([("labels", root())], Nullability::NonNullable);
        let pack = expr.as_opt::<Pack>().expect("pack expression");

        assert!(!is_direct_field_projection(&expr, &pack.names));
        assert!(!can_push_as_single_field(&expr, &FieldName::from("labels")));
    }
}
