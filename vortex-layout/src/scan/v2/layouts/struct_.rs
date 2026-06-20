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
use rustc_hash::FxHashMap;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
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
use vortex_scan::plan::ApplyScanPlan;
use vortex_scan::plan::MaskScanPlan;
use vortex_scan::plan::PrepareCtx;
use vortex_scan::plan::PreparedReadRef;
use vortex_scan::plan::PushCtx;
use vortex_scan::plan::ScanPlan;
use vortex_scan::plan::ScanPlanRef;
use vortex_scan::plan::StateCtx;
use vortex_scan::plan::StructValueScanPlan;
use vortex_scan::plan::request::ScanRequest;
use vortex_session::VortexSession;

use crate::LayoutChildType;
use crate::layout_v2::Layout;
use crate::layout_v2::LayoutRef;
use crate::layout_v2::Struct;
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
    Ok(Arc::new(StructScanPlan {
        layout: layout.to_layout(),
        session: session.clone(),
        children: Mutex::new(FxHashMap::default()),
        validity,
    }))
}

/// Plans struct field expressions through child scan plans.
pub struct StructScanPlan {
    layout: LayoutRef,
    session: VortexSession,
    children: Mutex<FxHashMap<FieldName, ScanPlanRef>>,
    validity: Option<ScanPlanRef>,
}

impl ScanPlan for StructScanPlan {
    type State = ();

    fn init_state(&self, _cx: &mut StateCtx<'_>) -> VortexResult<()> {
        Ok(())
    }

    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanPlanRef>> {
        let scope = struct_fields(self.layout.dtype())?;
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
            && pack.names.len() == 1
            && expr.child(0).is::<Root>()
        {
            return self.push_struct(pack.names.clone(), cx).map(Some);
        }
        let fields = referenced_fields(expr, &scope);
        if let [name] = fields.as_slice() {
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
        if let Some(hit) = self.children.lock().get(name) {
            return Ok(Arc::clone(hit));
        }
        for idx in 0..self.layout.nchildren() {
            if let Ok(LayoutChildType::Field(field)) = self.layout.child_type(idx)
                && field == *name
            {
                let child = self
                    .layout
                    .child(idx)?
                    .new_scan_plan(&mut ScanRequest::empty(), &self.session)?;
                let mut children = self.children.lock();
                return Ok(Arc::clone(children.entry(name.clone()).or_insert(child)));
            }
        }
        vortex_bail!("field {name} not found in struct layout")
    }

    fn push_struct(&self, names: FieldNames, cx: &mut PushCtx) -> VortexResult<ScanPlanRef> {
        let fields = names
            .iter()
            .map(|name| {
                let child = self.child_field(name)?;
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
