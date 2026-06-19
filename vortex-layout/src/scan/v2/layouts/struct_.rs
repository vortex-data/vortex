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

use crate::LayoutChildType;
use crate::layout_v2::Layout;
use crate::layout_v2::LayoutRef;
use crate::layout_v2::Struct;
use crate::scan::v2::node::ApplyScanNode;
use crate::scan::v2::node::ExpandCtx;
use crate::scan::v2::node::MaskScanNode;
use crate::scan::v2::node::PlanCtx;
use crate::scan::v2::node::PushCtx;
use crate::scan::v2::node::ReadPlanRef;
use crate::scan::v2::node::ScanNode;
use crate::scan::v2::node::ScanNodeRef;
use crate::scan::v2::node::StateCtx;
use crate::scan::v2::node::StructValueScanNode;
use crate::scan::v2::referenced_fields;
use crate::scan::v2::request::NodeRequest;
use crate::scan::v2::struct_fields;

pub(crate) fn new_scan_node(
    layout: Layout<Struct>,
    _req: &mut NodeRequest,
    cx: &ExpandCtx,
) -> VortexResult<ScanNodeRef> {
    let validity = layout
        .dtype()
        .is_nullable()
        .then(|| cx.expand(&layout.child(0)?, &mut NodeRequest::empty()))
        .transpose()?;
    Ok(Arc::new(StructScanNode {
        layout: layout.to_layout(),
        cx: cx.clone(),
        children: Mutex::new(FxHashMap::default()),
        validity,
    }))
}

/// Plans struct field expressions through child scan nodes.
pub struct StructScanNode {
    layout: LayoutRef,
    cx: ExpandCtx,
    children: Mutex<FxHashMap<FieldName, ScanNodeRef>>,
    validity: Option<ScanNodeRef>,
}

impl ScanNode for StructScanNode {
    type State = ();

    fn init_state(&self, _cx: &mut StateCtx<'_>) -> VortexResult<()> {
        Ok(())
    }

    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanNodeRef>> {
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
        Ok(Some(Arc::new(ApplyScanNode::new(input, expr.clone()))))
    }

    fn plan_read(self: Arc<Self>, _cx: &mut PlanCtx) -> VortexResult<Option<ReadPlanRef>> {
        Ok(None)
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "struct({})", self.layout.nchildren())
    }
}

impl StructScanNode {
    /// Apply this struct's validity to a pushed single-field node.
    ///
    /// The single-field fast paths route straight to a child node, bypassing
    /// the parent struct's validity. When the struct is nullable we wrap the
    /// child in a [`MaskScanNode`] so the parent's null mask is applied to the
    /// child result, mirroring the v1 struct reader's `array.mask(validity)`.
    fn apply_validity(&self, pushed: Option<ScanNodeRef>) -> Option<ScanNodeRef> {
        match (pushed, &self.validity) {
            (Some(node), Some(validity)) => {
                Some(Arc::new(MaskScanNode::new(node, Arc::clone(validity))))
            }
            (pushed, _) => pushed,
        }
    }

    fn child_field(&self, name: &FieldName) -> VortexResult<ScanNodeRef> {
        if let Some(hit) = self.children.lock().get(name) {
            return Ok(Arc::clone(hit));
        }
        for idx in 0..self.layout.nchildren() {
            if let Ok(LayoutChildType::Field(field)) = self.layout.child_type(idx)
                && field == *name
            {
                let mut req = NodeRequest::empty();
                let child = self.cx.expand(&self.layout.child(idx)?, &mut req)?;
                let mut children = self.children.lock();
                return Ok(Arc::clone(children.entry(name.clone()).or_insert(child)));
            }
        }
        vortex_bail!("field {name} not found in struct layout")
    }

    fn push_struct(&self, names: FieldNames, cx: &mut PushCtx) -> VortexResult<ScanNodeRef> {
        let fields = names
            .iter()
            .map(|name| {
                let child = self.child_field(name)?;
                child
                    .try_push_expr(&root(), cx)?
                    .ok_or_else(|| vortex_error::vortex_err!("field {name} did not push root"))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(Arc::new(StructValueScanNode::new(
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
