// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reads [`VariantLayout`]s, pruning the shredded tree to the paths an expression reads.
//!
//! An expression that uses the Variant column only through `variant_get(root, path, T)` calls
//! with concrete types `T` reads, for each path, the shredded columns on that path: a typed leaf,
//! or the result of `variant_get` on a nested Variant field for the rest of the path. Those values
//! are placed at their paths in a pruned shredded tree, which is combined with the core storage
//! into a Variant array the expression is evaluated over. The Variant `variant_get` kernel then
//! serves each path from the pruned tree exactly as it would from the full tree, falling back to
//! the core storage for rows the shredded values leave null. Any other use of the column reads
//! the full shredded tree.

use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::try_join;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VariantArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::arrays::variant::VariantArraySlotsExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::bound;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::expr::traversal::TraversalOrder;
use vortex_array::scalar_fn::fns::variant_get::VariantGet;
use vortex_array::scalar_fn::fns::variant_get::VariantPath;
use vortex_array::scalar_fn::fns::variant_get::VariantPathElement;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ArrayFuture;
use crate::LayoutReader;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::LazyReaderChildren;
use crate::RowSplits;
use crate::SplitRange;
use crate::layouts::variant::CORE_CHILD;
use crate::layouts::variant::SHREDDED_CHILD;
use crate::layouts::variant::VariantLayout;
use crate::segments::SegmentSource;

pub(super) struct VariantReader {
    layout: VariantLayout,
    name: Arc<str>,
    session: VortexSession,
    lazy_children: LazyReaderChildren,
}

/// How an expression over the Variant column reads the column.
enum Plan {
    /// Read the core storage and the full shredded tree.
    Full,
    /// Read the core storage and evaluate `shredded` over the shredded tree: a `pack` with one
    /// field per path, named as in `leaves`, holding the value of that path's typed shredded leaf
    /// or nested `variant_get`.
    Pruned {
        shredded: Option<BoundExpression>,
        leaves: Vec<(Vec<FieldName>, FieldName)>,
    },
}

impl VariantReader {
    pub(super) fn new(
        layout: VariantLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
        ctx: LayoutReaderContext,
    ) -> Self {
        let lazy_children = LazyReaderChildren::new(
            Arc::clone(layout.children()),
            vec![layout.dtype().clone(), layout.shredded_dtype().clone()],
            vec![
                Arc::from(format!("{name}.core")),
                Arc::from(format!("{name}.shredded")),
            ],
            segment_source,
            session.clone(),
            ctx,
        );
        Self {
            layout,
            name,
            session,
            lazy_children,
        }
    }

    fn core(&self) -> VortexResult<&LayoutReaderRef> {
        self.lazy_children.get(CORE_CHILD)
    }

    fn shredded(&self) -> VortexResult<&LayoutReaderRef> {
        self.lazy_children.get(SHREDDED_CHILD)
    }

    fn plan(&self, expr: &BoundExpression) -> VortexResult<Plan> {
        let mut gets = Vec::new();
        if !collect_variant_gets(expr, &mut gets) {
            return Ok(Plan::Full);
        }
        let mut unique: Vec<(Vec<FieldName>, DType)> = Vec::with_capacity(gets.len());
        for get in gets {
            if !unique.contains(&get) {
                unique.push(get);
            }
        }
        let gets = unique;
        // A path read as two types, or a path through another read path, would need two values
        // at one place in the pruned tree.
        for (idx, (path, _)) in gets.iter().enumerate() {
            if gets[idx + 1..]
                .iter()
                .any(|(other, _)| other.starts_with(path) || path.starts_with(other))
            {
                return Ok(Plan::Full);
            }
        }

        let shredded_dtype = self.layout.shredded_dtype();
        let shredded_root = BoundExpression::new_root(shredded_dtype.clone());
        let mut fields = Vec::new();
        let mut leaves = Vec::new();
        for (path, dtype) in gets {
            // Paths without a shredded value are served from the core storage alone, as the
            // kernel does when the full tree does not represent them.
            if let Some(expr) = shredded_value(shredded_dtype, &shredded_root, &path, &dtype) {
                let name = FieldName::from(format!("p{}", fields.len()));
                fields.push((name.clone(), expr));
                leaves.push((path, name));
            }
        }
        let shredded = (!fields.is_empty()).then(|| bound::pack(fields, Nullability::NonNullable));
        Ok(Plan::Pruned { shredded, leaves })
    }

    /// The Variant array an expression over the column is evaluated against, holding the core
    /// storage and the shredded tree, or the part of it `plan` reads.
    fn variant_future(
        &self,
        row_range: &Range<u64>,
        plan: Plan,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let core_root = BoundExpression::new_root(self.layout.dtype().clone());
        let core = self
            .core()?
            .projection_evaluation(row_range, &core_root, mask.clone())?;

        let shredded = match &plan {
            Plan::Full => {
                let root = BoundExpression::new_root(self.layout.shredded_dtype().clone());
                Some(
                    self.shredded()?
                        .projection_evaluation(row_range, &root, mask)?,
                )
            }
            Plan::Pruned {
                shredded: Some(shredded),
                ..
            } => Some(
                self.shredded()?
                    .projection_evaluation(row_range, shredded, mask)?,
            ),
            Plan::Pruned { shredded: None, .. } => None,
        };

        let session = self.session.clone();
        Ok(async move {
            let (core, shredded) = match shredded {
                Some(shredded) => {
                    let (core, shredded) = try_join!(core, shredded)?;
                    (core, Some(shredded))
                }
                None => (core.await?, None),
            };
            let mut ctx = session.create_execution_ctx();
            let core = unshredded_core(core, &mut ctx)?;
            let len = core.len();
            let shredded = match (plan, shredded) {
                (Plan::Full, shredded) => shredded,
                (Plan::Pruned { leaves, .. }, Some(values)) => {
                    let values = values.execute::<StructArray>(&mut ctx)?;
                    let leaves = leaves
                        .iter()
                        .map(|(path, name)| {
                            let value = values.unmasked_field_by_name(name.as_ref())?.clone();
                            Ok((path.as_slice(), value))
                        })
                        .collect::<VortexResult<Vec<_>>>()?;
                    Some(pruned_tree(&leaves, len)?)
                }
                (Plan::Pruned { .. }, None) => None,
            };
            Ok(VariantArray::try_new(core, shredded)?.into_array())
        }
        .boxed())
    }
}

/// Collects the `(path, dtype)` of every `variant_get(root, path, dtype)` in `expr`.
///
/// Returns `false` when `expr` uses the column other than through such calls with a concrete
/// non-Variant dtype and an object-field-only path.
fn collect_variant_gets(expr: &BoundExpression, gets: &mut Vec<(Vec<FieldName>, DType)>) -> bool {
    if expr.is_root() {
        return false;
    }
    if let Some(options) = expr.as_opt::<VariantGet>()
        && expr.child(0).is_root()
    {
        let Some(dtype) = options.dtype().filter(|dtype| !dtype.is_variant()) else {
            return false;
        };
        let Some(path) = options
            .path()
            .elements()
            .iter()
            .map(|element| match element {
                VariantPathElement::Field(name) => Some(name.clone()),
                VariantPathElement::Index(_) => None,
            })
            .collect::<Option<Vec<_>>>()
        else {
            return false;
        };
        gets.push((path, dtype.clone()));
        return true;
    }
    expr.children()
        .iter()
        .all(|child| collect_variant_gets(child, gets))
}

/// Collects the distinct `variant_get(root, ..)` nodes of `expr`.
fn collect_get_nodes(expr: &BoundExpression, gets: &mut Vec<BoundExpression>) {
    if expr.is::<VariantGet>() && expr.child(0).is_root() {
        if !gets.iter().any(|get| is_same_get(get, expr)) {
            gets.push(expr.clone());
        }
        return;
    }
    for child in expr.children() {
        collect_get_nodes(child, gets);
    }
}

/// Whether `node` is the `variant_get(root, ..)` call `get`.
fn is_same_get(get: &BoundExpression, node: &BoundExpression) -> bool {
    node.as_opt::<VariantGet>().is_some()
        && node.children().len() == 1
        && node.child(0).is_root()
        && node.as_opt::<VariantGet>() == get.as_opt::<VariantGet>()
}

/// The expression over the shredded tree of dtype `shredded_dtype` reading `path` as `dtype`, as
/// the Variant `variant_get` kernel serves it from shredded storage: the typed leaf at `path`, or
/// `variant_get` on the partially shredded field the path reaches.
///
/// Returns `None` when the shredded tree does not serve the path, so the kernel answers it from
/// the core storage alone.
fn shredded_value(
    shredded_dtype: &DType,
    shredded_root: &BoundExpression,
    path: &[FieldName],
    dtype: &DType,
) -> Option<BoundExpression> {
    let mut current_dtype = shredded_dtype.clone();
    let mut expr = shredded_root.clone();
    for (idx, name) in path.iter().enumerate() {
        match &current_dtype {
            DType::Struct(fields, _) => {
                let field_dtype = fields.field(name)?;
                expr = bound::get_item(name.clone(), expr);
                current_dtype = field_dtype;
            }
            DType::Variant(_) => {
                let rest =
                    VariantPath::new(path[idx..].iter().cloned().map(VariantPathElement::Field));
                return Some(bound::variant_get(expr, rest, Some(dtype.clone())));
            }
            _ => return None,
        }
    }
    match current_dtype {
        DType::Variant(_) => Some(bound::variant_get(
            expr,
            VariantPath::root(),
            Some(dtype.clone()),
        )),
        leaf if leaf.as_nullable() == dtype.as_nullable() => Some(expr),
        _ => None,
    }
}

/// Unwraps core storage that was written as an unshredded canonical Variant, so the kernels of its
/// underlying encoding apply.
fn unshredded_core(core: ArrayRef, ctx: &mut vortex_array::ExecutionCtx) -> VortexResult<ArrayRef> {
    if !core.is::<vortex_array::arrays::Variant>() {
        return Ok(core);
    }
    let variant = core.execute::<VariantArray>(ctx)?;
    Ok(match variant.shredded() {
        None => variant.core_storage().clone(),
        Some(_) => variant.into_array(),
    })
}

/// A non-nullable struct tree holding each value at its path.
fn pruned_tree(leaves: &[(&[FieldName], ArrayRef)], len: usize) -> VortexResult<ArrayRef> {
    if let [(path, value)] = leaves
        && path.is_empty()
    {
        return Ok(value.clone());
    }
    let mut groups: Vec<(FieldName, Vec<(&[FieldName], ArrayRef)>)> = Vec::new();
    for (path, value) in leaves {
        let (head, tail) = path
            .split_first()
            .ok_or_else(|| vortex_error::vortex_err!("conflicting Variant paths"))?;
        match groups.iter_mut().find(|(name, _)| name == head) {
            Some((_, group)) => group.push((tail, value.clone())),
            None => groups.push((head.clone(), vec![(tail, value.clone())])),
        }
    }
    let names = groups
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    let fields = groups
        .iter()
        .map(|(_, group)| pruned_tree(group, len))
        .collect::<VortexResult<Vec<_>>>()?;
    Ok(StructArray::try_new(names.into(), fields, len, Validity::NonNullable)?.into_array())
}

impl LayoutReader for VariantReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        self.core()?
            .register_splits(field_mask, split_range, splits)?;
        self.shredded()?
            .register_splits(&[FieldMask::All], split_range, splits)
    }

    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &BoundExpression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        Ok(MaskFuture::ready(mask))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        let values = self.projection_evaluation(row_range, expr, mask.clone())?;
        let session = self.session.clone();
        Ok(MaskFuture::new(mask.len(), async move {
            let (mask, values) = try_join!(mask, values)?;
            let mut ctx = session.create_execution_ctx();
            let values = values.null_as_false().execute(&mut ctx)?;
            Ok(mask.intersect_by_rank(&values))
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let plan = self.plan(expr)?;
        let separate_gets = matches!(plan, Plan::Pruned { .. });
        let variant = self.variant_future(row_range, plan, mask)?;
        let expr = expr.clone();
        let session = self.session.clone();
        Ok(async move {
            let variant = variant.await?;
            if !separate_gets {
                return variant.apply_bound(&expr);
            }
            // Execute each `variant_get` on its own so the rest of the expression sees the encoded
            // values the kernel returns, e.g. comparing dictionary-encoded strings by their
            // dictionary rather than after canonicalizing them.
            let mut ctx = session.create_execution_ctx();
            let mut names = Vec::new();
            let mut values = Vec::new();
            let mut gets: Vec<BoundExpression> = Vec::new();
            collect_get_nodes(&expr, &mut gets);
            for get in &gets {
                names.push(FieldName::from(format!("g{}", names.len())));
                values.push(
                    variant
                        .clone()
                        .apply_bound(get)?
                        .execute::<ArrayRef>(&mut ctx)?,
                );
            }
            let len = variant.len();
            let gets_struct =
                StructArray::try_new(names.clone().into(), values, len, Validity::NonNullable)?
                    .into_array();
            let gets_root = BoundExpression::new_root(gets_struct.dtype().clone());
            let rewritten = expr
                .transform_down(
                    |node| match gets.iter().position(|get| is_same_get(get, &node)) {
                        Some(idx) => Ok(Transformed {
                            value: bound::get_item(names[idx].clone(), gets_root.clone()),
                            changed: true,
                            order: TraversalOrder::Skip,
                        }),
                        None => Ok(Transformed::no(node)),
                    },
                )?
                .into_inner();
            gets_struct.apply_bound(&rewritten)
        }
        .boxed())
    }
}
