// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(crate) mod execute;
mod expr;
#[cfg(test)]
mod tests;

use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Lambda;
use vortex_array::expr::bound;
use vortex_array::expr::get_item;
use vortex_array::expr::list_transform;
use vortex_array::expr::root;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::expr::traversal::TraversalOrder;
use vortex_array::expr::var;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::is_null::IsNull;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use self::execute::ListSource;
use self::expr::ListRead;
pub use self::expr::ListTransforms;
use self::expr::is_validity_read;
use crate::execution::operators::UNBOUNDED;
use crate::execution::pipeline::Chain;
use crate::execution::pipeline::Compiler;
use crate::plan::Eval;
use crate::plan::EvalPlan;
use crate::plan::PackPlan;
use crate::plan::Plan;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanSlots;
use crate::plan::PlanVTable;
use crate::plan::optimizer::PlanParentReduceRule;
use crate::plan::selection::selected_slice;

const ELEMENTS: usize = 0;
const OFFSETS: usize = 1;
const VALIDITY: usize = 2;

/// Assembles a list from slots `[elements, offsets, validity]`, with validity present exactly
/// when the list is nullable.
///
/// The elements live in a row domain of their own: list `i` holds the elements from
/// `offsets[i]` to `offsets[i + 1]`, so the element rows of a range of lists are known only once
/// their offsets are read. Compiled over a range of lists, a list reads the offsets of the range
/// and the validity of its selected lists, then compiles the elements plan over the elements of
/// the selected valid lists alone and reads them as a scan of their own. Compiling therefore
/// registers the segments of the offsets and the validity, but not those of the elements.
///
/// An [`Eval`] above a list reads only the children its expression needs: the validity for
/// `is_null($)`, the offsets and the validity for `list_length($)`, and for the lambdas of
/// `list_transform($, lambda)` nodes the elements that the lambdas read.
#[derive(Clone, Debug)]
pub struct ListPack;

/// Operator-specific list assembly data.
#[derive(Clone, Debug)]
pub struct ListPackData;

/// A plan that assembles a list from its children.
pub type ListPackPlan = Plan<ListPack>;

impl ListPackPlan {
    /// Creates a list assembly from potentially unbuilt slots without validation.
    ///
    /// # Safety
    ///
    /// `slots` must be `[elements, offsets, validity]`. `dtype` must be a list whose element
    /// dtype matches the elements child. The offsets child must be a non-nullable integer with
    /// `row_count + 1` rows. A non-nullable boolean validity child with `row_count` rows must be
    /// present exactly when `dtype` is nullable.
    pub unsafe fn from_slots_unchecked(dtype: DType, row_count: u64, slots: PlanSlots) -> Self {
        PlanParts {
            vtable: ListPack,
            dtype,
            row_count,
            slots,
            data: ListPackData,
        }
        .into_typed()
    }

    /// Creates a list assembly from `elements` and `offsets`.
    ///
    /// `validity` is required exactly when `nullability` is [`Nullability::Nullable`]. The row
    /// domain is one fewer than the number of offsets.
    pub fn try_new(
        nullability: Nullability,
        row_count: u64,
        elements: PlanRef,
        offsets: PlanRef,
        validity: Option<PlanRef>,
    ) -> VortexResult<Self> {
        let dtype = DType::List(Arc::new(elements.dtype().clone()), nullability);
        let slots = [Some(elements), Some(offsets), validity]
            .into_iter()
            .collect();
        validate_slots(&dtype, row_count, &slots)?;

        // SAFETY: All slot shape invariants were validated above.
        Ok(unsafe { Self::from_slots_unchecked(dtype, row_count, slots) })
    }

    /// Returns the plan producing list elements.
    pub fn elements(&self) -> VortexResult<PlanRef> {
        self.slot_required(ELEMENTS)
    }

    /// Returns the plan producing list offsets.
    pub fn offsets(&self) -> VortexResult<PlanRef> {
        self.slot_required(OFFSETS)
    }

    /// Returns the plan producing list validity, if the list is nullable.
    pub fn validity(&self) -> VortexResult<Option<PlanRef>> {
        self.slot(VALIDITY)
    }
}

impl PlanVTable for ListPack {
    type PlanData = ListPackData;
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.list_pack");
        *ID
    }

    fn metadata(_plan: &Plan<Self>) -> Option<Self::Metadata> {
        // Nullability is recoverable from the plan dtype.
        Some(EmptyMetadata)
    }

    fn with_slots(
        plan: &Plan<Self>,
        slots: &PlanSlots,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        validate_slots(plan.dtype(), plan.row_count(), slots)
    }

    fn slot_name(_plan: &Plan<Self>, slot: usize) -> Cow<'_, str> {
        match slot {
            ELEMENTS => Cow::Borrowed("elements"),
            OFFSETS => Cow::Borrowed("offsets"),
            VALIDITY => Cow::Borrowed("validity"),
            _ => Cow::Owned(format!("slot[{slot}]")),
        }
    }

    /// Reads the offsets from the first selected list to the last and the validity of the
    /// selected lists, as inlets `[offsets, validity]` of a source that reads the elements once
    /// those have arrived.
    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        let Some((span, selected)) = selected_slice(mask, 0..mask.len()) else {
            return Ok(None);
        };
        let lists = rows.start + u64::try_from(span.start)?..rows.start + u64::try_from(span.end)?;
        let offsets = compiler
            .compile(
                &plan.offsets()?,
                lists.start..lists.end + 1,
                &Mask::new_true(span.len() + 1),
            )?
            .ok_or_else(|| vortex_err!("ListPack offsets produced no rows"))?;
        let mut chains = vec![offsets];
        if let Some(validity) = plan.validity()? {
            chains.push(
                compiler
                    .compile(&validity, lists, &selected)?
                    .ok_or_else(|| vortex_err!("ListPack validity produced no rows"))?,
            );
        }
        let source = ListSource::new(
            plan.clone(),
            selected,
            compiler.session().clone(),
            compiler.values().clone(),
        );
        Ok(Some(compiler.join_with(chains, |_| UNBOUNDED, source)))
    }
}

fn validate_slots(dtype: &DType, row_count: u64, slots: &PlanSlots) -> VortexResult<()> {
    let elements_dtype = dtype
        .as_list_element_opt()
        .ok_or_else(|| vortex_err!("ListPack output dtype must be a list, got {dtype}"))?;
    slots.check_len("ListPack", VALIDITY + 1)?;

    let elements = slots.required(ELEMENTS)?;
    if elements.dtype() != elements_dtype.as_ref() {
        vortex_bail!(
            "ListPack elements child has dtype {} but the list element dtype is {}",
            elements.dtype(),
            elements_dtype
        );
    }

    let offsets = slots.required(OFFSETS)?;
    if !offsets.dtype().is_int() || offsets.dtype().is_nullable() {
        vortex_bail!(
            "ListPack offsets child must have a non-nullable integer dtype, got {}",
            offsets.dtype()
        );
    }
    let offsets_row_count = row_count
        .checked_add(1)
        .ok_or_else(|| vortex_err!("ListPack offsets row count overflow"))?;
    if offsets.row_count() != offsets_row_count {
        vortex_bail!(
            "ListPack offsets child has {} rows but must have {offsets_row_count}",
            offsets.row_count()
        );
    }

    match (dtype.is_nullable(), slots.get(VALIDITY)?) {
        (true, Some(validity)) => {
            let validity_dtype = DType::Bool(Nullability::NonNullable);
            if validity.dtype() != &validity_dtype {
                vortex_bail!(
                    "ListPack validity child has dtype {} but must have dtype {validity_dtype}",
                    validity.dtype()
                );
            }
            if validity.row_count() != row_count {
                vortex_bail!(
                    "ListPack validity child has {} rows but the plan has {row_count}",
                    validity.row_count()
                );
            }
        }
        (false, None) => {}
        (true, None) => vortex_bail!("Nullable ListPack has no validity child"),
        (false, Some(_)) => vortex_bail!("Non-nullable ListPack has a validity child"),
    }
    Ok(())
}

/// Reads only the children of a [`ListPack`] that the expression of an [`Eval`] above it needs.
///
/// - An expression that reads at most the validity, through `is_null($)` and `is_not_null($)`,
///   evaluates over the validity child, or over no child at all for a non-nullable list.
/// - One that also reads the lengths, through `list_length($)`, evaluates over lists of empty
///   structs, which read only the offsets and the validity.
/// - One that reads the elements only through the lambdas of `list_transform($, lambda)` nodes
///   evaluates their bodies over the elements child, and the rest over the lists of the results.
///   So `list_transform($, x -> x.sku)` reads the column of `sku` alone. The bodies of several
///   lambdas are packed into one struct, so the lambdas share one read of the offsets and the
///   validity, and each node takes its field from the lists of those structs.
///
/// A list reads only the elements of its selected valid lists, so a body runs on no element that
/// `list_transform` would not run it on, even when it can fail.
#[derive(Debug)]
pub(crate) struct ExpressionListPackRule;

impl PlanParentReduceRule<ListPack> for ExpressionListPackRule {
    type Parent = Eval;

    fn reduce_parent(
        &self,
        child: &ListPackPlan,
        parent: &Plan<Eval>,
        _child_idx: usize,
        _session: &VortexSession,
    ) -> VortexResult<Option<PlanRef>> {
        let expression = parent.expression();
        let rewritten = match ListRead::of(expression)? {
            ListRead::Validity => read_validity(child, expression)?,
            ListRead::Lengths => read_lengths(child, expression)?,
            ListRead::Transforms(transforms) => read_transforms(child, expression, &transforms)?,
            ListRead::Elements => None,
        };
        Ok(rewritten.filter(|plan| plan.dtype() == parent.dtype()))
    }
}

/// Evaluates `expression`, which reads at most the validity of `list`, over the validity child.
fn read_validity(
    list: &ListPackPlan,
    expression: &BoundExpression,
) -> VortexResult<Option<PlanRef>> {
    let Some(validity) = list.validity()? else {
        let expression = replace_validity_reads(expression, |is_null| bound::lit(!is_null))?;
        let empty = PackPlan::try_new(
            StructFields::empty(),
            Nullability::NonNullable,
            list.row_count(),
            Vec::new(),
            None,
        )?;
        return Ok(Some(
            EvalPlan::try_new(expression, empty.into_plan())?.into_plan(),
        ));
    };
    let valid = BoundExpression::new_root(validity.dtype().clone());
    let expression = replace_validity_reads(expression, |is_null| {
        if is_null {
            bound::not(valid.clone())
        } else {
            valid.clone()
        }
    })?;
    Ok(Some(EvalPlan::try_new(expression, validity)?.into_plan()))
}

/// `expression` with each `is_null($)` and `is_not_null($)` replaced by `replacement`, which is
/// told whether it replaces `is_null($)`.
fn replace_validity_reads(
    expression: &BoundExpression,
    replacement: impl Fn(bool) -> BoundExpression,
) -> VortexResult<BoundExpression> {
    Ok(expression
        .clone()
        .transform_down(|node| {
            if !is_validity_read(&node) {
                return Ok(Transformed::no(node));
            }
            Ok(Transformed {
                value: replacement(node.is::<IsNull>()),
                changed: true,
                order: TraversalOrder::Skip,
            })
        })?
        .into_inner())
}

/// Evaluates `expression`, which reads at most the lengths and the validity of `list`, over lists
/// of empty structs, which read no element.
fn read_lengths(
    list: &ListPackPlan,
    expression: &BoundExpression,
) -> VortexResult<Option<PlanRef>> {
    let elements = list.elements()?;
    if is_empty_struct(elements.dtype()) {
        return Ok(None);
    }
    let empty = PackPlan::try_new(
        StructFields::empty(),
        Nullability::NonNullable,
        elements.row_count(),
        Vec::new(),
        None,
    )?;
    let lengths = list.with_elements(empty.into_plan())?;
    let expression = rebind_root(expression, lengths.dtype())?;
    Ok(Some(
        EvalPlan::try_new(expression, lengths.into_plan())?.into_plan(),
    ))
}

/// Evaluates the lambda bodies of `transforms` over the elements of `list`, and `expression` over
/// the lists of their results.
fn read_transforms(
    list: &ListPackPlan,
    expression: &BoundExpression,
    transforms: &ListTransforms,
) -> VortexResult<Option<PlanRef>> {
    let elements = list.elements()?;
    if let [body] = transforms.bodies() {
        let projected = list.with_elements(eval_unless_root(body, elements)?)?;
        let lists = BoundExpression::new_root(projected.dtype().clone());
        let expression = transforms.rewrite(expression, |_| Ok(lists.clone()), &lists)?;
        return Ok(Some(
            EvalPlan::try_new(expression, projected.into_plan())?.into_plan(),
        ));
    }
    if is_projection_of_every_field(transforms.bodies(), elements.dtype()) {
        // Packing these bodies would only rename the fields of the elements.
        return Ok(None);
    }
    let names: FieldNames = (0..transforms.bodies().len())
        .map(|index| index.to_string())
        .collect();
    let packed = bound::pack(
        names
            .iter()
            .cloned()
            .zip(transforms.bodies().iter().cloned()),
        Nullability::NonNullable,
    );
    let projected = list.with_elements(EvalPlan::try_new(packed, elements)?.into_plan())?;
    let lists = BoundExpression::new_root(projected.dtype().clone());
    let expression = transforms.rewrite(
        expression,
        |index| {
            let field = Lambda::new(["element"], get_item(names[index].clone(), var("element")))?;
            list_transform(root(), field).bind(projected.dtype())
        },
        &lists,
    )?;
    Ok(Some(
        EvalPlan::try_new(expression, projected.into_plan())?.into_plan(),
    ))
}

impl ListPackPlan {
    /// This list over `elements` in place of its elements child, with the same offsets and
    /// validity.
    fn with_elements(&self, elements: PlanRef) -> VortexResult<Self> {
        Self::try_new(
            self.dtype().nullability(),
            self.row_count(),
            elements,
            self.offsets()?,
            self.validity()?,
        )
    }
}

/// `elements`, or an evaluation of `body` over them unless `body` is the root.
fn eval_unless_root(body: &BoundExpression, elements: PlanRef) -> VortexResult<PlanRef> {
    if body.is_root() {
        return Ok(elements);
    }
    Ok(EvalPlan::try_new(body.clone(), elements)?.into_plan())
}

/// `expression` with every root bound to `dtype` instead.
fn rebind_root(expression: &BoundExpression, dtype: &DType) -> VortexResult<BoundExpression> {
    Ok(expression
        .clone()
        .transform_down(|node| {
            if !node.is_root() {
                return Ok(Transformed::no(node));
            }
            Ok(Transformed::yes(BoundExpression::new_root(dtype.clone())))
        })?
        .into_inner())
}

/// Whether `dtype` is a non-nullable struct with no field.
fn is_empty_struct(dtype: &DType) -> bool {
    !dtype.is_nullable()
        && dtype
            .as_struct_fields_opt()
            .is_some_and(|fields| fields.nfields() == 0)
}

/// Whether `bodies` read every field of non-nullable struct elements of `dtype`, one each and in
/// field order, and nothing else.
fn is_projection_of_every_field(bodies: &[BoundExpression], dtype: &DType) -> bool {
    let Some(fields) = dtype.as_struct_fields_opt() else {
        return false;
    };
    !dtype.is_nullable()
        && bodies.len() == fields.nfields()
        && bodies
            .iter()
            .zip(fields.names().iter())
            .all(|(body, name)| body.as_opt::<GetItem>() == Some(name) && body.child(0).is_root())
}
