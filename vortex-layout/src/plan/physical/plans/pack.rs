// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::iter;
use std::ops::Range;

use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr as unbound;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::is_not_null::IsNotNull;
use vortex_array::scalar_fn::fns::is_null::IsNull;
use vortex_array::scalar_fn::fns::pack::Pack as PackFn;
use vortex_array::scalar_fn::fns::select::FieldSelection;
use vortex_array::scalar_fn::fns::select::Select;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::partition::PartitionRule;
use crate::partition::PartitionSession;
use crate::partition::PartitionedExpression;
use crate::partition::PartitionedPlan;
use crate::partition::Partitions;
use crate::partition::PlanSlot;
use crate::partition::Slots;
use crate::partition::positional_name;
use crate::plan::physical::Eval;
use crate::plan::physical::EvalPlan;
use crate::plan::physical::Plan;
use crate::plan::physical::PlanId;
use crate::plan::physical::PlanParts;
use crate::plan::physical::PlanRef;
use crate::plan::physical::PlanSlots;
use crate::plan::physical::PlanVTable;
use crate::plan::physical::operators::OnceSource;
use crate::plan::physical::operators::PackSource;
use crate::plan::physical::operators::WrapStage;
use crate::plan::physical::operators::empty_struct;
use crate::plan::physical::optimizer::PlanParentReduceRule;
use crate::plan::physical::pipeline::Chain;
use crate::plan::physical::pipeline::Compiler;

/// Assembles a struct from a validity slot followed by one slot per field.
///
/// The slots match those of a struct array and a struct layout: slot
/// [`VALIDITY_SLOT`](Self::VALIDITY_SLOT) holds the validity, present exactly when the struct is
/// nullable, and field `i` is in slot `FIELDS_OFFSET + i`.
#[derive(Clone, Debug)]
pub struct Pack;

impl Pack {
    /// The slot holding struct validity, absent when the struct is non-nullable.
    pub const VALIDITY_SLOT: usize = 0;
    /// The slot of field zero; field `i` is in slot `FIELDS_OFFSET + i`.
    pub const FIELDS_OFFSET: usize = 1;
}

/// Operator-specific struct assembly data.
#[derive(Clone, Debug)]
pub struct PackData;

/// A plan that assembles a struct from its children.
pub type PackPlan = Plan<Pack>;

impl PackPlan {
    /// Creates a struct assembly from potentially unbuilt slots without validation.
    ///
    /// # Safety
    ///
    /// `slots` must hold a validity slot followed by one slot per field. The validity slot must
    /// be present exactly when `nullability` is nullable and produce non-nullable booleans. Every
    /// field slot must be present with its field dtype, and every child must have `row_count`
    /// rows.
    pub(crate) unsafe fn from_slots_unchecked(
        fields: StructFields,
        nullability: Nullability,
        row_count: u64,
        slots: PlanSlots,
    ) -> Self {
        PlanParts {
            vtable: Pack,
            dtype: DType::Struct(fields, nullability),
            row_count,
            slots,
            data: PackData,
        }
        .into_typed()
    }

    /// Creates a struct assembly from `fields` and one child per field.
    ///
    /// Each field child must have the corresponding field dtype and `row_count` rows. `validity`
    /// is required exactly when `nullability` is [`Nullability::Nullable`] and must produce
    /// non-nullable booleans with `row_count` rows.
    pub fn try_new(
        fields: StructFields,
        nullability: Nullability,
        row_count: u64,
        field_plans: Vec<PlanRef>,
        validity: Option<PlanRef>,
    ) -> VortexResult<Self> {
        if field_plans.len() != fields.nfields() {
            vortex_bail!(
                "Pack expects {} field children but got {}",
                fields.nfields(),
                field_plans.len()
            );
        }
        if validity.is_some() != (nullability == Nullability::Nullable) {
            vortex_bail!("Pack validity child must be present exactly when the struct is nullable");
        }

        for (index, (field_dtype, field_plan)) in
            fields.fields().zip(field_plans.iter()).enumerate()
        {
            validate_field_child(index, &field_dtype, row_count, field_plan)?;
        }
        if let Some(validity) = validity.as_ref() {
            validate_validity_child(row_count, validity)?;
        }

        // SAFETY: The child count, presence, dtypes, and row counts were validated above.
        Ok(unsafe { Self::new_unchecked(fields, nullability, row_count, field_plans, validity) })
    }

    /// Creates a struct assembly without validating its children.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    ///
    /// - `field_plans` contains exactly one child per field, in field order;
    /// - every field child has the corresponding field dtype and `row_count` rows; and
    /// - `validity` is present exactly when the struct is nullable and, when present, produces
    ///   non-nullable booleans with `row_count` rows.
    pub unsafe fn new_unchecked(
        fields: StructFields,
        nullability: Nullability,
        row_count: u64,
        field_plans: Vec<PlanRef>,
        validity: Option<PlanRef>,
    ) -> Self {
        let slots = iter::once(validity)
            .chain(field_plans.into_iter().map(Some))
            .collect();
        // SAFETY: The caller guarantees the same slot invariants required by this constructor.
        unsafe { Self::from_slots_unchecked(fields, nullability, row_count, slots) }
    }

    /// Returns the struct fields assembled by this plan.
    pub fn fields(&self) -> &StructFields {
        self.dtype()
            .as_struct_fields_opt()
            .vortex_expect("Pack dtype must be a struct")
    }

    /// Returns the number of struct fields, excluding any validity child.
    pub fn nfields(&self) -> usize {
        self.fields().nfields()
    }

    /// Returns the plan producing struct validity, if the struct is nullable.
    pub fn validity(&self) -> VortexResult<Option<PlanRef>> {
        self.slot(Pack::VALIDITY_SLOT)
    }

    /// Returns the plan producing field `index`.
    pub fn field(&self, index: usize) -> VortexResult<PlanRef> {
        self.slot_required(Pack::FIELDS_OFFSET + index)
    }
}

impl PlanVTable for Pack {
    type PlanData = PackData;
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.pack");
        *ID
    }

    fn metadata(_plan: &Plan<Self>) -> Option<Self::Metadata> {
        // The struct fields are recoverable from the plan dtype.
        Some(EmptyMetadata)
    }

    fn with_slots(
        plan: &Plan<Self>,
        slots: &PlanSlots,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        slots.check_len("Pack", Pack::FIELDS_OFFSET + plan.nfields())?;
        match (plan.dtype().is_nullable(), slots.get(Pack::VALIDITY_SLOT)?) {
            (true, Some(validity)) => validate_validity_child(plan.row_count(), validity)?,
            (false, None) => {}
            (true, None) => vortex_bail!("Nullable Pack has no validity child"),
            (false, Some(_)) => vortex_bail!("Non-nullable Pack has a validity child"),
        }
        for (index, field_dtype) in plan.fields().fields().enumerate() {
            let child = slots.required(Pack::FIELDS_OFFSET + index)?;
            validate_field_child(index, &field_dtype, plan.row_count(), child)?;
        }
        Ok(())
    }

    fn slot_name(plan: &Plan<Self>, slot: usize) -> Cow<'_, str> {
        if slot == Pack::VALIDITY_SLOT {
            return Cow::Borrowed("validity");
        }
        let name = plan
            .fields()
            .field_name(slot - Pack::FIELDS_OFFSET)
            .vortex_expect("Pack slot index out of bounds");
        Cow::Borrowed(name.as_ref())
    }

    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        // Inlet `i` reads the `i`-th present slot: validity first when nullable, then the fields.
        let count = plan.slots().present().count();
        if count == 0 {
            // A struct with no fields still has rows, so no child can carry them.
            let array = empty_struct(
                plan.fields().clone(),
                plan.dtype().nullability(),
                mask.true_count(),
            )?;
            return Ok(Some(Chain::new(OnceSource::new(array))));
        }
        if mask.all_false() {
            return Ok(None);
        }
        let mut chains = Vec::with_capacity(count);
        for child in plan.children() {
            chains.push(
                compiler
                    .compile(child?, rows.clone(), mask)?
                    .ok_or_else(|| vortex_err!("A Pack child produced no rows"))?,
            );
        }
        let nullable = plan.dtype().is_nullable();
        if count == 1 && !nullable {
            // One field needs no zip: each of its batches is wrapped as it passes.
            let chain = chains.remove(0);
            return Ok(Some(chain.with(WrapStage::new(plan.fields().clone()))));
        }
        let source = PackSource::new(plan.fields().clone(), nullable, count);
        Ok(Some(compiler.join(chains, source)))
    }
}

fn validate_field_child(
    index: usize,
    expected_dtype: &DType,
    expected_row_count: u64,
    child: &PlanRef,
) -> VortexResult<()> {
    if child.dtype() != expected_dtype {
        vortex_bail!(
            "Pack field child {index} has dtype {} but the field has dtype {expected_dtype}",
            child.dtype()
        );
    }
    if child.row_count() != expected_row_count {
        vortex_bail!(
            "Pack field child {index} has {} rows but the plan has {expected_row_count}",
            child.row_count()
        );
    }
    Ok(())
}

fn validate_validity_child(expected_row_count: u64, child: &PlanRef) -> VortexResult<()> {
    let expected_dtype = DType::Bool(Nullability::NonNullable);
    if child.dtype() != &expected_dtype {
        vortex_bail!(
            "Pack validity child has dtype {} but must have dtype {expected_dtype}",
            child.dtype()
        );
    }
    if child.row_count() != expected_row_count {
        vortex_bail!(
            "Pack validity child has {} rows but the plan has {expected_row_count}",
            child.row_count()
        );
    }
    Ok(())
}

impl PackPlan {
    /// Rebuilds this plan with only `fields`, which must be a subset of the current fields.
    ///
    /// Pruning is only sound for a non-nullable struct: dropping a field of a nullable struct
    /// would drop the validity child that the remaining fields depend on.
    pub(crate) fn with_pruned_fields(
        &self,
        fields: Vec<(FieldName, PlanRef)>,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            !self.dtype().is_nullable(),
            "Cannot prune fields from a nullable Pack"
        );
        let struct_fields = StructFields::from_iter(
            fields
                .iter()
                .map(|(name, plan)| (name.clone(), plan.dtype().clone())),
        );
        let field_plans = fields.into_iter().map(|(_, plan)| plan).collect::<Vec<_>>();
        PackPlan::try_new(
            struct_fields,
            Nullability::NonNullable,
            self.row_count(),
            field_plans,
            None,
        )
    }
}

/// Pushes an expression into the referenced fields of a [`Pack`], pruning the rest.
#[derive(Debug)]
pub(crate) struct ExpressionPackRule;

impl PlanParentReduceRule<Pack> for ExpressionPackRule {
    type Parent = Eval;

    fn reduce_parent(
        &self,
        child: &Plan<Pack>,
        parent: &Plan<Eval>,
        _child_idx: usize,
        session: &VortexSession,
    ) -> VortexResult<Option<PlanRef>> {
        if child.dtype().is_nullable() {
            return Ok(None);
        }

        let expression = parent.expression();
        let fields = child.fields();
        if fields.nfields() == 0 {
            return Ok(None);
        }
        let partitioned = match Pack.partition(child.dtype(), expression, session)? {
            PartitionedExpression::One(slot, expr) => {
                let field = child.slot_required(slot)?;
                return Ok(Some(EvalPlan::try_new(expr, field)?.into_plan()));
            }
            PartitionedExpression::Any(expr) => {
                let rewritten = child.with_pruned_fields(Vec::new())?.into_plan();
                return Ok(Some(EvalPlan::try_new(expr, rewritten)?.into_plan()));
            }
            PartitionedExpression::Many(partitioned) => partitioned,
        };
        let Partitions {
            combiner,
            partitions,
        } = partitioned;

        let mut fields_changed = partitions.len() != fields.nfields();
        let mut pruned_fields = Vec::with_capacity(partitions.len());
        for (position, partition) in partitions.into_iter().enumerate() {
            fields_changed |= partition.slot != Pack::FIELDS_OFFSET + position;
            fields_changed |= fields
                .field_name(position)
                .is_none_or(|name| *name != positional_name(position));

            let field = child.slot_required(partition.slot)?;
            let field = if is_identity_expression(&partition.expr, field.dtype()) {
                field
            } else {
                fields_changed = true;
                EvalPlan::try_new(partition.expr, field)?.into_plan()
            };
            pruned_fields.push((positional_name(position), field));
        }
        let rewritten = if fields_changed {
            child.with_pruned_fields(pruned_fields)?.into_plan()
        } else {
            child.to_plan()
        };

        if !fields_changed && combiner == *expression {
            return Ok(None);
        }

        Ok(Some(EvalPlan::try_new(combiner, rewritten)?.into_plan()))
    }
}

fn is_identity_expression(expression: &BoundExpression, input_dtype: &DType) -> bool {
    if expression.is_root() {
        return expression.dtype() == input_dtype;
    }
    if input_dtype.is_nullable() {
        return false;
    }
    let Some(fields) = input_dtype.as_struct_fields_opt() else {
        return false;
    };
    let Some(options) = expression.as_opt::<PackFn>() else {
        return false;
    };
    !options.nullability.is_nullable()
        && options.names == *fields.names()
        && expression
            .children()
            .iter()
            .zip(fields.names().iter())
            .all(|(child, name)| {
                child.as_opt::<GetItem>().is_some_and(|item| item == name)
                    && child.children()[0].is_root()
                    && child.children()[0].dtype() == input_dtype
            })
}

#[derive(Clone, Copy)]
pub enum PackSlot {
    Field(PlanSlot),
    Validity,
}
impl From<PackSlot> for PlanSlot {
    fn from(slot: PackSlot) -> Self {
        match slot {
            PackSlot::Validity => Pack::VALIDITY_SLOT,
            PackSlot::Field(index) => Pack::FIELDS_OFFSET + index,
        }
    }
}

impl PartitionedPlan for Pack {
    type Slot = PackSlot;
    fn slot_dtype(scope: &DType, slot: PackSlot) -> VortexResult<DType> {
        match slot {
            PackSlot::Validity => Ok(validity_dtype()),
            PackSlot::Field(index) => field_dtype(struct_fields(scope)?, index),
        }
    }
}

pub fn register_pack_partition_rules(session: &PartitionSession) {
    session.register_root_rule(Pack, &RootRule);
    session.register_rule(Pack, GetItem, &GetItemRule);
    session.register_rule(Pack, Select, &SelectRule);
    session.register_rule(Pack, IsNotNull, &IsNotNullRule);
    session.register_rule(Pack, IsNull, &IsNullRule);
}

fn struct_fields(scope: &DType) -> VortexResult<&StructFields> {
    scope
        .as_struct_fields_opt()
        .ok_or_else(|| vortex_err!("Not a struct scope"))
}

fn field_dtype(fields: &StructFields, index: PlanSlot) -> VortexResult<DType> {
    fields
        .field_by_index(index)
        .ok_or_else(|| vortex_err!("No field {index} in struct scope"))
}

fn validity_dtype() -> DType {
    DType::Bool(Nullability::NonNullable)
}

fn select_fields(
    slots: &Slots<'_, Pack>,
    fields: &StructFields,
    names: FieldNames,
    indices: Vec<PlanSlot>,
) -> VortexResult<Expression> {
    vortex_ensure!(fields.nfields() > 0);
    let elements = names
        .iter()
        .cloned()
        .zip(indices)
        .map(|(name, index)| (name, slots.read(PackSlot::Field(index))));
    let packed = unbound::pack(elements, Nullability::NonNullable);
    if !slots.scope().is_nullable() {
        return Ok(packed);
    }
    Ok(unbound::mask(packed, slots.read(PackSlot::Validity)))
}

/// $ -> pack(...) (non-nullable) or mask(pack(...), validity)
struct RootRule;
impl PartitionRule for RootRule {
    type Plan = Pack;
    fn apply(
        &self,
        slots: &Slots<'_, Pack>,
        _node: &BoundExpression,
    ) -> VortexResult<Option<Expression>> {
        let fields = struct_fields(slots.scope())?;
        let indices = (0..fields.nfields()).collect();
        select_fields(slots, fields, fields.names().clone(), indices).map(Some)
    }
}

/// select(..., $) -> pack(... + optional validity)
struct SelectRule;
impl PartitionRule for SelectRule {
    type Plan = Pack;
    fn apply(
        &self,
        slots: &Slots<'_, Pack>,
        node: &BoundExpression,
    ) -> VortexResult<Option<Expression>> {
        let selection = node.as_::<Select>();
        let fields = struct_fields(slots.scope())?;
        let names = match selection {
            FieldSelection::Include(names) => names.clone(),
            FieldSelection::Exclude(_) => selection.normalize_to_included_fields(fields.names())?,
        };
        let indices = names
            .iter()
            .map(|name| {
                fields
                    .find(name)
                    .ok_or_else(|| vortex_err!("field {name} not found"))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        select_fields(slots, fields, names, indices).map(Some)
    }
}

/// $.x -> $.x (non-nullable) or mask($.x, $.validity)
struct GetItemRule;
impl PartitionRule for GetItemRule {
    type Plan = Pack;
    fn apply(
        &self,
        slots: &Slots<'_, Pack>,
        node: &BoundExpression,
    ) -> VortexResult<Option<Expression>> {
        let field_name = node.as_::<GetItem>();
        let index = struct_fields(slots.scope())?
            .find(field_name)
            .ok_or_else(|| vortex_err!("Field {field_name} not found"))?;
        let read = slots.read(PackSlot::Field(index));
        if !slots.scope().is_nullable() {
            return Ok(Some(read));
        }
        Ok(Some(unbound::mask(read, slots.read(PackSlot::Validity))))
    }
}

/// is_not_null -> lit(true) (non-nullable) or $.validity
struct IsNotNullRule;
impl PartitionRule for IsNotNullRule {
    type Plan = Pack;
    fn apply(
        &self,
        slots: &Slots<'_, Pack>,
        _node: &BoundExpression,
    ) -> VortexResult<Option<Expression>> {
        let expr = if slots.scope().is_nullable() {
            slots.read(PackSlot::Validity)
        } else {
            unbound::lit(true)
        };
        Ok(Some(expr))
    }
}

/// is_null-> lit(false) (non-nullable) or !$.validity
struct IsNullRule;
impl PartitionRule for IsNullRule {
    type Plan = Pack;
    fn apply(
        &self,
        slots: &Slots<'_, Pack>,
        _node: &BoundExpression,
    ) -> VortexResult<Option<Expression>> {
        let expr = if slots.scope().is_nullable() {
            unbound::not(slots.read(PackSlot::Validity))
        } else {
            unbound::lit(false)
        };
        Ok(Some(expr))
    }
}

#[cfg(test)]
mod tests {
    use insta::assert_snapshot;
    use vortex_array::dtype::PType;
    use vortex_array::expr as unbound;
    use vortex_array::expr::bound::get_item;
    use vortex_array::expr::bound::is_not_null;
    use vortex_array::expr::bound::pack;

    use super::*;
    use crate::partition::PartitionedExpression;
    use crate::partition::Partitions;
    use crate::test::SESSION;

    fn scope_of(expr: &BoundExpression) -> Option<&DType> {
        if expr.is_root() {
            return Some(expr.dtype());
        }
        expr.children().iter().find_map(scope_of)
    }

    fn i32() -> DType {
        DType::Primitive(PType::I32, Nullability::NonNullable)
    }

    fn struct_of(fields: &[(&str, DType)], nullability: Nullability) -> DType {
        DType::Struct(
            StructFields::from_iter(fields.iter().map(|(name, dtype)| (*name, dtype.clone()))),
            nullability,
        )
    }

    fn two_field_struct() -> DType {
        struct_of(&[("a", i32()), ("b", i32())], Nullability::NonNullable)
    }

    fn fmt_slot(slot: PlanSlot) -> String {
        match slot {
            Pack::VALIDITY_SLOT => "validity".to_string(),
            slot => (slot - Pack::FIELDS_OFFSET).to_string(),
        }
    }

    fn line(label: &str, expr: &BoundExpression) -> String {
        let scope = scope_of(expr)
            .map(|dtype| dtype.to_string())
            .unwrap_or_else(|| "-".to_string());
        format!("{label} scope={scope}: {expr}\n")
    }

    fn snap(partitioning: PartitionedExpression) -> String {
        match partitioning {
            PartitionedExpression::Any(expr) => line("mono none", &expr),
            PartitionedExpression::One(slot, expr) => {
                line(&format!("mono {}", fmt_slot(slot)), &expr)
            }
            PartitionedExpression::Many(Partitions {
                combiner: root,
                partitions,
            }) => {
                let mut out = String::new();
                for partition in &partitions {
                    out.push_str(&line(
                        &format!("partition {}", fmt_slot(partition.slot)),
                        &partition.expr,
                    ));
                }
                out.push_str(&line("root", &root));
                out
            }
        }
    }

    fn partition(expr: Expression, scope: &DType) -> VortexResult<String> {
        Ok(snap(Pack.partition(scope, &expr.bind(scope)?, &SESSION)?))
    }

    #[test]
    fn is_not_null_of_non_nullable() -> VortexResult<()> {
        let root = BoundExpression::new_root(two_field_struct());
        let expr = pack(
            [
                ("v", is_not_null(root.clone())),
                ("a", get_item("a", root.clone())),
                ("b", get_item("b", root)),
            ],
            Nullability::NonNullable,
        );
        assert_snapshot!(snap(Pack.partition(&two_field_struct(), &expr, &SESSION)?), @"
        partition 0 scope=i32: $
        partition 1 scope=i32: $
        root scope={0=i32, 1=i32}: pack(v: true, a: $.0, b: $.1)
        ");
        Ok(())
    }

    #[test]
    fn validity_select_and_root() -> VortexResult<()> {
        let scope = struct_of(&[("a", i32()), ("b", i32())], Nullability::Nullable);
        let expr = unbound::select(["a"], unbound::root());
        assert_snapshot!(partition(expr, &scope)?, @"
        partition 0 scope=i32: $
        partition validity scope=bool: $
        root scope={0=i32, 1=bool}: vortex.mask(pack(a: $.0), $.1)
        ");
        assert_snapshot!(partition(unbound::root(), &scope)?, @"
        partition 0 scope=i32: $
        partition 1 scope=i32: $
        partition validity scope=bool: $
        root scope={0=i32, 1=i32, 2=bool}: vortex.mask(pack(a: $.0, b: $.1), $.2)
        ");
        Ok(())
    }

    #[test]
    fn validity_and_field() -> VortexResult<()> {
        let inner = struct_of(&[("c", i32()), ("d", i32())], Nullability::NonNullable);
        let scope = struct_of(&[("a", i32()), ("s", inner)], Nullability::Nullable);
        let expr = unbound::and(
            unbound::is_null(unbound::root()),
            unbound::gt(
                unbound::get_item("d", unbound::get_item("s", unbound::root())),
                unbound::lit(0i32),
            ),
        );
        assert_snapshot!(partition(expr, &scope)?, @"
        partition 1 scope={c=i32, d=i32}: $
        partition validity scope=bool: pack(0: $, 1: vortex.not($))
        root scope={0={c=i32, d=i32}, 1={0=bool, 1=bool}}: ($.1.1 and (vortex.mask($.0, $.1.0).d > 0i32))
        ");
        Ok(())
    }

    #[test]
    fn validity_only() -> VortexResult<()> {
        let scope = struct_of(&[("a", i32()), ("b", i32())], Nullability::Nullable);
        let expr = unbound::is_not_null(unbound::root());
        assert_snapshot!(partition(expr, &scope)?, @"mono validity scope=bool: $");
        let expr = unbound::is_null(unbound::root());
        assert_snapshot!(partition(expr, &scope)?, @"mono validity scope=bool: vortex.not($)");

        let a = struct_of(&[("x", i32())], Nullability::Nullable);
        let b = struct_of(&[("y", i32())], Nullability::Nullable);
        let scope = struct_of(&[("a", a), ("b", b)], Nullability::NonNullable);
        let expr = unbound::and(
            unbound::is_not_null(unbound::get_item("a", unbound::root())),
            unbound::is_not_null(unbound::get_item("b", unbound::root())),
        );
        assert_snapshot!(partition(expr, &scope)?, @"
        partition 0 scope={x=i32}?: is_not_null($)
        partition 1 scope={y=i32}?: is_not_null($)
        root scope={0=bool, 1=bool}: ($.0 and $.1)
        ");
        Ok(())
    }

    #[test]
    fn equal_cuts() -> VortexResult<()> {
        let root = BoundExpression::new_root(two_field_struct());
        let a = get_item("a", root.clone());
        let b = get_item("b", root);
        let expr = pack(
            [("x", a.clone()), ("y", b), ("z", a)],
            Nullability::NonNullable,
        );
        assert_snapshot!(snap(Pack.partition(&two_field_struct(), &expr, &SESSION)?), @"
        partition 0 scope=i32: $
        partition 1 scope=i32: $
        root scope={0=i32, 1=i32}: pack(x: $.0, y: $.1, z: $.0)
        ");
        Ok(())
    }

    #[test]
    fn is_not_null_one_partition() -> VortexResult<()> {
        let expr = unbound::is_not_null(unbound::get_item("a", unbound::root()));
        assert_snapshot!(partition(expr, &two_field_struct())?, @"mono 0 scope=i32: is_not_null($)");
        Ok(())
    }

    #[test]
    fn is_not_null_two_fields() -> VortexResult<()> {
        let scope = struct_of(&[("a", i32()), ("b", i32())], Nullability::Nullable);
        let expr = unbound::is_not_null(unbound::get_item("a", unbound::root()));
        assert_snapshot!(partition(expr, &scope)?, @"
        partition 0 scope=i32: $
        partition validity scope=bool: $
        root scope={0=i32, 1=bool}: $.1
        ");
        Ok(())
    }

    #[test]
    fn test_pack() -> VortexResult<()> {
        let expr = unbound::get_item(
            "a",
            unbound::get_item(
                "a",
                unbound::pack([("a", unbound::root())], Nullability::NonNullable),
            ),
        );
        assert_snapshot!(partition(expr, &two_field_struct())?, @"
        partition 0 scope=i32: $
        partition 1 scope=i32: $
        root scope={0=i32, 1=i32}: $.0
        ");
        Ok(())
    }

    #[test]
    fn test_ab_greater_c() -> VortexResult<()> {
        let inner = struct_of(&[("b", i32())], Nullability::NonNullable);
        let scope = struct_of(&[("a", inner), ("c", i32())], Nullability::NonNullable);
        let expr = unbound::gt(
            unbound::get_item("b", unbound::get_item("a", unbound::root())),
            unbound::get_item("c", unbound::root()),
        );
        assert_snapshot!(partition(expr, &scope)?, @"
        partition 0 scope={b=i32}: $.b
        partition 1 scope=i32: $
        root scope={0=i32, 1=i32}: ($.0 > $.1)
        ");
        Ok(())
    }

    #[test]
    fn test_x_plus_1() -> VortexResult<()> {
        let scope = struct_of(&[("x", i32()), ("y", i32())], Nullability::NonNullable);
        let expr = unbound::gt(
            unbound::checked_add(unbound::get_item("x", unbound::root()), unbound::lit(1i32)),
            unbound::get_item("y", unbound::root()),
        );
        assert_snapshot!(partition(expr, &scope)?, @"
        partition 0 scope=i32: ($ + 1i32)
        partition 1 scope=i32: $
        root scope={0=i32, 1=i32}: ($.0 > $.1)
        ");
        Ok(())
    }

    #[test]
    fn test_nested_comparison() -> VortexResult<()> {
        let inner = struct_of(&[("a", i32())], Nullability::NonNullable);
        let scope = struct_of(
            &[("a", inner.clone()), ("b", inner)],
            Nullability::NonNullable,
        );
        let expr = unbound::gt(
            unbound::get_item("a", unbound::get_item("a", unbound::root())),
            unbound::get_item("a", unbound::get_item("b", unbound::root())),
        );
        assert_snapshot!(partition(expr, &scope)?, @"
        partition 0 scope={a=i32}: $.a
        partition 1 scope={a=i32}: $.a
        root scope={0=i32, 1=i32}: ($.0 > $.1)
        ");
        Ok(())
    }

    #[test]
    fn test_nested_and() -> VortexResult<()> {
        let a = unbound::get_item("a", unbound::root());
        let b = unbound::get_item("b", unbound::root());
        let expr = unbound::and(
            unbound::and(
                unbound::gt(a.clone(), unbound::lit(5i32)),
                unbound::gt(b, unbound::lit(5i32)),
            ),
            unbound::lt(a, unbound::lit(1i32)),
        );
        assert_snapshot!(partition(expr, &two_field_struct())?, @"
        partition 0 scope=i32: pack(0: ($ > 5i32), 1: ($ < 1i32))
        partition 1 scope=i32: ($ > 5i32)
        root scope={0={0=bool, 1=bool}, 1=bool}: (($.0.0 and $.1) and $.0.1)
        ");
        Ok(())
    }

    #[test]
    fn test_duplicate_predicate() -> VortexResult<()> {
        let a = unbound::get_item("a", unbound::root());
        let b = unbound::get_item("b", unbound::root());
        let gt5_a = unbound::gt(a, unbound::lit(5i32));
        let expr = unbound::and(
            unbound::and(gt5_a.clone(), unbound::gt(b, unbound::lit(5i32))),
            gt5_a,
        );
        assert_snapshot!(partition(expr, &two_field_struct())?, @"
        partition 0 scope=i32: ($ > 5i32)
        partition 1 scope=i32: ($ > 5i32)
        root scope={0=bool, 1=bool}: (($.0 and $.1) and $.0)
        ");
        Ok(())
    }

    #[test]
    fn test_deeply_nested_comparison() -> VortexResult<()> {
        let leaf = struct_of(&[("c", i32()), ("d", i32())], Nullability::NonNullable);
        let middle = struct_of(&[("b", leaf)], Nullability::NonNullable);
        let scope = struct_of(&[("a", middle)], Nullability::NonNullable);
        let expr = unbound::gt(
            unbound::get_item(
                "c",
                unbound::get_item("b", unbound::get_item("a", unbound::root())),
            ),
            unbound::get_item(
                "d",
                unbound::get_item("b", unbound::get_item("a", unbound::root())),
            ),
        );
        assert_snapshot!(partition(expr, &scope)?, @"mono 0 scope={b={c=i32, d=i32}}: ($.b.c > $.b.d)");
        Ok(())
    }
}
