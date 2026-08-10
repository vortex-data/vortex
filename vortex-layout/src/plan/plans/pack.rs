// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;

use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::ExactBoundExpr;
use vortex_array::expr::descendent_bound_annotations;
use vortex_array::expr::make_bound_free_field_annotator;
use vortex_array::expr::transform::partition_bound;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::expr::traversal::TraversalOrder;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::pack::Pack as PackFn;
use vortex_array::scalar_fn::fns::pack::PackOptions;
use vortex_array::scalar_fn::fns::select::Select;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;

use crate::plan::Eval;
use crate::plan::EvalPlan;
use crate::plan::Plan;
use crate::plan::PlanChildren;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanVTable;
use crate::plan::optimizer::PlanParentReduceRule;
use crate::plan::plans::eval::rewrite_partition_root;

/// Assembles a struct from one child per field, plus an optional trailing validity child.
#[derive(Clone, Debug)]
pub struct Pack;

/// Operator-specific struct assembly data.
#[derive(Clone, Debug)]
pub struct PackData;

/// A plan that assembles a struct from its children.
pub type PackPlan = Plan<Pack>;

impl PackPlan {
    /// Creates a struct assembly from potentially unresolved children without validation.
    ///
    /// # Safety
    ///
    /// `children` must contain one child per field, followed by a non-nullable boolean validity
    /// child exactly when `nullability` is nullable. Every child must have `row_count` rows, and
    /// every field child must have its corresponding field dtype.
    pub(crate) unsafe fn from_children_unchecked(
        fields: StructFields,
        nullability: Nullability,
        row_count: u64,
        children: PlanChildren,
    ) -> Self {
        PlanParts {
            vtable: Pack,
            dtype: DType::Struct(fields, nullability),
            row_count,
            children,
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
        let mut children = field_plans;
        children.extend(validity);
        // SAFETY: The caller guarantees the same child invariants required by this constructor.
        unsafe { Self::from_children_unchecked(fields, nullability, row_count, children.into()) }
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
        self.child(self.nfields())
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

    fn with_children(
        plan: &Plan<Self>,
        children: &PlanChildren,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        let expected_children = plan.nfields() + usize::from(plan.dtype().is_nullable());
        if children.len() != expected_children {
            vortex_bail!(
                "Pack expects {expected_children} children but got {}",
                children.len()
            );
        }

        for (index, field_dtype) in plan.fields().fields().enumerate() {
            let child = children
                .get(index)?
                .ok_or_else(|| vortex_err!("Pack field child {index} is absent"))?;
            validate_field_child(index, &field_dtype, plan.row_count(), &child)?;
        }
        if plan.dtype().is_nullable() {
            let validity = children
                .get(plan.nfields())?
                .ok_or_else(|| vortex_err!("Pack validity child is absent"))?;
            validate_validity_child(plan.row_count(), &validity)?;
        }
        Ok(())
    }

    fn child_name(plan: &Plan<Self>, index: usize) -> Cow<'_, str> {
        assert!(
            index < plan.children().len(),
            "Pack child index out of bounds: {index} of {}",
            plan.children().len()
        );
        if let Some(name) = plan.fields().field_name(index) {
            return Cow::Borrowed(name.as_ref());
        }
        Cow::Borrowed("validity")
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
    ) -> VortexResult<Option<PlanRef>> {
        if child.dtype().is_nullable() {
            return Ok(None);
        }

        let expression = parent.expression();
        let fields = child.fields();
        let referenced_fields =
            descendent_bound_annotations(expression, make_bound_free_field_annotator(fields))
                .get(&ExactBoundExpr(expression.clone()))
                .vortex_expect("Bound expression missing free-field annotations")
                .clone();
        let expanded_root = expanded_struct_root(child.dtype(), fields)?;
        let expanded = expand_struct_root(expression.clone(), &expanded_root, fields)?;
        let partitioned =
            partition_bound(expanded.clone(), make_bound_free_field_annotator(fields))?;

        if partitioned.partition_names.is_empty() {
            let selected_indices = fields
                .names()
                .iter()
                .enumerate()
                .filter_map(|(index, name)| referenced_fields.contains(name).then_some(index))
                .collect::<Vec<_>>();
            if selected_indices.len() == fields.nfields() {
                return Ok(None);
            }

            let pruned_fields = selected_indices
                .into_iter()
                .map(|field_index| {
                    Ok((
                        field_name(fields, field_index)?,
                        field_plan(child, field_index)?,
                    ))
                })
                .collect::<VortexResult<Vec<_>>>()?;
            let rewritten = child.with_pruned_fields(pruned_fields)?.into_plan();
            return Ok(Some(
                EvalPlan::try_new(expression.clone(), rewritten)?.into_plan(),
            ));
        }

        if partitioned.partition_names.len() == 1 {
            let name = partitioned
                .partition_names
                .get(0)
                .ok_or_else(|| vortex_err!("Struct expression partition has no field"))?;
            let index = fields.find(name).ok_or_else(|| {
                vortex_err!("Struct expression references unknown field '{name}'")
            })?;
            let field = field_plan(child, index)?;
            let lowered = step_into_struct_field(expanded, name, field.dtype().clone())?;
            return Ok(Some(EvalPlan::try_new(lowered, field)?.into_plan()));
        }

        let residual = partitioned.root;
        let mut collapsed = Vec::with_capacity(partitioned.partitions.len());
        let mut field_expressions = vec![None; fields.nfields()];
        for index in 0..partitioned.partitions.len() {
            let name = &partitioned.partition_names[index];
            let partition = &partitioned.partitions[index];
            let field_index = fields.find(name).ok_or_else(|| {
                vortex_err!("Struct expression references unknown field '{name}'")
            })?;
            let field = field_plan(child, field_index)?;
            let lowered = if let Some(pack) = partition
                .as_scalar()
                .and_then(|scalar_fn| scalar_fn.as_opt::<PackFn>())
                && partition.children().len() == 1
            {
                let value_name = pack
                    .names
                    .get(0)
                    .ok_or_else(|| vortex_err!("Struct expression partition pack is empty"))?;
                collapsed.push((name.clone(), value_name.clone()));
                partition.children()[0].clone()
            } else {
                partition.clone()
            };
            let lowered = step_into_struct_field(lowered, name, field.dtype().clone())?;
            field_expressions[field_index] = Some(lowered);
        }

        let mut pruned_fields = Vec::with_capacity(partitioned.partition_names.len());
        for (field_index, expression) in field_expressions.into_iter().enumerate() {
            let Some(expression) = expression else {
                continue;
            };
            let field = field_plan(child, field_index)?;
            pruned_fields.push((
                field_name(fields, field_index)?,
                EvalPlan::try_new(expression, field)?.into_plan(),
            ));
        }
        let rewritten = child.with_pruned_fields(pruned_fields)?.into_plan();
        let residual = rewrite_partition_root(residual, rewritten.dtype().clone(), &collapsed)?;

        Ok(Some(EvalPlan::try_new(residual, rewritten)?.into_plan()))
    }
}

fn field_name(fields: &StructFields, index: usize) -> VortexResult<FieldName> {
    Ok(fields
        .field_name(index)
        .ok_or_else(|| vortex_err!("Struct field {index} has no name"))?
        .clone())
}

fn field_plan(plan: &Plan<Pack>, index: usize) -> VortexResult<PlanRef> {
    plan.child(index)?
        .ok_or_else(|| vortex_err!("Struct field {index} has no plan"))
}

fn expanded_struct_root(
    root_dtype: &DType,
    fields: &StructFields,
) -> VortexResult<BoundExpression> {
    let root = BoundExpression::new_root(root_dtype.clone());
    let children = fields
        .names()
        .iter()
        .map(|name| BoundExpression::try_new(GetItem.bind(name.clone()), [root.clone()]))
        .collect::<VortexResult<Vec<_>>>()?;
    bound_pack(fields.names().clone(), children)
}

fn expand_struct_root(
    expression: BoundExpression,
    expanded_root: &BoundExpression,
    fields: &StructFields,
) -> VortexResult<BoundExpression> {
    Ok(expression
        .transform_down(|node| {
            if node.is_root() {
                return Ok(Transformed {
                    value: expanded_root.clone(),
                    changed: true,
                    order: TraversalOrder::Skip,
                });
            }

            let Some(scalar_fn) = node.as_scalar() else {
                return Ok(Transformed::no(node));
            };
            if !node
                .children()
                .first()
                .is_some_and(BoundExpression::is_root)
            {
                return Ok(Transformed::no(node));
            }

            if let Some(field_name) = scalar_fn.as_opt::<GetItem>() {
                let index = fields.find(field_name).ok_or_else(|| {
                    vortex_err!("Field {field_name} not found while expanding struct root")
                })?;
                return Ok(Transformed {
                    value: expanded_root.children()[index].clone(),
                    changed: true,
                    order: TraversalOrder::Skip,
                });
            }

            if let Some(selection) = scalar_fn.as_opt::<Select>() {
                let names = selection.normalize_to_included_fields(fields.names())?;
                let children = names
                    .iter()
                    .map(|name| {
                        let index = fields
                            .find(name)
                            .vortex_expect("normalized selection fields must exist in the root");
                        expanded_root.children()[index].clone()
                    })
                    .collect();
                return Ok(Transformed {
                    value: bound_pack(names, children)?,
                    changed: true,
                    order: TraversalOrder::Skip,
                });
            }

            Ok(Transformed::no(node))
        })?
        .into_inner())
}

fn step_into_struct_field(
    expression: BoundExpression,
    field_name: &FieldName,
    field_dtype: DType,
) -> VortexResult<BoundExpression> {
    Ok(expression
        .transform_down(|node| {
            let is_field_access = node
                .as_scalar()
                .and_then(|scalar_fn| scalar_fn.as_opt::<GetItem>())
                .is_some_and(|name| name == field_name)
                && node.children()[0].is_root();

            if is_field_access {
                Ok(Transformed {
                    value: BoundExpression::new_root(field_dtype.clone()),
                    changed: true,
                    order: TraversalOrder::Skip,
                })
            } else {
                Ok(Transformed::no(node))
            }
        })?
        .into_inner())
}

fn bound_pack(names: FieldNames, children: Vec<BoundExpression>) -> VortexResult<BoundExpression> {
    BoundExpression::try_new(
        PackFn.bind(PackOptions {
            names,
            nullability: Nullability::NonNullable,
        }),
        children,
    )
}
