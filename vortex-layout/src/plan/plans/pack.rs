// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;

use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;

use crate::plan::Plan;
use crate::plan::PlanChildren;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanVTable;

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
