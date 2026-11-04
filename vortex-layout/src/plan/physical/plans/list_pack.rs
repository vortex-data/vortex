// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::sync::Arc;

use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;

use crate::plan::physical::Plan;
use crate::plan::physical::PlanId;
use crate::plan::physical::PlanParts;
use crate::plan::physical::PlanRef;
use crate::plan::physical::PlanSlots;
use crate::plan::physical::PlanVTable;
use crate::plan::physical::plan_slots;

/// Assembles a list from slots `[elements, offsets, validity]`, with validity present exactly
/// when the list is nullable.
#[derive(Clone, Debug)]
pub struct ListPack;

/// Operator-specific list assembly data.
#[derive(Clone, Debug)]
pub struct ListPackData;

/// A plan that assembles a list from its children.
pub type ListPackPlan = Plan<ListPack>;

/// The child slots of a [`ListPack`] plan.
#[plan_slots(ListPack)]
pub struct ListPackSlots {
    /// The plan producing list elements.
    #[slot(0)]
    pub elements: PlanRef,
    /// The plan producing list offsets.
    #[slot(1)]
    pub offsets: PlanRef,
    /// The plan producing list validity, present exactly when the list is nullable.
    #[slot(2)]
    pub validity: Option<PlanRef>,
}

impl ListPackPlan {
    /// Creates a list assembly from potentially unbuilt slots without validation.
    ///
    /// # Safety
    ///
    /// `slots` must be `[elements, offsets, validity]`. `dtype` must be a list whose element
    /// dtype matches the elements child. The offsets child must be a non-nullable integer with
    /// `row_count + 1` rows. A non-nullable boolean validity child with `row_count` rows must be
    /// present exactly when `dtype` is nullable.
    pub(crate) unsafe fn from_slots_unchecked(
        dtype: DType,
        row_count: u64,
        slots: PlanSlots,
    ) -> Self {
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
        let slots = ListPackSlots {
            elements,
            offsets,
            validity,
        }
        .into_slots();
        validate_slots(&dtype, row_count, &slots)?;

        // SAFETY: All slot shape invariants were validated above.
        Ok(unsafe { Self::from_slots_unchecked(dtype, row_count, slots) })
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
        ListPackSlots::slot_name(slot)
    }
}

fn validate_slots(dtype: &DType, row_count: u64, slots: &PlanSlots) -> VortexResult<()> {
    let elements_dtype = dtype
        .as_list_element_opt()
        .ok_or_else(|| vortex_err!("ListPack output dtype must be a list, got {dtype}"))?;
    let ListPackSlotsView {
        elements,
        offsets,
        validity,
    } = ListPackSlotsView::try_from_slots(slots)?;

    if elements.dtype() != elements_dtype.as_ref() {
        vortex_bail!(
            "ListPack elements child has dtype {} but the list element dtype is {}",
            elements.dtype(),
            elements_dtype
        );
    }

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

    match (dtype.is_nullable(), validity) {
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
