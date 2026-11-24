// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Assembling lists from their offsets, validity and elements.

use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::registry::CachedId;

use self::execute::ListIndex;
use self::execute::ListSource;
use self::execute::Offsets;
use self::execute::element_selection;
use self::execute::element_spans;
use self::execute::offsets_ahead;
use crate::execution::operators::UNBOUNDED;
use crate::execution::pipeline::Chain;
use crate::execution::pipeline::Compiler;
use crate::plan::LazyPlanRef;
use crate::plan::Plan;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanSlots;
use crate::plan::PlanVTable;

pub(crate) mod execute;
pub(crate) mod lengths;
pub(crate) mod pushdown;

const ELEMENTS: usize = 0;
const OFFSETS: usize = 1;
const VALIDITY: usize = 2;
const INDEX: usize = 3;

/// Assembles a list from slots `[elements, offsets, validity, index]`, with validity present
/// exactly when the list is nullable.
///
/// The optional index is a coarse copy of the offsets, for lists whose offsets are too many to
/// read whole ahead of a scan's chunks. It holds `k` list rows followed by the `k` element rows
/// those lists start at, both increasing: consecutive entries bound a window of whole lists.
/// With it, compiling bounds the element rows of a selection of lists to within a window without
/// reading the offsets.
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
    /// `slots` must be `[elements, offsets, validity, index]`. `dtype` must be a list whose element
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
        let slots = [Some(elements), Some(offsets), validity, None]
            .into_iter()
            .collect();
        validate_slots(&dtype, row_count, &slots)?;

        // SAFETY: All slot shape invariants were validated above.
        Ok(unsafe { Self::from_slots_unchecked(dtype, row_count, slots) })
    }

    /// This assembly with `index` as its index of the offsets.
    pub fn with_index(&self, index: PlanRef) -> VortexResult<Self> {
        let mut slots = self.slots().to_vec();
        slots[INDEX] = Some(LazyPlanRef::ready(index));
        let slots = PlanSlots::from(slots);
        validate_slots(self.dtype(), self.row_count(), &slots)?;
        // SAFETY: All slot shape invariants were validated above.
        Ok(unsafe { Self::from_slots_unchecked(self.dtype().clone(), self.row_count(), slots) })
    }

    /// Returns the plan producing the index of the offsets, if the list has one.
    pub fn index(&self) -> VortexResult<Option<PlanRef>> {
        self.slot(INDEX)
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
            INDEX => Cow::Borrowed("index"),
            _ => Cow::Owned(format!("slot[{slot}]")),
        }
    }

    /// Reads the validity of the selected lists and their elements, and assembles them.
    ///
    /// The element rows a selection of lists holds are data: they depend on the offsets. A scan
    /// reads a list column's offsets whole ahead of its chunks and keeps them with the compiler's
    /// whole values, so compiling finds them there and bounds the elements exactly, without
    /// reading anything. Offsets too many to read ahead have an index, read ahead in their place:
    /// compiling bounds the elements with it to within a window, the list's source reads the
    /// selected lists' offsets beside them, and keeps the elements those offsets name. With
    /// neither read ahead, the source reads every element of the column: correct, but only a
    /// scan that reads ahead reads lists well.
    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        let (Some(first), Some(last)) = (mask.first(), mask.last()) else {
            return Ok(None);
        };
        let mask = mask.slice(first..last + 1);
        let rows = rows.start + u64::try_from(first)?..rows.start + u64::try_from(last + 1)?;
        let offsets_plan = plan.offsets()?;
        let elements_plan = plan.elements()?;
        let mut chains = Vec::new();
        // The element rows the elements inlet holds, when the offsets do not say exactly.
        let mut read = Vec::new();

        let ahead = compiler.values().get(&offsets_plan);
        let (offsets, element_rows) = match ahead {
            Some(offsets) => {
                let stored = offsets_ahead(&offsets, rows.clone(), compiler.session())?;
                let selection = element_selection(&element_spans(&stored, &mask))?;
                (Offsets::Ahead(stored), selection)
            }
            None => {
                let chain = compiler
                    .compile(
                        &offsets_plan,
                        rows.start..rows.end + 1,
                        &Mask::new_true(mask.len() + 1),
                    )?
                    .ok_or_else(|| vortex_err!("List offsets compiled to no rows"))?;
                chains.push(chain);
                let elements = elements_plan.row_count();
                let index = match plan.index()? {
                    Some(index) => compiler.values().get(&index),
                    None => None,
                };
                read = match index {
                    Some(index) => ListIndex::try_new(&index)?.windows(&rows, &mask, elements),
                    None => {
                        tracing::debug!(
                            rows = ?rows,
                            "list offsets were not read ahead: reading every element of the column"
                        );
                        if elements > 0 {
                            vec![(0, elements)]
                        } else {
                            Vec::new()
                        }
                    }
                };
                (Offsets::Inlet(0), element_selection(&read)?)
            }
        };

        let validity = match plan.validity()? {
            Some(validity) => {
                let chain = compiler
                    .compile(&validity, rows, &mask)?
                    .ok_or_else(|| vortex_err!("List validity compiled to no rows"))?;
                chains.push(chain);
                Some(chains.len() - 1)
            }
            None => None,
        };
        let elements = match element_rows {
            Some((element_rows, element_mask)) => compiler
                .compile(&elements_plan, element_rows, &element_mask)?
                .map(|chain| {
                    chains.push(chain);
                    chains.len() - 1
                }),
            None => None,
        };

        let source = ListSource::new(
            elements_plan.dtype().clone(),
            offsets_plan.dtype().clone(),
            offsets,
            read,
            validity,
            elements,
            chains.len(),
            mask,
        );
        Ok(Some(compiler.join_with(chains, |_| UNBOUNDED, source)))
    }
}

fn validate_slots(dtype: &DType, row_count: u64, slots: &PlanSlots) -> VortexResult<()> {
    let elements_dtype = dtype
        .as_list_element_opt()
        .ok_or_else(|| vortex_err!("ListPack output dtype must be a list, got {dtype}"))?;
    let nullability = dtype.nullability();
    slots.check_len("ListPack", INDEX + 1)?;
    if let Some(index) = slots.get(INDEX)? {
        if index.dtype() != &DType::Primitive(PType::U64, Nullability::NonNullable)
            || !index.row_count().is_multiple_of(2)
        {
            vortex_bail!(
                "ListPack index child must hold pairs of non-nullable u64, got {} rows of {}",
                index.row_count(),
                index.dtype()
            );
        }
    }

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

    match (nullability.is_nullable(), slots.get(VALIDITY)?) {
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

#[cfg(test)]
mod tests;
