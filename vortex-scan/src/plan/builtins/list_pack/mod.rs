// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Assembling lists from their offsets, validity and elements.

use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_session::registry::CachedId;

pub use self::element_index::ElementIndex;
use self::execute::ListSource;
use self::execute::element_selection;
use crate::execution::operators::UNBOUNDED;
use crate::execution::pipeline::Chain;
use crate::execution::pipeline::Compiler;
use crate::plan::Plan;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanSlots;
use crate::plan::PlanVTable;

mod element_index;
pub(crate) mod execute;
pub(crate) mod lengths;
pub(crate) mod pushdown;

const ELEMENTS: usize = 0;
const OFFSETS: usize = 1;
const VALIDITY: usize = 2;

/// Assembles a list from slots `[elements, offsets, validity]`, with validity present exactly
/// when the list is nullable.
///
/// The plan may carry the [`ElementIndex`] of its elements: the list holding every so many
/// element rows, kept wherever the list is described rather than with its data, so it is
/// there before anything is read. With it, compiling bounds the element rows of a selection of
/// lists to within a stride.
#[derive(Clone, Debug)]
pub struct ListPack;

/// Operator-specific list assembly data.
#[derive(Clone, Debug, Default)]
pub struct ListPackData {
    element_index: Option<ElementIndex>,
}

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
        // SAFETY: the caller upholds the same contract.
        unsafe { Self::from_slots_with_element_index_unchecked(dtype, row_count, slots, None) }
    }

    /// [`Self::from_slots_unchecked`], with the `element_index` a writer recorded of the elements.
    ///
    /// # Safety
    ///
    /// As for [`Self::from_slots_unchecked`], and `element_index`, when given, must index the
    /// elements that the offsets child's offsets name.
    pub unsafe fn from_slots_with_element_index_unchecked(
        dtype: DType,
        row_count: u64,
        slots: PlanSlots,
        element_index: Option<ElementIndex>,
    ) -> Self {
        PlanParts {
            vtable: ListPack,
            dtype,
            row_count,
            slots,
            data: ListPackData { element_index },
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

    /// This assembly with `element_index` as the index of its elements.
    ///
    /// It must index the elements that the offsets child's offsets name: an entry that does not
    /// match them makes a scan fail, when the elements it bounds turn out not to hold a list.
    pub fn with_element_index(&self, element_index: ElementIndex) -> Self {
        // SAFETY: the slots are this plan's, which hold its invariants.
        unsafe {
            Self::from_slots_with_element_index_unchecked(
                self.dtype().clone(),
                self.row_count(),
                self.slots().clone(),
                Some(element_index),
            )
        }
    }

    /// Returns the index of the elements, if the plan carries one.
    pub fn element_index(&self) -> Option<&ElementIndex> {
        self.data().element_index.as_ref()
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

    fn metadata(plan: &Plan<Self>) -> Option<Self::Metadata> {
        // Nullability is recoverable from the plan dtype. An element index is not serialized yet.
        plan.element_index().is_none().then_some(EmptyMetadata)
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

    /// Reads the offsets and validity of the selected lists and a bound on their elements, all at
    /// once, and assembles the lists.
    ///
    /// The element rows a selection of lists holds are data: they depend on the offsets. The
    /// plan's element index bounds them to within a stride without reading anything, so every
    /// read of a chunk is known when it is compiled, at any depth of nesting. The list's source
    /// reads the selected lists' offsets beside those elements, and keeps the elements the
    /// offsets name.
    ///
    /// A list without an element index reads every element of the column: correct, and slow for a
    /// selection of few lists.
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

        let offsets = compiler
            .compile(
                &offsets_plan,
                rows.start..rows.end + 1,
                &Mask::new_true(mask.len() + 1),
            )?
            .ok_or_else(|| vortex_err!("List offsets compiled to no rows"))?;
        // The offsets are the source's first inlet, `OFFSETS_INLET`.
        let mut chains = vec![offsets];

        let validity = match plan.validity()? {
            Some(validity) => {
                let chain = compiler
                    .compile(&validity, rows.clone(), &mask)?
                    .ok_or_else(|| vortex_err!("List validity compiled to no rows"))?;
                chains.push(chain);
                Some(chains.len() - 1)
            }
            None => None,
        };

        let read = element_ranges(
            plan.element_index(),
            &rows,
            &mask,
            elements_plan.row_count(),
        )?;
        let elements = match element_selection(&read)? {
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
            read,
            validity,
            elements,
            chains.len(),
            mask,
        );
        Ok(Some(compiler.join_with(chains, |_| UNBOUNDED, source)))
    }
}

/// The element rows to read for the lists `mask` selects of `rows`, merged where they meet, in
/// order. The element index bounds each run of lists to within a stride; without one, every
/// element of the column is read.
fn element_ranges(
    element_index: Option<&ElementIndex>,
    rows: &Range<u64>,
    mask: &Mask,
    element_rows: u64,
) -> VortexResult<Vec<(u64, u64)>> {
    let Some(element_index) = element_index else {
        return Ok((element_rows > 0)
            .then_some((0, element_rows))
            .into_iter()
            .collect());
    };
    let mut ranges: Vec<(u64, u64)> = Vec::new();
    let mut add = |lists: Range<u64>| {
        let range = element_index.element_range(lists, element_rows);
        if range.is_empty() {
            return;
        }
        match ranges.last_mut() {
            Some(previous) if previous.1 >= range.start => previous.1 = previous.1.max(range.end),
            _ => ranges.push((range.start, range.end)),
        }
    };
    match mask.slices() {
        AllOr::All => add(rows.clone()),
        AllOr::None => {}
        AllOr::Some(slices) => {
            for &(start, end) in slices {
                add(rows.start + u64::try_from(start)?..rows.start + u64::try_from(end)?);
            }
        }
    }
    Ok(ranges)
}

fn validate_slots(dtype: &DType, row_count: u64, slots: &PlanSlots) -> VortexResult<()> {
    let elements_dtype = dtype
        .as_list_element_opt()
        .ok_or_else(|| vortex_err!("ListPack output dtype must be a list, got {dtype}"))?;
    let nullability = dtype.nullability();
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
