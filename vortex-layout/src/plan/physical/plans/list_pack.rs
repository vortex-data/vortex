// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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

use crate::layouts::list::ElementSamples;
use crate::plan::physical::Plan;
use crate::plan::physical::PlanId;
use crate::plan::physical::PlanParts;
use crate::plan::physical::PlanRef;
use crate::plan::physical::PlanSlots;
use crate::plan::physical::PlanVTable;
use crate::plan::physical::operators::ListSource;
use crate::plan::physical::pipeline::Chain;
use crate::plan::physical::pipeline::Compiler;

const ELEMENTS: usize = 0;
const OFFSETS: usize = 1;
const VALIDITY: usize = 2;

/// Assembles a list from slots `[elements, offsets, validity]`, with validity present exactly
/// when the list is nullable.
#[derive(Clone, Debug)]
pub struct ListPack;

/// Operator-specific list assembly data.
#[derive(Clone, Debug)]
pub struct ListPackData {
    /// The list holding every so many elements, when the writer recorded them: they map a
    /// selection of lists to the elements it reads without reading the offsets first.
    samples: Option<ElementSamples>,
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
    pub(crate) unsafe fn from_slots_unchecked(
        dtype: DType,
        row_count: u64,
        slots: PlanSlots,
    ) -> Self {
        // SAFETY: the caller upholds the same contract.
        unsafe { Self::from_slots_with_samples_unchecked(dtype, row_count, slots, None) }
    }

    /// [`Self::from_slots_unchecked`], reading the elements of a selection of lists by the
    /// `samples` the writer recorded.
    ///
    /// # Safety
    ///
    /// As for [`Self::from_slots_unchecked`], and `samples` must sample the elements and offsets
    /// slots.
    pub(crate) unsafe fn from_slots_with_samples_unchecked(
        dtype: DType,
        row_count: u64,
        slots: PlanSlots,
        samples: Option<ElementSamples>,
    ) -> Self {
        PlanParts {
            vtable: ListPack,
            dtype,
            row_count,
            slots,
            data: ListPackData { samples },
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

    /// The element ranges holding the lists `mask` selects of `rows`, merged and in order. The
    /// samples bound them to within a stride; without samples, every element is read.
    fn element_ranges(
        &self,
        rows: &Range<u64>,
        mask: &Mask,
        element_rows: u64,
    ) -> VortexResult<Vec<Range<u64>>> {
        let Some(samples) = &self.data().samples else {
            return Ok((element_rows > 0)
                .then_some(0..element_rows)
                .into_iter()
                .collect());
        };
        let mut ranges: Vec<Range<u64>> = Vec::new();
        let mut add = |lists: Range<u64>| {
            let range = samples.element_range(lists, element_rows);
            if range.is_empty() {
                return;
            }
            match ranges.last_mut() {
                Some(previous) if previous.end >= range.start => {
                    previous.end = previous.end.max(range.end)
                }
                _ => ranges.push(range),
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

    /// Reads the offsets from the first selected list to the last, the validity of the selected
    /// lists, and the elements the samples place them in, all compiled now, so they are read
    /// together. The source drops the elements of lists the mask leaves out.
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
        let offsets = compiler
            .compile(
                &offsets_plan,
                rows.start..rows.end + 1,
                &Mask::new_true(mask.len() + 1),
            )?
            .ok_or_else(|| vortex_err!("List offsets compiled to no rows"))?;
        let mut chains = vec![offsets];
        if let Some(validity) = plan.validity()? {
            chains.push(
                compiler
                    .compile(&validity, rows.clone(), &mask)?
                    .ok_or_else(|| vortex_err!("List validity compiled to no rows"))?,
            );
        }
        let elements_plan = plan.elements()?;
        let ranges = plan.element_ranges(&rows, &mask, elements_plan.row_count())?;
        let elements = match (ranges.first(), ranges.last()) {
            (Some(first), Some(last)) => {
                let (start, end) = (first.start, last.end);
                let len = usize::try_from(end - start)?;
                let element_mask = if ranges.len() == 1 {
                    Mask::new_true(len)
                } else {
                    Mask::from_slices(
                        len,
                        ranges
                            .iter()
                            .map(|range| -> VortexResult<_> {
                                Ok((
                                    usize::try_from(range.start - start)?,
                                    usize::try_from(range.end - start)?,
                                ))
                            })
                            .collect::<VortexResult<Vec<_>>>()?,
                    )
                };
                compiler.compile(&elements_plan, start..end, &element_mask)?
            }
            _ => None,
        };
        let has_elements = elements.is_some();
        chains.extend(elements);
        let source = ListSource::new(
            elements_plan.dtype().clone(),
            offsets_plan.dtype().clone(),
            plan.dtype().nullability(),
            mask,
            has_elements.then_some(ranges),
        );
        Ok(Some(compiler.join_with(
            chains,
            ListSource::capacity,
            source,
        )))
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
