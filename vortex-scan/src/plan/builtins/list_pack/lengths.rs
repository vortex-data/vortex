// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The lengths of lists, from their offsets alone.

use std::borrow::Cow;
use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_session::registry::CachedId;

use super::execute::Offsets;
use super::execute::offsets_ahead;
use super::execute::wide_offsets_dtype;
use crate::execution::operator::Cx;
use crate::execution::operator::Input;
use crate::execution::operator::Operator;
use crate::execution::operator::Source;
use crate::execution::operator::SourceBlocked;
use crate::execution::operator::SourceReady;
use crate::execution::operator::Step;
use crate::execution::operators::UNBOUNDED;
use crate::execution::operators::join_inlet;
use crate::execution::pipeline::Chain;
use crate::execution::pipeline::Compiler;
use crate::plan::Plan;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanSlots;
use crate::plan::PlanVTable;

const OFFSETS: usize = 0;
const VALIDITY: usize = 1;

/// The lengths of lists, from slots `[offsets, validity]`, with validity present exactly when the
/// lists are nullable. A null list has a null length.
///
/// An expression that reads a list only through `list_length` is rewritten over this plan, so it
/// reads no element. With the offsets read ahead of a scan's chunks, it reads nothing but the
/// validity.
#[derive(Clone, Debug)]
pub struct ListLengths;

/// Operator-specific list lengths data.
#[derive(Clone, Debug)]
pub struct ListLengthsData;

/// A plan that produces the lengths of lists.
pub type ListLengthsPlan = Plan<ListLengths>;

impl ListLengthsPlan {
    /// Creates the lengths of the lists `offsets` bound.
    ///
    /// `validity` is required exactly when `nullability` is [`Nullability::Nullable`]. The row
    /// domain is one fewer than the number of offsets.
    pub fn try_new(
        nullability: Nullability,
        row_count: u64,
        offsets: PlanRef,
        validity: Option<PlanRef>,
    ) -> VortexResult<Self> {
        let slots = [Some(offsets), validity].into_iter().collect();
        let dtype = DType::Primitive(PType::U64, nullability);
        validate_slots(&dtype, row_count, &slots)?;
        Ok(PlanParts {
            vtable: ListLengths,
            dtype,
            row_count,
            slots,
            data: ListLengthsData,
        }
        .into_typed())
    }

    /// Returns the plan producing list offsets.
    pub fn offsets(&self) -> VortexResult<PlanRef> {
        self.slot_required(OFFSETS)
    }

    /// Returns the plan producing list validity, if the lists are nullable.
    pub fn validity(&self) -> VortexResult<Option<PlanRef>> {
        self.slot(VALIDITY)
    }
}

impl PlanVTable for ListLengths {
    type PlanData = ListLengthsData;
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.list_lengths");
        *ID
    }

    fn metadata(_plan: &Plan<Self>) -> Option<Self::Metadata> {
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
            OFFSETS => Cow::Borrowed("offsets"),
            VALIDITY => Cow::Borrowed("validity"),
            _ => Cow::Owned(format!("slot[{slot}]")),
        }
    }

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
        let mut chains = Vec::new();
        let offsets = match compiler.values().get(&offsets_plan) {
            Some(offsets) => {
                Offsets::Ahead(offsets_ahead(&offsets, rows.clone(), compiler.session())?)
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
                Offsets::Inlet(0)
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
        let source = ListLengthsSource {
            offsets_dtype: offsets_plan.dtype().clone(),
            offsets,
            validity,
            inlets: chains.len(),
            mask,
            done: false,
        };
        Ok(Some(compiler.join_with(chains, |_| UNBOUNDED, source)))
    }
}

fn validate_slots(dtype: &DType, row_count: u64, slots: &PlanSlots) -> VortexResult<()> {
    slots.check_len("ListLengths", VALIDITY + 1)?;
    let offsets = slots.required(OFFSETS)?;
    if !offsets.dtype().is_int() || offsets.dtype().is_nullable() {
        vortex_bail!(
            "ListLengths offsets child must have a non-nullable integer dtype, got {}",
            offsets.dtype()
        );
    }
    vortex_ensure_eq!(
        offsets.row_count(),
        row_count
            .checked_add(1)
            .ok_or_else(|| vortex_err!("ListLengths offsets row count overflow"))?
    );
    match (dtype.is_nullable(), slots.get(VALIDITY)?) {
        (true, Some(validity)) => {
            vortex_ensure_eq!(validity.dtype(), &DType::Bool(Nullability::NonNullable));
            vortex_ensure_eq!(validity.row_count(), row_count);
        }
        (false, None) => {}
        (true, None) => vortex_bail!("Nullable ListLengths has no validity child"),
        (false, Some(_)) => vortex_bail!("Non-nullable ListLengths has a validity child"),
    }
    Ok(())
}

/// Emits the lengths of the lists a mask selects, once every inlet has closed.
struct ListLengthsSource {
    offsets_dtype: DType,
    offsets: Offsets,
    validity: Option<usize>,
    inlets: usize,
    /// The lists whose lengths to produce, over the rows the offsets cover.
    mask: Mask,
    done: bool,
}

impl ListLengthsSource {
    fn lengths(&mut self, cx: &mut Cx<'_>) -> VortexResult<ArrayRef> {
        let lengths = match &self.offsets {
            Offsets::Ahead(stored) => selected_lengths(stored, &self.mask),
            Offsets::Inlet(inlet) => {
                let stored = join_inlet(cx, *inlet, &self.offsets_dtype)?
                    .cast(wide_offsets_dtype())?
                    .execute::<PrimitiveArray>(cx.exec())?;
                let stored = stored.as_slice::<u64>();
                vortex_ensure_eq!(stored.len(), self.mask.len() + 1);
                vortex_ensure!(stored.is_sorted(), "List offsets decrease");
                selected_lengths(stored, &self.mask)
            }
        };
        let validity = match self.validity {
            Some(inlet) => Validity::Array(join_inlet(
                cx,
                inlet,
                &DType::Bool(Nullability::NonNullable),
            )?),
            None => Validity::NonNullable,
        };
        Ok(PrimitiveArray::new(lengths.freeze(), validity).into_array())
    }
}

/// The length of each list `mask` selects. `stored` holds the offsets of the lists `mask` covers,
/// and one more.
fn selected_lengths(stored: &[u64], mask: &Mask) -> BufferMut<u64> {
    let mut lengths = BufferMut::<u64>::with_capacity(mask.true_count());
    let mut add = |start: usize, end: usize| {
        lengths.extend(stored[start..=end].windows(2).map(|pair| pair[1] - pair[0]));
    };
    match mask.slices() {
        AllOr::All => add(0, mask.len()),
        AllOr::None => {}
        AllOr::Some(slices) => {
            for &(start, end) in slices {
                add(start, end);
            }
        }
    }
    lengths
}

impl Source for ListLengthsSource {
    fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        for index in 0..self.inlets {
            if !cx.inlet(index).closed() {
                return Ok(SourceReady::Blocked(SourceBlocked::Inlet(index)));
            }
        }
        Ok(SourceReady::Ready)
    }
}

impl Operator for ListLengthsSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        if self.done {
            return Ok(Step::End);
        }
        self.done = true;
        Ok(Step::Final(self.lengths(cx)?))
    }
}
