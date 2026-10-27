// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hasher;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::Interleave;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::EqMode;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::ArraySlots;
use crate::array::TypedArrayRef;
use crate::array_slots;
use crate::dtype::DType;

/// Slot layout of an interleave: two selectors followed by its value arrays.
#[array_slots(Interleave)]
pub(super) struct InterleaveSlots {
    /// The value-array index for each output row.
    #[slot(0)]
    pub array_indices: ArrayRef,
    /// The row index within the selected value array for each output row.
    #[slot(1)]
    pub row_indices: ArrayRef,
    /// The independent source arrays selected by the two selectors.
    #[slot(2..)]
    pub values: Vec<ArrayRef>,
}

/// An [`Interleave`]-encoded Vortex array. See the [module docs](super) for the specification.
pub type InterleaveArray = Array<Interleave>;

/// Per-array metadata for an [`InterleaveArray`].
///
/// The selectors occupy the leading slots, followed by the `num_values` value arrays.
#[derive(Clone, Debug)]
pub struct InterleaveData {
    pub(crate) num_values: usize,
}

impl Display for InterleaveData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "num_values: {}", self.num_values)
    }
}

impl ArrayHash for InterleaveData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        state.write_usize(self.num_values);
    }
}

impl ArrayEq for InterleaveData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.num_values == other.num_values
    }
}

/// Accessors for the values and selectors of an [`InterleaveArray`].
pub trait InterleaveArrayExt: TypedArrayRef<Interleave> {
    /// The number of value arrays (two fewer than the number of children).
    fn num_values(&self) -> usize {
        self.num_values
    }

    /// The `idx`-th value array (holding the rows that `array_indices` routes to it).
    fn value(&self, idx: usize) -> &ArrayRef {
        self.slots()[InterleaveSlots::VALUES_OFFSET + idx]
            .as_ref()
            .vortex_expect("validated interleave value slot")
    }

    /// The selector routing each output row to a value array.
    fn array_indices(&self) -> &ArrayRef {
        self.slots()[InterleaveSlots::ARRAY_INDICES]
            .as_ref()
            .vortex_expect("validated interleave array_indices slot")
    }

    /// The selector naming each output row's position within its value array.
    fn row_indices(&self) -> &ArrayRef {
        self.slots()[InterleaveSlots::ROW_INDICES]
            .as_ref()
            .vortex_expect("validated interleave row_indices slot")
    }
}
impl<T: TypedArrayRef<Interleave>> InterleaveArrayExt for T {}

impl Array<Interleave> {
    /// Constructs a new [`InterleaveArray`] from `values` and the `array_indices` / `row_indices`
    /// selectors.
    ///
    /// See the [module docs](super) for the full specification and invariants. The selectors must be
    /// non-nullable: they record a definite `(array_index, row_index)` per row, so null-predicate
    /// handling is the caller's responsibility, resolved before the interleave is constructed. The
    /// per-row bounds on the selector values are a runtime precondition checked during execution.
    pub fn try_new(
        values: Vec<ArrayRef>,
        array_indices: ArrayRef,
        row_indices: ArrayRef,
    ) -> VortexResult<Self> {
        let dtype = Interleave::check(&values, &array_indices, &row_indices)?;
        // SAFETY: `check` just validated every invariant and computed the matching `dtype`.
        Ok(unsafe { Self::new_unchecked(values, array_indices, row_indices, dtype) })
    }

    /// Constructs an [`InterleaveArray`] without re-validating the spec invariants.
    ///
    /// This is the assembly half of [`try_new`](Self::try_new): it lays out the selectors followed
    /// by the value arrays and stamps `dtype`, skipping the `Interleave::check` pass for callers
    /// that already uphold the invariants.
    ///
    /// # Safety
    ///
    /// The caller must uphold every [module invariant](super): at least two `values` sharing a dtype
    /// up to nullability, both selectors non-nullable unsigned integers of equal length, and `dtype`
    /// equal to the value returned by `Interleave::check` for these arguments (the shared value
    /// type with the union of the values' nullabilities).
    pub unsafe fn new_unchecked(
        values: Vec<ArrayRef>,
        array_indices: ArrayRef,
        row_indices: ArrayRef,
        dtype: DType,
    ) -> Self {
        let slots = InterleaveSlots {
            array_indices,
            row_indices,
            values,
        }
        .into_slots();

        // SAFETY: the caller of `new_unchecked` upholds every invariant; here we only assemble the
        // canonical slot layout (`array_indices`, `row_indices`, then values) that follows.
        unsafe { Self::new_unchecked_slots(slots, dtype) }
    }

    /// Constructs an [`InterleaveArray`] from a pre-assembled `slots` buffer, skipping both the
    /// spec re-check and the slot copy that [`new_unchecked`](Self::new_unchecked) performs.
    ///
    /// This is the lowest-level assembly path: it stamps `dtype` onto `slots` as-is. Callers that
    /// already own a correctly laid-out [`ArraySlots`] — for example reusing a validated parent's
    /// selectors while swapping its values — avoid materializing an intermediate `Vec<ArrayRef>`
    /// and re-pushing it into a fresh `ArraySlots`.
    ///
    /// # Safety
    ///
    /// The caller must uphold every [module invariant](super) *and* the slot layout: the
    /// `array_indices` selector comes first, followed by the `row_indices` selector and then the
    /// value arrays. Every slot must be `Some`, there must be at least two values, and `dtype`
    /// must equal the value returned by `Interleave::check` for these arguments.
    pub unsafe fn new_unchecked_slots(slots: ArraySlots, dtype: DType) -> Self {
        let num_values = slots.len() - InterleaveSlots::VALUES_OFFSET;
        let len = slots[InterleaveSlots::ARRAY_INDICES]
            .as_ref()
            .vortex_expect("interleave array_indices slot present")
            .len();

        unsafe {
            Array::from_parts_unchecked(ArrayParts::new(
                Interleave,
                dtype,
                len,
                InterleaveData { num_values },
                slots,
            ))
        }
    }
}
