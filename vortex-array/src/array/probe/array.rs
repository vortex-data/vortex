// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::needs_drop;

use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::array::probe::RepeatedArrayProbe;
use crate::array::probe::RepeatedState;
use crate::array::probe::repeated::child_probe;
use crate::arrays::Primitive;
use crate::scalar::Scalar;
use crate::vtable::OperationsVTable;

/// A borrowed row accessor.
///
/// Either a one-off reader over a borrowed array, which retains nothing, or a borrow of a
/// [`RepeatedArrayProbe`] that keeps state between reads. Both variants hold only references,
/// so an `ArrayProbe` has no destructor and building one per read, including for every child
/// slot of a nested array, costs nothing.
pub enum ArrayProbe<'a> {
    /// A single read of a borrowed array; nothing outlives the call.
    Once(&'a ArrayRef),
    /// A read through a retained probe.
    Repeated(&'a mut RepeatedArrayProbe),
}

// Everything built per read must be free to drop, or the unwind path pins it in memory and
// blocks the tail call into the encoding. Keep these at compile time.
const _: () = assert!(!needs_drop::<ArrayProbe<'_>>());
const _: () = assert!(!needs_drop::<ProbeState<'_, Primitive>>());

impl ArrayProbe<'_> {
    /// The array this probe reads from.
    #[inline]
    pub fn array(&self) -> &ArrayRef {
        match self {
            Self::Once(array) => array,
            Self::Repeated(probe) => probe.array(),
        }
    }

    /// Read the scalar at `index`, including its nullness.
    #[inline]
    pub fn execute_scalar(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
        match self {
            Self::Once(array) => execute_scalar_once(array, index, ctx),
            Self::Repeated(probe) => probe.execute_scalar(index, ctx),
        }
    }

    /// Whether the row at `index` is valid.
    #[inline]
    pub fn execute_is_valid(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        match self {
            Self::Once(array) => execute_is_valid_once(array, index, ctx),
            Self::Repeated(probe) => probe.execute_is_valid(index, ctx),
        }
    }

    /// Whether the row at `index` is null.
    #[inline]
    pub fn execute_is_invalid(
        &mut self,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        Ok(!self.execute_is_valid(index, ctx)?)
    }
}

/// One-off scalar read: nothing outlives the call.
#[inline]
fn execute_scalar_once(
    array: &ArrayRef,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    if !execute_is_valid_once(array, index, ctx)? {
        return Ok(Scalar::null(array.dtype().clone()));
    }
    check_dtype(
        array,
        array.dyn_array().probe_scalar_once(array, index, ctx),
    )
}

/// One-off validity read: nothing outlives the call.
#[inline]
fn execute_is_valid_once(
    array: &ArrayRef,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    check_bounds(array, index)?;
    if !array.dtype().is_nullable() {
        return Ok(true);
    }
    // Matching the validity directly keeps this path free of any retained temporary.
    array.validity()?.execute_is_valid(index, ctx)
}

/// Pass an encoding's result through, checking its dtype in debug builds.
///
/// Unwrapping and re-wrapping the result here would cost an extra copy of the scalar on every
/// read.
#[inline]
pub(super) fn check_dtype(array: &ArrayRef, result: VortexResult<Scalar>) -> VortexResult<Scalar> {
    result.inspect(|scalar| {
        debug_assert_eq!(scalar.dtype(), array.dtype(), "Scalar dtype mismatch");
    })
}

#[inline]
pub(super) fn check_bounds(array: &ArrayRef, index: usize) -> VortexResult<()> {
    if index >= array.len() {
        return Err(out_of_bounds(index, array.len()));
    }
    Ok(())
}

/// Kept out of line so the error path, which captures a backtrace, does not count against the
/// hot probe bodies when the inliner sizes them.
#[cold]
#[inline(never)]
fn out_of_bounds(index: usize, len: usize) -> VortexError {
    vortex_err!(OutOfBounds: index, 0, len)
}

/// The encoding state type of `V`'s operations vtable.
pub type EncodingProbeState<V> =
    <<V as VTable>::OperationsVTable as OperationsVTable<V>>::ProbeState;

/// Everything an encoding's `probe_scalar` runs with: the typed view of the array being read
/// and, for a repeated read, a borrow of the state its [`RepeatedArrayProbe`] keeps.
///
/// Passed to [`OperationsVTable::probe_scalar`].
/// A one-off read gets [`ProbeState::once`], which holds only the view. A repeated read borrows
/// the encoding's own state and the lazily created child probes. Neither owns anything, so
/// building one per read is free.
///
/// Encodings never inspect the policy: [`ProbeState::slot`] hands out a probe over a child under
/// whichever policy is in force, [`ProbeState::retained`] hands out the encoding state only when
/// it is kept, and [`ProbeState::split`] gives both at once.
pub struct ProbeState<'a, V: VTable> {
    array: ArrayView<'a, V>,
    retained: Option<&'a mut RepeatedState<EncodingProbeState<V>>>,
}

impl<'a, V: VTable> ProbeState<'a, V> {
    /// State for a single read of `array`.
    ///
    /// Encodings use it to run their `probe_scalar` path from the deprecated `scalar_at`; it
    /// goes away with `scalar_at`.
    #[inline]
    pub fn once(array: ArrayView<'a, V>) -> Self {
        Self {
            array,
            retained: None,
        }
    }

    /// State for a read through a [`RepeatedArrayProbe`], borrowing what it keeps.
    #[inline]
    pub(crate) fn repeated(
        array: ArrayView<'a, V>,
        retained: &'a mut RepeatedState<EncodingProbeState<V>>,
    ) -> Self {
        Self {
            array,
            retained: Some(retained),
        }
    }

    /// The typed view of the array being read.
    #[inline]
    pub fn array(&self) -> ArrayView<'a, V> {
        self.array
    }

    /// The encoding's retained state, or `None` for a one-off read.
    ///
    /// Encodings whose repeated algorithm differs from their one-off one branch on this. To hold
    /// the state while reading children, use [`Self::split`].
    #[inline]
    pub fn retained(&mut self) -> Option<&mut EncodingProbeState<V>> {
        self.retained.as_deref_mut().map(RepeatedState::state_mut)
    }

    /// The encoding's retained state and the array's children, borrowed apart, so a cache can
    /// stay borrowed while children are read.
    #[inline]
    pub fn split(&mut self) -> (Option<&mut EncodingProbeState<V>>, ProbeChildren<'_, 'a, V>) {
        let array = self.array;
        match &mut self.retained {
            None => (None, ProbeChildren { array, slots: None }),
            Some(repeated) => {
                let (state, slots) = repeated.split_mut();
                let children = ProbeChildren {
                    array,
                    slots: Some(slots),
                };
                (Some(state), children)
            }
        }
    }

    /// A probe over the array's child in `slot`, under this state's policy.
    ///
    /// Shorthand for [`Self::split`] followed by [`ProbeChildren::slot`]. `None` if the slot is
    /// absent, an error if it is out of bounds.
    #[inline]
    pub fn slot(&mut self, slot: usize) -> VortexResult<Option<ArrayProbe<'_>>> {
        let parent = self.array.array();
        match &mut self.retained {
            None => Ok(child_of(parent, slot)?.map(ArrayProbe::Once)),
            Some(repeated) => {
                Ok(child_probe(repeated.split_mut().1, parent, slot)?.map(ArrayProbe::Repeated))
            }
        }
    }
}

/// The child slots of the array a [`ProbeState`] reads, borrowed apart from the encoding state
/// by [`ProbeState::split`].
pub struct ProbeChildren<'s, 'a, V: VTable> {
    array: ArrayView<'a, V>,
    /// The retained slot table, or `None` for a one-off read.
    slots: Option<&'s mut Vec<Option<RepeatedArrayProbe>>>,
}

impl<V: VTable> ProbeChildren<'_, '_, V> {
    /// A probe over the child in `slot`, under the read's policy.
    ///
    /// For a one-off read this borrows the child; for a repeated read it is the
    /// [`RepeatedArrayProbe`] kept in the slot table, created on first use. `None` if the slot
    /// is absent, an error if it is out of bounds.
    #[inline]
    pub fn slot(&mut self, slot: usize) -> VortexResult<Option<ArrayProbe<'_>>> {
        let parent = self.array.array();
        match &mut self.slots {
            None => Ok(child_of(parent, slot)?.map(ArrayProbe::Once)),
            Some(slots) => Ok(child_probe(slots, parent, slot)?.map(ArrayProbe::Repeated)),
        }
    }
}

/// The child of `parent` in `slot`: `None` if the slot is absent, an error if it is out of bounds.
#[inline]
pub(super) fn child_of(parent: &ArrayRef, slot: usize) -> VortexResult<Option<&ArrayRef>> {
    Ok(parent
        .slots()
        .get(slot)
        .ok_or_else(|| vortex_err!("Probe slot {slot} is out of bounds"))?
        .as_ref())
}
#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::*;
    use crate::VortexSessionExecute;
    use crate::array::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::Struct;
    use crate::arrays::StructArray;

    fn nullable_ints() -> ArrayRef {
        PrimitiveArray::from_option_iter([Some(10i32), None, Some(30)]).into_array()
    }

    fn check_reads(mut probe: ArrayProbe<'_>, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        assert!(probe.execute_scalar(3, ctx).is_err());
        assert!(probe.execute_is_valid(3, ctx).is_err());
        assert!(probe.execute_scalar(1, ctx)?.is_null());
        assert!(!probe.execute_is_valid(1, ctx)?);
        assert!(probe.execute_is_invalid(1, ctx)?);
        assert_eq!(probe.execute_scalar(2, ctx)?, Scalar::from(Some(30i32)));
        assert_eq!(probe.execute_scalar(0, ctx)?, Scalar::from(Some(10i32)));
        Ok(())
    }

    #[test]
    fn once_probe_checks_bounds_and_nulls() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let array = nullable_ints();
        check_reads(array.probe(), &mut ctx)
    }

    #[test]
    fn once_state_reads_children_without_retaining() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let array = struct_of_two_fields()?;
        let typed = array
            .as_opt::<Struct>()
            .ok_or_else(|| vortex_err!("expected a struct"))?;
        let mut state = ProbeState::once(typed);

        assert!(state.slot(5).is_err());
        // Slot 0 is the struct's absent validity.
        assert!(state.slot(0)?.is_none());
        {
            let mut field = state.slot(2)?.ok_or_else(|| vortex_err!("missing field"))?;
            assert!(matches!(field, ArrayProbe::Once(_)));
            assert_eq!(field.array().len(), 2);
            assert_eq!(field.execute_scalar(1, &mut ctx)?, Scalar::from(4i64));
            assert!(field.execute_is_valid(0, &mut ctx)?);
        }
        assert!(state.retained().is_none());
        let (retained, mut children) = state.split();
        assert!(retained.is_none());
        let mut field = children
            .slot(1)?
            .ok_or_else(|| vortex_err!("missing field"))?;
        assert_eq!(field.execute_scalar(1, &mut ctx)?, Scalar::from(2i32));
        Ok(())
    }

    fn struct_of_two_fields() -> VortexResult<ArrayRef> {
        Ok(StructArray::from_fields(&[
            ("a", PrimitiveArray::from_iter([1i32, 2]).into_array()),
            ("b", PrimitiveArray::from_iter([3i64, 4]).into_array()),
        ])?
        .into_array())
    }
}
