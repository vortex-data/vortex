// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::probe::ArrayProbe;
use crate::array::probe::array::check_bounds;
use crate::array::probe::array::check_dtype;
use crate::array::probe::array::child_of;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// A row accessor that owns its array and keeps preparation between reads.
///
/// The encoding's state, the validity probe and the probes over child slots are created on
/// first use and reused by every following read. Dropping the probe drops all of them. Array
/// handles share their buffers, so the probe can outlive the handle it was built from. Probes
/// are local to a thread.
pub struct RepeatedArrayProbe {
    array: ArrayRef,
    /// The encoding's [`RepeatedState`], type-erased, created on the first read.
    state: Option<Box<dyn Any>>,
    /// Resolved on the first read of a nullable array: `Some(valid)` for uniform validity,
    /// otherwise `validity` holds a probe over the validity array, boxed because it is a probe.
    uniform_validity: Option<bool>,
    validity: Option<Box<RepeatedArrayProbe>>,
}

impl RepeatedArrayProbe {
    /// Own an array for repeated reads.
    pub fn new(array: ArrayRef) -> Self {
        Self {
            array,
            state: None,
            uniform_validity: None,
            validity: None,
        }
    }

    /// The array this probe reads from.
    pub fn array(&self) -> &ArrayRef {
        &self.array
    }

    /// Borrow this probe as an [`ArrayProbe`].
    #[inline]
    pub fn as_probe(&mut self) -> ArrayProbe<'_> {
        ArrayProbe::Repeated(self)
    }

    /// Read the scalar at `index`, including its nullness, reusing retained preparation.
    pub fn execute_scalar(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
        if !self.execute_is_valid(index, ctx)? {
            return Ok(Scalar::null(self.array.dtype().clone()));
        }
        let result =
            self.array
                .dyn_array()
                .probe_scalar_retained(&self.array, index, &mut self.state, ctx);
        check_dtype(&self.array, result)
    }

    /// Whether the row at `index` is valid, through the retained validity.
    pub fn execute_is_valid(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        check_bounds(&self.array, index)?;
        if !self.array.dtype().is_nullable() {
            return Ok(true);
        }
        if let Some(valid) = self.uniform_validity {
            return Ok(valid);
        }
        if self.validity.is_none() {
            match self.array.validity()? {
                Validity::NonNullable | Validity::AllValid => {
                    self.uniform_validity = Some(true);
                    return Ok(true);
                }
                Validity::AllInvalid => {
                    self.uniform_validity = Some(false);
                    return Ok(false);
                }
                Validity::Array(array) => {
                    self.validity = Some(Box::new(RepeatedArrayProbe::new(array)));
                }
            }
        }
        self.validity
            .as_mut()
            .ok_or_else(|| vortex_err!("validity probe was just initialized"))?
            .execute_scalar(index, ctx)?
            .as_bool()
            .value()
            .ok_or_else(|| vortex_err!("validity value at index {index} is null"))
    }

    /// Whether the row at `index` is null.
    pub fn execute_is_invalid(
        &mut self,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        Ok(!self.execute_is_valid(index, ctx)?)
    }
}

/// Get or create the [`RepeatedState`] for encoding state `S` in a probe's erased slot.
pub(crate) fn repeated_state<S: Default + 'static>(
    slot: &mut Option<Box<dyn Any>>,
) -> VortexResult<&mut RepeatedState<S>> {
    slot.get_or_insert_with(|| Box::new(RepeatedState::<S>::default()))
        .downcast_mut::<RepeatedState<S>>()
        .ok_or_else(|| vortex_err!("Probe state type mismatch"))
}

/// What a [`RepeatedArrayProbe`] keeps for its encoding between reads.
pub(crate) struct RepeatedState<S> {
    state: S,
    /// Probes over the source's child slots, allocated on first use. Slots that are never
    /// requested stay empty.
    slots: Vec<Option<RepeatedArrayProbe>>,
}

impl<S: Default> Default for RepeatedState<S> {
    fn default() -> Self {
        Self {
            state: S::default(),
            slots: Vec::new(),
        }
    }
}

impl<S> RepeatedState<S> {
    #[inline]
    pub(super) fn state_mut(&mut self) -> &mut S {
        &mut self.state
    }

    /// The encoding state and the child slot table as disjoint borrows.
    #[inline]
    pub(super) fn split_mut(&mut self) -> (&mut S, &mut Vec<Option<RepeatedArrayProbe>>) {
        (&mut self.state, &mut self.slots)
    }
}

/// Get or create the retained probe over `parent`'s child in `slot`.
pub(super) fn child_probe<'s>(
    slots: &'s mut Vec<Option<RepeatedArrayProbe>>,
    parent: &ArrayRef,
    slot: usize,
) -> VortexResult<Option<&'s mut RepeatedArrayProbe>> {
    let Some(child) = child_of(parent, slot)? else {
        return Ok(None);
    };
    if slots.is_empty() {
        slots.resize_with(parent.slots().len(), || None);
    }
    Ok(Some(slots[slot].get_or_insert_with(|| {
        RepeatedArrayProbe::new(child.clone())
    })))
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::*;
    use crate::VortexSessionExecute;
    use crate::array::IntoArray;
    use crate::array::probe::ProbeState;
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
    fn repeated_probe_initializes_lazily_and_outlives_handle() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let array = nullable_ints();
        let mut probe = RepeatedArrayProbe::new(array.clone());
        drop(array);
        assert!(probe.state.is_none());
        assert!(probe.validity.is_none());

        check_reads(probe.as_probe(), &mut ctx)?;

        assert!(probe.state.is_some());
        assert!(probe.validity.is_some());
        assert!(probe.uniform_validity.is_none());
        Ok(())
    }

    #[test]
    fn repeated_probe_resolves_uniform_validity_once() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let array = PrimitiveArray::from_option_iter([Some(1i32), Some(2)]).into_array();
        let mut probe = array.repeated_probe();
        assert!(probe.execute_is_valid(1, &mut ctx)?);
        assert_eq!(probe.uniform_validity, Some(true));
        assert!(probe.validity.is_none());
        Ok(())
    }

    #[test]
    fn repeated_state_creates_children_on_demand() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let array = struct_of_two_fields()?;
        let typed = array
            .as_opt::<Struct>()
            .ok_or_else(|| vortex_err!("expected a struct"))?;
        let mut repeated = RepeatedState::<()>::default();
        {
            let mut state = ProbeState::repeated(typed, &mut repeated);
            assert!(state.slot(5).is_err());
            assert!(state.slot(0)?.is_none());
            assert!(state.retained().is_some());
            let mut field = state.slot(2)?.ok_or_else(|| vortex_err!("missing field"))?;
            assert!(matches!(field, ArrayProbe::Repeated(_)));
            assert_eq!(field.execute_scalar(1, &mut ctx)?, Scalar::from(4i64));
            assert_eq!(field.execute_scalar(0, &mut ctx)?, Scalar::from(3i64));
        }

        assert_eq!(repeated.slots.len(), array.slots().len());
        assert!(repeated.slots[1].is_none());
        let child = repeated.slots[2]
            .as_ref()
            .ok_or_else(|| vortex_err!("missing child probe"))?;
        assert_eq!(child.array().len(), 2);
        Ok(())
    }

    fn struct_of_two_fields() -> VortexResult<ArrayRef> {
        Ok(StructArray::from_fields(&[
            ("a", PrimitiveArray::from_iter([1i32, 2]).into_array()),
            ("b", PrimitiveArray::from_iter([3i64, 4]).into_array()),
        ])?
        .into_array())
    }

    #[test]
    fn state_slot_rejects_mismatched_type() -> VortexResult<()> {
        let mut slot: Option<Box<dyn Any>> = None;
        repeated_state::<()>(&mut slot)?;
        assert!(repeated_state::<u8>(&mut slot).is_err());
        Ok(())
    }
}
