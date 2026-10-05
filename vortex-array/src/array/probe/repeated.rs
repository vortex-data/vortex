// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::probe::ArrayProbe;
use crate::array::probe::array::check_bounds;
use crate::array::probe::array::check_dtype;
use crate::array::probe::array::child_of;
use crate::arrays::BoolArray;
use crate::arrays::ScalarFn;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// A row accessor that owns its array and keeps preparation between reads.
///
/// The encoding's state, its validity probe and the probes over child slots are created on
/// first use and reused by every following read. Dropping the probe drops all of them. Array
/// handles share their buffers, so the probe can outlive the handle it was built from. Probes
/// are local to a thread.
pub struct RepeatedArrayProbe {
    array: ArrayRef,
    /// The encoding's [`RepeatedState`], type-erased, created on the first read.
    state: Option<Box<dyn Any>>,
}

impl RepeatedArrayProbe {
    /// Own an array for repeated reads.
    pub fn new(array: ArrayRef) -> Self {
        Self { array, state: None }
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
        check_bounds(&self.array, index)?;
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
        self.array
            .dyn_array()
            .probe_is_valid_retained(&self.array, index, &mut self.state, ctx)
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

/// The validity of a [`RepeatedState`], resolved on first use and kept between reads.
enum ResolvedValidity {
    Uniform(bool),
    Array(RepeatedArrayProbe),
}

impl ResolvedValidity {
    /// Whether row `index` is valid.
    #[inline]
    fn is_valid(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        match self {
            Self::Uniform(valid) => Ok(*valid),
            Self::Array(probe) => is_valid_scalar(probe.execute_scalar(index, ctx)?, index),
        }
    }

    /// The resolved validity of `array`.
    fn resolve(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        if !array.dtype().is_nullable() {
            return Ok(Self::Uniform(true));
        }
        Ok(match array.validity()? {
            Validity::NonNullable | Validity::AllValid => Self::Uniform(true),
            Validity::AllInvalid => Self::Uniform(false),
            Validity::Array(validity) => {
                // ScalarFn validity is lazy; materialize it once instead of once per row.
                // TODO(myrrc): remove this once the probing API can specify validity.
                let validity = if array.is::<ScalarFn>() {
                    validity.execute::<BoolArray>(ctx)?.into_array()
                } else {
                    validity
                };
                Self::Array(RepeatedArrayProbe::new(validity))
            }
        })
    }
}

/// The validity bit read out of a validity array's scalar.
#[inline]
pub(super) fn is_valid_scalar(scalar: Scalar, index: usize) -> VortexResult<bool> {
    scalar
        .as_bool()
        .value()
        .ok_or_else(|| vortex_err!("validity value at index {index} is null"))
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
    /// The source's validity, resolved on first use.
    validity: Option<ResolvedValidity>,
}

impl<S: Default> Default for RepeatedState<S> {
    fn default() -> Self {
        Self {
            state: S::default(),
            slots: Vec::new(),
            validity: None,
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

    /// Whether row `index` of `array` is valid, through the kept validity.
    #[inline]
    pub(crate) fn is_valid(
        &mut self,
        array: &ArrayRef,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        let validity = match &mut self.validity {
            Some(validity) => validity,
            slot @ None => slot.insert(ResolvedValidity::resolve(array, ctx)?),
        };
        validity.is_valid(index, ctx)
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

        check_reads(probe.as_probe(), &mut ctx)?;

        let state = repeated_state::<()>(&mut probe.state)?;
        assert!(matches!(state.validity, Some(ResolvedValidity::Array(_))));
        Ok(())
    }

    #[test]
    fn repeated_probe_resolves_uniform_validity_once() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let array = PrimitiveArray::from_option_iter([Some(1i32), Some(2)]).into_array();
        let mut probe = array.repeated_probe();
        assert!(probe.execute_is_valid(1, &mut ctx)?);
        let state = repeated_state::<()>(&mut probe.state)?;
        assert!(matches!(
            state.validity,
            Some(ResolvedValidity::Uniform(true))
        ));
        assert_eq!(probe.execute_scalar(1, &mut ctx)?, Scalar::from(Some(2i32)));
        Ok(())
    }

    #[test]
    fn validity_and_scalar_reads_share_the_validity_slot() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let mut probe = nullable_ints().repeated_probe();
        assert!(!probe.execute_is_valid(1, &mut ctx)?);
        let Some(ResolvedValidity::Array(validity)) =
            &repeated_state::<()>(&mut probe.state)?.validity
        else {
            return Err(vortex_err!("validity slot should hold a probe"));
        };
        assert!(validity.state.is_some());
        assert!(probe.execute_scalar(1, &mut ctx)?.is_null());
        assert_eq!(
            probe.execute_scalar(2, &mut ctx)?,
            Scalar::from(Some(30i32))
        );
        Ok(())
    }

    #[test]
    fn repeated_probe_reads_null_rows_through_encoding() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let array = PrimitiveArray::from_option_iter([Some(1i32), None]).into_array();
        let mut probe = array.repeated_probe();
        assert!(probe.execute_scalar(1, &mut ctx)?.is_null());
        assert_eq!(probe.execute_scalar(0, &mut ctx)?, Scalar::from(Some(1i32)));
        assert!(!probe.execute_is_valid(1, &mut ctx)?);
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
