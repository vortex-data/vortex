// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`Fused`] array, which carries a block program up the executor's stack.

use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use parking_lot::Mutex;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::registry::CachedId;

use super::BlockProgram;
use super::ProgramInput;
use super::fusable_ptype;
use crate::AnyCanonical;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::array::vtable::NotSupported;
use crate::array::vtable::ValidityVTable;
use crate::array::vtable::with_empty_buffers;
use crate::arrays::Primitive;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::matcher::Matcher;
use crate::trace_op;
use crate::validity::Validity;

/// An array whose values a block program produces: the parent it stands for, with the chain below
/// it fused into the program.
///
/// The executor builds it on the way back up the stack, one parent at a time, and drives it when
/// the next parent cannot extend it, when its `execute` runs the whole program in one pass. It
/// never leaves `execute_until`.
#[derive(Clone, Debug)]
pub(crate) struct Fused;

#[derive(Clone)]
pub(crate) struct FusedData {
    state: Arc<FusedState>,
}

struct FusedState {
    /// Taken by the parent that extends it, or by `execute`.
    program: Mutex<Option<BlockProgram>>,
    validity: Validity,
}

impl Debug for FusedData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FusedData").finish_non_exhaustive()
    }
}

impl Display for FusedData {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

impl ArrayHash for FusedData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        Arc::as_ptr(&self.state).hash(state);
    }
}

impl ArrayEq for FusedData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
    }
}

impl ValidityVTable<Fused> for Fused {
    fn validity(array: ArrayView<'_, Fused>) -> VortexResult<Validity> {
        Ok(array.data().state.validity.clone())
    }
}

impl VTable for Fused {
    type TypedArrayData = FusedData;
    type OperationsVTable = NotSupported;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.fused");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        _len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(slots.is_empty(), "a fused array has no slots");
        vortex_ensure!(
            matches!(dtype, DType::Primitive(..)),
            "a fused array is primitive, not {dtype}"
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("Fused buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        vortex_panic!("Fused slot index {idx} out of bounds")
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let state = &array.as_view().data().state;
        let program = take_program(state)?;
        trace_op!(record_block_drive(
            program.input_name(),
            program.ops(),
            array.len()
        ));
        Ok(ExecutionResult::done(
            program.run(state.validity.clone(), ctx)?.into_array(),
        ))
    }
}

fn take_program(state: &FusedState) -> VortexResult<BlockProgram> {
    state
        .program
        .lock()
        .take()
        .ok_or_else(|| vortex_err!("fused array was already driven or extended"))
}

/// Fuse `child`, the array in slot `slot_idx` of the suspended `parent`, into a [`Fused`] array
/// standing for `parent`, if `parent` transforms blocks of that slot and `child` can be the input
/// of a block program. Returns `child` back otherwise.
///
/// `child` is a [`Fused`] array, whose program is extended, a canonical primitive array, an array
/// with a block source, or an array whose own op transforms a canonical primitive child.
pub(crate) fn fuse_into_parent(
    child: ArrayRef,
    parent: &ArrayRef,
    slot_idx: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Result<ArrayRef, ArrayRef>> {
    let Some(parent_ptype) = fusable_ptype(parent) else {
        return Ok(Err(child));
    };
    if child.is_empty() {
        return Ok(Err(child));
    }
    let Some(op) = parent.block_op(slot_idx, child.dtype(), ctx)? else {
        return Ok(Err(child));
    };
    let Some(mut program) = program_for(&child, ctx)? else {
        return Ok(Err(child));
    };
    program.push(op, parent_ptype);

    // The parent's validity and statistics, from the parent with its child back in its slot.
    // SAFETY: `child` is the slot's array, or a fused array of its dtype and length.
    let whole = unsafe { parent.clone().put_slot_unchecked(slot_idx, child) }?;
    let validity = whole.validity()?;
    let stats = whole.statistics().to_owned();
    trace_op!(record_fuse_extend(&whole, program.ops()));
    // Release the parent and child, so a canonical input is again uniquely owned.
    drop(whole);

    let fused = Array::<Fused>::try_from_parts(ArrayParts::new(
        Fused,
        parent.dtype().clone(),
        parent.len(),
        FusedData {
            state: Arc::new(FusedState {
                program: Mutex::new(Some(program)),
                validity,
            }),
        },
        Default::default(),
    ))?
    .with_stats_set(stats)
    .into_array();
    Ok(Ok(fused))
}

/// The program producing `child`'s values, if `child` can be the input of one.
fn program_for(child: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Option<BlockProgram>> {
    if let Some(fused) = child.as_opt::<Fused>() {
        return take_program(&fused.data().state).map(Some);
    }
    let Some(ptype) = fusable_ptype(child) else {
        return Ok(None);
    };
    if let Some(primitive) = child.as_opt::<Primitive>() {
        trace_op!(record_fuse_start("canonical", child));
        return Ok(Some(BlockProgram::new(
            ProgramInput::Canonical(primitive.into_owned()),
            ptype,
            child.len(),
        )));
    }
    if let Some(source) = child.block_source(ctx)? {
        trace_op!(record_fuse_start("source", child));
        return Ok(Some(BlockProgram::new(
            ProgramInput::Source(source),
            ptype,
            child.len(),
        )));
    }
    // A child whose own op transforms a canonical primitive child of its own, which it would
    // execute right away, so the executor never descends to it.
    for (slot_idx, grandchild) in child.slots().iter().enumerate() {
        let Some(grandchild) = grandchild else {
            continue;
        };
        let (Some(grandchild_ptype), Some(primitive)) =
            (fusable_ptype(grandchild), grandchild.as_opt::<Primitive>())
        else {
            continue;
        };
        if grandchild.len() != child.len() || !AnyCanonical::matches(grandchild) {
            continue;
        }
        let Some(op) = child.block_op(slot_idx, grandchild.dtype(), ctx)? else {
            continue;
        };
        trace_op!(record_fuse_start("canonical", grandchild));
        let mut program = BlockProgram::new(
            ProgramInput::Canonical(primitive.into_owned()),
            grandchild_ptype,
            grandchild.len(),
        );
        program.push(op, ptype);
        trace_op!(record_fuse_extend(child, program.ops()));
        return Ok(Some(program));
    }
    Ok(None)
}
