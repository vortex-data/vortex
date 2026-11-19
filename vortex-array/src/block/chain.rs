// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The fused chain the executor carries up its stack while it builds a block program.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use super::BlockProgram;
use super::ProgramInput;
use super::fusable_ptype;
use crate::AnyCanonical;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Primitive;
use crate::dtype::DType;
use crate::matcher::Matcher;
use crate::stats::StatsSet;
use crate::trace_op;
use crate::validity::Validity;

/// A block program standing for the parent that last extended it.
///
/// The executor starts one from a spine child on the way back up the stack, extends it with each
/// spine parent above that transforms blocks, and drives it as soon as the next parent cannot, so
/// a chain never waits: it is a value of the executor's loop, not an array.
pub(crate) struct FusedChain {
    program: BlockProgram,
    /// The parent the chain stands for: the dtype, length, validity and statistics of its values.
    dtype: DType,
    len: usize,
    validity: Validity,
    stats: StatsSet,
}

impl FusedChain {
    /// Start a chain from `child`, the array in slot `slot_idx` of the suspended `parent`, whose
    /// validity is `validity`, if `parent` transforms blocks of that slot and `child` can be the
    /// input of a block program. Returns `child` back otherwise.
    ///
    /// `child` is a canonical primitive array, an array with a block source, or an array whose own
    /// op transforms a canonical primitive child. The executor checked on the way down that both
    /// arrays are primitive, non-empty and of one length.
    pub(crate) fn start(
        child: ArrayRef,
        parent: &ArrayRef,
        slot_idx: usize,
        validity: &Validity,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Result<Self, ArrayRef>> {
        vortex_ensure_eq!(
            child.len(),
            parent.len(),
            "a spine child has its parent's length"
        );
        // The input first, so a child that cannot start a program costs no op.
        let Some(program) = program_for(&child, ctx)? else {
            return Ok(Err(child));
        };
        let chain = Self {
            program,
            dtype: child.dtype().clone(),
            len: child.len(),
            validity: Validity::NonNullable,
            stats: StatsSet::default(),
        };
        match chain.extend(parent, slot_idx, validity, ctx)? {
            // Release the child, so a canonical input is again uniquely owned.
            Ok(chain) => Ok(Ok(chain)),
            Err(_) => Ok(Err(child)),
        }
    }

    /// Extend the chain with `parent`, which waits for its values in slot `slot_idx` and whose
    /// validity is `validity`. Returns the chain back if `parent` does not transform blocks of
    /// that slot.
    pub(crate) fn extend(
        mut self,
        parent: &ArrayRef,
        slot_idx: usize,
        validity: &Validity,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Result<Self, Self>> {
        let Some(ptype) = fusable_ptype(parent) else {
            return Ok(Err(self));
        };
        vortex_ensure_eq!(
            self.len,
            parent.len(),
            "a spine child has its parent's length"
        );
        let Some(op) = parent.block_op(slot_idx, &self.dtype, ctx)? else {
            return Ok(Err(self));
        };
        self.program.push(op, ptype);
        trace_op!(record_fuse_extend(parent, self.program.ops()));
        self.dtype = parent.dtype().clone();
        self.validity = validity.clone();
        self.stats = parent.statistics().to_owned();
        Ok(Ok(self))
    }

    /// Run the program: the values of the parent the chain stands for.
    pub(crate) fn drive(self, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        trace_op!(record_block_drive(
            self.program.input_name(),
            self.program.ops(),
            self.len
        ));
        // Level-wise execution materializes validity, so a fused result does too.
        let validity = self.validity.execute(ctx)?;
        let values = self.program.run(validity, ctx)?;
        Ok(values.with_stats_set(self.stats).into_array())
    }
}

/// The program producing `child`'s values, if `child` can be the input of one.
fn program_for(child: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Option<BlockProgram>> {
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
