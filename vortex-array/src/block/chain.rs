// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The fused chain the executor carries up its stack while it builds a block program.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use super::BlockProgram;
use super::fusable_ptype;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ExtensionArray;
use crate::dtype::DType;
use crate::stats::StatsSet;
use crate::trace_op;
use crate::validity::Validity;

/// A block program standing for the parent that last extended it.
///
/// The executor starts one from a leaf on the way back up the stack, extends it with each spine
/// parent above that transforms blocks, and drives it as soon as the next parent cannot, so a
/// chain never waits: it is a value of the executor's loop, not an array.
pub(crate) struct FusedChain {
    program: BlockProgram,
    /// The dtype, length and statistics of the parent the chain stands for.
    dtype: DType,
    len: usize,
    stats: StatsSet,
    /// The leaf's validity, which a spine parent passes through as its own.
    validity: Validity,
}

impl FusedChain {
    /// Start a chain from `child`, the array in slot `slot_idx` of the suspended `parent`, if
    /// `child` is a leaf, one with a block source, and `parent` transforms blocks of that slot.
    /// Returns `child` back otherwise.
    ///
    /// The executor checked on the way down that both arrays are primitive, non-empty and of one
    /// length.
    pub(crate) fn start(
        child: ArrayRef,
        parent: &ArrayRef,
        slot_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Result<Self, ArrayRef>> {
        vortex_ensure_eq!(
            child.len(),
            parent.len(),
            "a spine child has its parent's length"
        );
        let (Some(ptype), Some(source)) = (fusable_ptype(&child), child.block_source(ctx)?) else {
            return Ok(Err(child));
        };
        trace_op!(record_fuse_start(&child));
        let chain = Self {
            program: BlockProgram::new(source, ptype, child.len()),
            dtype: child.dtype().clone(),
            len: child.len(),
            stats: StatsSet::default(),
            validity: child.validity()?,
        };
        match chain.extend(parent, slot_idx, ctx)? {
            // Release the child, so a leaf that owns its buffer is again its only owner.
            Ok(chain) => Ok(Ok(chain)),
            Err(_) => Ok(Err(child)),
        }
    }

    /// Extend the chain with `parent`, which waits for its values in slot `slot_idx`. Returns the
    /// chain back if `parent` does not transform blocks of that slot.
    pub(crate) fn extend(
        mut self,
        parent: &ArrayRef,
        slot_idx: usize,
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
        self.stats = parent.statistics().to_owned();
        Ok(Ok(self))
    }

    /// Run the program: the values of the parent the chain stands for.
    pub(crate) fn drive(self, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        trace_op!(record_block_drive(self.program.ops(), self.len));
        // Level-wise execution materializes validity, so a fused result does too.
        let validity = self
            .validity
            .union_nullability(self.dtype.nullability())
            .execute(ctx)?;
        let values = self.program.run(validity, ctx)?;
        Ok(match &self.dtype {
            DType::Extension(ext) => ExtensionArray::try_new(ext.clone(), values.into_array())?
                .with_stats_set(self.stats)
                .into_array(),
            _ => values.with_stats_set(self.stats).into_array(),
        })
    }
}
