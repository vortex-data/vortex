// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The fused chain the executor carries up its stack while it builds a block program.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use super::BlockOp;
use super::BlockProgram;
use super::BlockSource;
use super::fusable_ptype;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::dtype::DType;
use crate::stats::StatsSet;
use crate::trace_op;
use crate::validity::Validity;

/// A block program standing for the parent that last extended it.
///
/// The executor starts one from a leaf on the way back up the stack, pushes the op of each spine
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
    /// Start a chain from `leaf`, whose values `source` emits block by block.
    pub(crate) fn start(leaf: &ArrayRef, source: Box<dyn BlockSource>) -> VortexResult<Self> {
        let ptype = fusable_ptype(leaf)
            .ok_or_else(|| vortex_err!("a leaf is primitive, not {}", leaf.dtype()))?;
        trace_op!(record_fuse_start(leaf));
        Ok(Self {
            program: BlockProgram::new(source, ptype, leaf.len()),
            dtype: leaf.dtype().clone(),
            len: leaf.len(),
            stats: StatsSet::default(),
            validity: leaf.validity()?,
        })
    }

    /// The dtype of the values the chain produces: the dtype of the parent it stands for.
    pub(crate) fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Extend the chain with `parent`'s `op`, which transforms the chain's values into the
    /// parent's: the chain then stands for `parent`.
    pub(crate) fn push(&mut self, parent: &ArrayRef, op: Box<dyn BlockOp>) -> VortexResult<()> {
        let ptype = fusable_ptype(parent)
            .ok_or_else(|| vortex_err!("a spine parent is primitive, not {}", parent.dtype()))?;
        vortex_ensure_eq!(
            self.len,
            parent.len(),
            "a spine child has its parent's length"
        );
        self.program.push(op, ptype);
        trace_op!(record_fuse_extend(parent, self.program.ops()));
        self.dtype = parent.dtype().clone();
        self.stats = parent.statistics().to_owned();
        Ok(())
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
        Ok(values.with_stats_set(self.stats).into_array())
    }
}
