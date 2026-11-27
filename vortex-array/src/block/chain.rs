// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The fused chain the executor carries up its stack while it builds a block program.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use super::BlockOp;
use super::BlockProgram;
use super::BlockSource;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ExtensionArray;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::stats::StatsSet;
use crate::trace_op;

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
        })
    }

    /// The dtype of the values the chain produces: the dtype of the parent it stands for.
    pub(crate) fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Extend the chain with `parent`'s `op`, which transforms the chain's values into the
    /// parent's: the chain then stands for `parent`.
    pub(crate) fn push(
        &mut self,
        parent: &ArrayRef,
        op: Box<dyn BlockOp>,
        absorb: bool,
    ) -> VortexResult<()> {
        let ptype = fusable_ptype(parent)
            .ok_or_else(|| vortex_err!("a spine parent is primitive, not {}", parent.dtype()))?;
        vortex_ensure_eq!(
            self.len,
            parent.len(),
            "a spine child has its parent's length"
        );
        let absorbed = absorb && self.program.try_absorb(op.as_ref(), ptype.into());
        if !absorbed {
            self.program.push(op, ptype);
        }
        trace_op!(record_fuse_extend(parent, self.program.ops(), absorbed));
        self.dtype = parent.dtype().clone();
        self.stats = parent.statistics().to_owned();
        Ok(())
    }

    /// Run the program: the values of the parent the chain stands for.
    pub(crate) fn drive(self, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let values = self.program.run(self.dtype.nullability(), ctx)?;
        Ok(match &self.dtype {
            DType::Extension(ext) => ExtensionArray::try_new(ext.clone(), values.into_array())?
                .with_stats_set(self.stats)
                .into_array(),
            _ => values.with_stats_set(self.stats).into_array(),
        })
    }
}

/// The primitive type of an array the executor can fuse: its own, or its storage type for an
/// extension type stored as a primitive. `None` for any other array.
pub(crate) fn fusable_ptype(array: &ArrayRef) -> Option<PType> {
    match array.dtype() {
        DType::Primitive(ptype, _) => Some(*ptype),
        DType::Extension(ext) => match ext.storage_dtype() {
            DType::Primitive(ptype, _) => Some(*ptype),
            _ => None,
        },
        _ => None,
    }
}
