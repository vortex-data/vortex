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
use crate::arrays::ExtensionArray;
use crate::builders::ArrayBuilder;
use crate::builders::PrimitiveBuilder;
use crate::dtype::DType;
use crate::match_each_native_ptype;
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
        Ok(match &self.dtype {
            DType::Extension(ext) => ExtensionArray::try_new(ext.clone(), values.into_array())?
                .with_stats_set(self.stats)
                .into_array(),
            _ => values.with_stats_set(self.stats).into_array(),
        })
    }

    /// Drive the program straight into `builder`, appending the values of the parent the chain
    /// stands for without a buffer of their own, if `builder` is a primitive builder of the
    /// chain's dtype. Returns the chain back otherwise.
    pub(crate) fn drive_into(
        self,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Result<(), Self>> {
        if builder.dtype() != &self.dtype {
            return Ok(Err(self));
        }
        let DType::Primitive(ptype, _) = self.dtype else {
            return Ok(Err(self));
        };
        match_each_native_ptype!(ptype, |T| {
            let Some(builder) = builder.as_any_mut().downcast_mut::<PrimitiveBuilder<T>>() else {
                return Ok(Err(self));
            };
            trace_op!(record_block_drive_into_builder(
                self.program.ops(),
                self.len
            ));
            let mask = self
                .validity
                .union_nullability(self.dtype.nullability())
                .execute_mask(self.len, ctx)?;
            builder.reserve_exact(self.len);
            let mut range = builder.uninit_range(self.len);
            // SAFETY: the range holds `len` slots, which `write_into` initializes before
            // `finish`, and the mask describes those `len` rows.
            unsafe {
                self.program
                    .write_into(range.slice_uninit_mut(0, self.len))?;
                range.append_mask(&mask);
                range.finish();
            }
            Ok(Ok(()))
        })
    }
}
