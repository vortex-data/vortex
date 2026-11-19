// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The fused chain the executor carries up its stack while it builds a block program.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use super::BlockCombineSpec;
use super::BlockProgram;
use super::BlockSource;
use super::ProgramInput;
use super::ZipSource;
use super::fusable_ptype;
use super::program::PrimitiveSource;
use crate::AnyCanonical;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ExtensionArray;
use crate::arrays::Primitive;
use crate::builders::ArrayBuilder;
use crate::builders::PrimitiveBuilder;
use crate::dtype::DType;
use crate::match_each_native_ptype;
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

    /// Combine the children of `array`, which has not executed and whose combine is `spec`, into
    /// a chain standing for `array`, if each child the combine reads is canonical or a block
    /// source that `uses_source` allows, and at least one is a source: combining only canonical
    /// children saves no buffer. Returns `array` back otherwise.
    pub(crate) fn combine(
        array: ArrayRef,
        spec: BlockCombineSpec,
        uses_source: impl Fn(&ArrayRef) -> bool,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Result<Self, ArrayRef>> {
        let Some(ptype) = fusable_ptype(&array) else {
            return Ok(Err(array));
        };
        if array.is_empty() {
            return Ok(Err(array));
        }
        // Every child is checked before any source is built, since building one can execute
        // patches.
        let mut children = Vec::with_capacity(spec.slots.len());
        for &slot in &spec.slots {
            let Some(child) = array.slots().get(slot).and_then(Option::as_ref) else {
                return Ok(Err(array));
            };
            let Some(child_ptype) = fusable_ptype(child) else {
                return Ok(Err(array));
            };
            let canonical = child.is::<Primitive>();
            if child.len() != array.len() || !(canonical || uses_source(child)) {
                return Ok(Err(array));
            }
            children.push((child, child_ptype, canonical));
        }
        if children.iter().all(|&(_, _, canonical)| canonical) {
            return Ok(Err(array));
        }
        let mut inputs = Vec::with_capacity(children.len());
        for (child, child_ptype, _) in children {
            let source = if let Some(primitive) = child.as_opt::<Primitive>() {
                match_each_native_ptype!(child_ptype, |T| {
                    Box::new(PrimitiveSource::new(primitive.to_buffer::<T>()))
                        as Box<dyn BlockSource>
                })
            } else if let Some(source) = child.block_source(ctx)? {
                source
            } else {
                // The child is not ready: the array executes, and is combined again once it is.
                return Ok(Err(array));
            };
            inputs.push((source, child_ptype));
        }
        trace_op!(record_fuse_start("zip", &array));
        Ok(Ok(Self {
            program: BlockProgram::new(
                ProgramInput::Zip(ZipSource::new(inputs, spec.combine, ptype)),
                ptype,
                array.len(),
            ),
            dtype: array.dtype().clone(),
            len: array.len(),
            validity: array.validity()?,
            stats: array.statistics().to_owned(),
        }))
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
                self.program.input_name(),
                self.program.ops(),
                self.len
            ));
            let mask = self.validity.execute_mask(self.len, ctx)?;
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
