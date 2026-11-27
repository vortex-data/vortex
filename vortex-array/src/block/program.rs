// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A source and its ops, and the loop that runs them block by block.

use std::mem::MaybeUninit;
use std::ops::Range;

use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_mask::Mask;

use super::BLOCK_LEN;
use super::Block;
use super::BlockBuf;
use super::BlockIn;
use super::BlockInOut;
use super::BlockOp;
use super::BlockOut;
use super::BlockSource;
use super::BlockType;
use super::BlockValidity;
use super::SourceBlock;
use super::Values;
use super::cast;
use super::validity::OutputValidity;
use super::validity::Window;
use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::match_each_native_ptype;
use crate::trace_op;
use crate::validity::Validity;

/// A [`BlockSource`] and the [`BlockOp`]s that transform its values in turn, run block by block.
pub struct BlockProgram {
    source: Box<dyn BlockSource>,
    ops: Vec<Box<dyn BlockOp>>,
    /// The source's type, then each op's output type.
    types: Vec<BlockType>,
    len: usize,
}

impl BlockProgram {
    /// A program over the `len` values of type `ty` that `source` emits.
    pub fn new(source: Box<dyn BlockSource>, ty: impl Into<BlockType>, len: usize) -> Self {
        Self {
            source,
            ops: Vec::new(),
            types: vec![ty.into()],
            len,
        }
    }

    /// Transform the program's values with `op`, which writes values of type `ty`.
    pub fn push(&mut self, op: Box<dyn BlockOp>, ty: impl Into<BlockType>) {
        self.ops.push(op);
        self.types.push(ty.into());
    }

    /// The number of ops.
    pub fn ops(&self) -> usize {
        self.ops.len()
    }

    /// The number of rows.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the program has no rows.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The type of the values the program produces.
    pub fn block_type(&self) -> BlockType {
        *self.types.last().vortex_expect("the source's type")
    }

    /// Run the program into a new array of `nullability`, with the source's validity as the ops
    /// change it.
    ///
    /// When every op keeps the width and the source owns its buffer outright, each block is
    /// decoded where it lies and that buffer becomes the result.
    pub fn run(
        mut self,
        nullability: Nullability,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<PrimitiveArray> {
        let validity = self.source.validity().clone();
        vortex_ensure_eq!(
            validity.len(),
            self.len,
            "the source's validity covers the program's rows"
        );
        let BlockType::Primitive(ptype) = self.block_type();
        trace_op!(record_block_drive(self.ops.len(), self.len));
        let mut changed = OutputValidity::new();
        let in_place = (0..self.ops.len()).all(|i| self.in_place(i));
        if in_place && let Some(mut values) = self.source.take_buffer() {
            vortex_ensure_eq!(
                values.len(),
                self.len * self.types[0].byte_width(),
                "an owned source buffer holds the program's rows"
            );
            self.run_blocks(Target::Leaf(&mut values), &validity, &mut changed)?;
            return Ok(PrimitiveArray::from_byte_buffer(
                values.freeze(),
                ptype,
                output_validity(changed.finish(&validity), nullability)?,
            ));
        }
        match_each_native_ptype!(ptype, |T| {
            self.run_into_buffer::<T>(&validity, nullability, ctx)
        })
    }

    fn run_into_buffer<T: NativePType>(
        mut self,
        validity: &Mask,
        nullability: Nullability,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<PrimitiveArray> {
        let len = self.len;
        let mut output = BufferMut::<T>::with_capacity_in(len, ctx.allocator().clone());
        let mut changed = OutputValidity::new();
        let dst = cast::uninit_bytes_mut(&mut output.spare_capacity_mut()[..len]);
        self.run_blocks(Target::Output(dst), validity, &mut changed)?;
        // SAFETY: `run_blocks` succeeded, so the last op of every block, or the copy when there are
        // no ops, wrote that block's rows of the output.
        unsafe { output.set_len(len) };
        Ok(PrimitiveArray::new(
            output.freeze(),
            output_validity(changed.finish(validity), nullability)?,
        ))
    }

    /// Whether `ops[i]` may overwrite its input: it supports it and keeps the width.
    fn in_place(&self, i: usize) -> bool {
        self.ops[i].supports_in_place()
            && self.types[i].byte_width() == self.types[i + 1].byte_width()
    }

    /// Run the source through every op, block by block, into `target`, starting each block's
    /// validity from `validity`, the source's, and recording blocks whose validity an op edits in
    /// `changed`.
    ///
    /// Each block is the source's, in its scratch or lent, or the leaf's own memory. An op that
    /// keeps the width and allows it overwrites its block; any other op copies it into the other
    /// scratch block, and the last op writes `target`. Every block of a [`Target::Output`] is
    /// written once this returns `Ok`.
    fn run_blocks(
        &mut self,
        mut target: Target<'_>,
        validity: &Mask,
        changed: &mut OutputValidity,
    ) -> VortexResult<()> {
        let in_place: Vec<bool> = (0..self.ops.len()).map(|i| self.in_place(i)).collect();
        let Self {
            source,
            ops,
            types,
            len,
        } = self;
        let (len, last) = (*len, ops.len().checked_sub(1));
        let offset = match target {
            Target::Leaf(_) => 0,
            Target::Output(_) => source.block_offset() % BLOCK_LEN,
        };
        // The source fills the first scratch block; the second is taken when an op copies between
        // them. Both go back to this thread's spares once every block is written.
        let mut scratch: [Option<Box<BlockBuf>>; 2] = [None, None];
        let mut start = 0;
        while start < len {
            let rows = block_rows(start, len, offset);
            let n = rows.len();
            // Where the block's values are, and the index of its first value there.
            let (mut at, mut first, mut lent) = (At::Leaf, start, None);
            let mut window: Window = None;
            if let Target::Output(_) = target {
                let buf = scratch[0].get_or_insert_with(BlockBuf::take);
                let mut block_validity = BlockValidity::new(validity, start, n, &mut window);
                match source.fill(rows.clone(), buf, &mut block_validity)? {
                    SourceBlock::Scratch {
                        offset,
                        len: filled,
                    } => {
                        vortex_ensure_eq!(
                            filled,
                            n,
                            "block source emitted the wrong number of rows"
                        );
                        vortex_ensure!(
                            offset + filled <= BLOCK_LEN,
                            "block source overran its scratch"
                        );
                        (at, first) = (At::Scratch(0), offset);
                    }
                    SourceBlock::Lent(block) => {
                        vortex_ensure_eq!(
                            block.len,
                            n,
                            "block source emitted the wrong number of rows"
                        );
                        vortex_ensure_eq!(block.ty, types[0]);
                        (at, first, lent) = (At::Lent, 0, Some(block));
                    }
                }
            }
            for (i, op) in ops.iter_mut().enumerate() {
                let (in_type, out_type) = (types[i], types[i + 1]);
                let input_bytes = first * in_type.byte_width()..(first + n) * in_type.byte_width();
                let from = at;
                // The scratch block the values are in, and the other one.
                let [scratch0, scratch1] = &mut scratch;
                let (here, other) = match at {
                    At::Scratch(0) => (scratch0, scratch1),
                    // From scratch1 or the source, a copy goes into scratch0.
                    _ => (scratch1, scratch0),
                };
                let to_output = Some(i) == last && matches!(target, Target::Output(_));
                let values = if in_place[i] && at.writable() && !to_output {
                    let bytes = match (&mut target, at) {
                        (Target::Leaf(leaf), At::Leaf) => &mut leaf[input_bytes],
                        (_, At::Scratch(_)) => &mut scratch_bytes(here)?[input_bytes],
                        _ => vortex_bail!("a block is overwritten only in scratch or the leaf"),
                    };
                    Values::InPlace(BlockInOut {
                        bytes,
                        len: n,
                        in_type,
                        out_type,
                    })
                } else {
                    let input = match at {
                        At::Lent => lent.vortex_expect("a lent block").bytes,
                        At::Scratch(_) => &scratch_bytes(here)?[input_bytes],
                        At::Leaf | At::Output => {
                            vortex_bail!("a block is copied only from the source or scratch")
                        }
                    };
                    let width = out_type.byte_width();
                    let out = match &mut target {
                        Target::Output(dst) if to_output => {
                            at = At::Output;
                            &mut dst[start * width..(start + n) * width]
                        }
                        _ => {
                            let to = At::Scratch(usize::from(at == At::Scratch(0)));
                            (at, first) = (to, 0);
                            let bytes = &mut other.get_or_insert_with(BlockBuf::take).0;
                            // SAFETY: a `BlockOut` writes only initialized values.
                            unsafe { cast::as_uninit_mut(&mut bytes[..n * width]) }
                        }
                    };
                    Values::Copy {
                        input: BlockIn {
                            bytes: input,
                            len: n,
                            ty: in_type,
                        },
                        out: BlockOut {
                            bytes: out,
                            len: n,
                            ty: out_type,
                            filled: 0,
                        },
                    }
                };
                if start == 0 {
                    let shape = match values {
                        Values::InPlace(_) => "in_place",
                        Values::Copy { .. } => "copy",
                    };
                    trace_op!(record_block_op(
                        i,
                        shape,
                        from.name(),
                        at.name(),
                        in_type,
                        out_type
                    ));
                }
                let block = Block {
                    values,
                    validity: BlockValidity::new(validity, start, n, &mut window),
                    start,
                };
                let written = op.process(block)?;
                vortex_ensure_eq!(written.len, n, "block op wrote the wrong number of rows");
                vortex_ensure_eq!(written.ty, out_type, "block op wrote the wrong type");
            }
            if let (None, Target::Output(dst)) = (last, &mut target) {
                // With no ops the source's values are the output.
                let width = types[0].byte_width();
                let input = match at {
                    At::Lent => lent.vortex_expect("a lent block").bytes,
                    _ => &scratch_bytes(&mut scratch[0])?[first * width..(first + n) * width],
                };
                dst[start * width..(start + n) * width].write_copy_of_slice(input);
            }
            changed.push(validity, start, n, &window);
            start = rows.end;
        }
        scratch.into_iter().flatten().for_each(BlockBuf::give_back);
        Ok(())
    }
}

/// The bytes of a scratch block the program allocated.
fn scratch_bytes(scratch: &mut Option<Box<BlockBuf>>) -> VortexResult<&mut [u8]> {
    match scratch {
        Some(buf) => Ok(&mut buf.0),
        None => vortex_bail!("a block is read from scratch the program never allocated"),
    }
}

/// The output's validity from its mask, which a non-nullable output must have no nulls in.
fn output_validity(mask: Mask, nullability: Nullability) -> VortexResult<Validity> {
    if nullability == Nullability::NonNullable {
        vortex_ensure!(
            mask.all_true(),
            "a non-nullable block program produced null rows"
        );
        return Ok(Validity::NonNullable);
    }
    Ok(Validity::from_mask(mask, nullability))
}

/// Where a program writes its values.
enum Target<'t> {
    /// Room for every value of the output type.
    Output(&'t mut [MaybeUninit<u8>]),
    /// The source's own values, which every op overwrites in place.
    Leaf(&'t mut [u8]),
}

/// The memory a block's values are in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum At {
    /// Lent by the source, so only read.
    Lent,
    Scratch(usize),
    Leaf,
    Output,
}

impl At {
    fn writable(self) -> bool {
        matches!(self, At::Scratch(_) | At::Leaf)
    }

    /// How a trace names the memory.
    fn name(self) -> &'static str {
        match self {
            At::Lent => "source",
            At::Scratch(0) => "scratch0",
            At::Scratch(_) => "scratch1",
            At::Leaf => "leaf",
            At::Output => "output",
        }
    }
}

/// The rows of the block starting at `start`: up to the next multiple of [`BLOCK_LEN`] past
/// `offset`, so every block lines up with the source's own blocks.
fn block_rows(start: usize, len: usize, offset: usize) -> Range<usize> {
    let to_boundary = BLOCK_LEN - (offset + start) % BLOCK_LEN;
    start..(start + to_boundary).min(len)
}
