// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A source and its ops, and the loop that runs them block by block.

use std::mem::MaybeUninit;
use std::ops::Range;

use vortex_buffer::BitBufferMut;
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
use super::BlockValidity;
use super::SourceBlock;
use super::Values;
use super::cast;
use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::trace_op;
use crate::validity::Validity;

/// A [`BlockSource`] and the [`BlockOp`]s that transform its values in turn, run block by block.
pub struct BlockProgram {
    source: Box<dyn BlockSource>,
    ops: Vec<Box<dyn BlockOp>>,
    /// The source's type, then each op's output type.
    ptypes: Vec<PType>,
    len: usize,
}

impl BlockProgram {
    /// A program over the `len` values of `ptype` that `source` emits.
    pub fn new(source: Box<dyn BlockSource>, ptype: PType, len: usize) -> Self {
        Self {
            source,
            ops: Vec::new(),
            ptypes: vec![ptype],
            len,
        }
    }

    /// Transform the program's values with `op`, which writes values of `ptype`.
    pub fn push(&mut self, op: Box<dyn BlockOp>, ptype: PType) {
        self.ops.push(op);
        self.ptypes.push(ptype);
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
    pub fn ptype(&self) -> PType {
        *self.ptypes.last().vortex_expect("the source's type")
    }

    /// Run the program into a new array of `nullability`, whose rows start with `validity`.
    ///
    /// When every op keeps the width and the source owns its buffer outright, each block is
    /// decoded where it lies and that buffer becomes the result.
    pub fn run(
        mut self,
        validity: &Mask,
        nullability: Nullability,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<PrimitiveArray> {
        vortex_ensure_eq!(
            validity.len(),
            self.len,
            "validity covers the program's rows"
        );
        trace_op!(record_block_drive(self.ops.len(), self.len));
        let mut changed = None;
        let in_place = (0..self.ops.len()).all(|i| self.in_place(i));
        if in_place && let Some(mut values) = self.source.take_buffer() {
            vortex_ensure_eq!(
                values.len(),
                self.len * self.ptypes[0].byte_width(),
                "an owned source buffer holds the program's rows"
            );
            self.run_blocks(Target::Leaf(&mut values), validity, &mut changed)?;
            return Ok(PrimitiveArray::from_byte_buffer(
                values.freeze(),
                self.ptype(),
                output_validity(validity, changed, nullability)?,
            ));
        }
        let ptype = self.ptype();
        match_each_native_ptype!(ptype, |T| {
            self.run_into_buffer::<T>(validity, nullability, ctx)
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
        let mut changed = None;
        let dst = cast::uninit_bytes_mut(&mut output.spare_capacity_mut()[..len]);
        self.run_blocks(Target::Output(dst), validity, &mut changed)?;
        // SAFETY: `run_blocks` succeeded, so the last op of every block, or the copy when there are
        // no ops, wrote that block's rows of the output.
        unsafe { output.set_len(len) };
        Ok(PrimitiveArray::new(
            output.freeze(),
            output_validity(validity, changed, nullability)?,
        ))
    }

    /// Whether `ops[i]` may overwrite its input: it supports it and keeps the width.
    fn in_place(&self, i: usize) -> bool {
        self.ops[i].supports_in_place()
            && self.ptypes[i].byte_width() == self.ptypes[i + 1].byte_width()
    }

    /// Run the source through every op, block by block, into `target`, reading each block's
    /// validity from `validity` and recording rows whose validity changes in `changed`.
    ///
    /// Each block is the source's, in its scratch or lent, or the leaf's own memory. An op that
    /// keeps the width and allows it overwrites its block; any other op copies it into the other
    /// scratch block, and the last op writes `target`. Every block of a [`Target::Output`] is
    /// written once this returns `Ok`.
    fn run_blocks(
        &mut self,
        mut target: Target<'_>,
        validity: &Mask,
        changed: &mut Option<BitBufferMut>,
    ) -> VortexResult<()> {
        let in_place: Vec<bool> = (0..self.ops.len()).map(|i| self.in_place(i)).collect();
        let Self {
            source,
            ops,
            ptypes,
            len,
        } = self;
        let (len, last) = (*len, ops.len().checked_sub(1));
        let offset = match target {
            Target::Leaf(_) => 0,
            Target::Output(_) => source.block_offset() % BLOCK_LEN,
        };
        // The source fills the first scratch block; the second is allocated when an op copies
        // between them.
        let mut scratch: [Option<Box<BlockBuf>>; 2] = [None, None];
        let mut start = 0;
        while start < len {
            let rows = block_rows(start, len, offset);
            let n = rows.len();
            // Where the block's values are, and the index of its first value there.
            let (mut at, mut first, mut lent) = (At::Leaf, start, None);
            if let Target::Output(_) = target {
                let buf = scratch[0].get_or_insert_with(BlockBuf::new_boxed);
                let mut block_validity = BlockValidity::new(start, n, validity, changed);
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
                            block.len(),
                            n,
                            "block source emitted the wrong number of rows"
                        );
                        vortex_ensure_eq!(block.ptype(), ptypes[0]);
                        (at, first, lent) = (At::Lent, 0, Some(block));
                    }
                }
            }
            for (i, op) in ops.iter_mut().enumerate() {
                let (in_ptype, out_ptype) = (ptypes[i], ptypes[i + 1]);
                let input_bytes =
                    first * in_ptype.byte_width()..(first + n) * in_ptype.byte_width();
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
                        in_ptype,
                        out_ptype,
                    })
                } else {
                    let input = match at {
                        At::Lent => lent.vortex_expect("a lent block").bytes,
                        At::Scratch(_) => &scratch_bytes(here)?[input_bytes],
                        At::Leaf | At::Output => {
                            vortex_bail!("a block is copied only from the source or scratch")
                        }
                    };
                    let width = out_ptype.byte_width();
                    let out = match &mut target {
                        Target::Output(dst) if to_output => {
                            at = At::Output;
                            &mut dst[start * width..(start + n) * width]
                        }
                        _ => {
                            let to = At::Scratch(usize::from(at == At::Scratch(0)));
                            (at, first) = (to, 0);
                            let bytes = &mut other.get_or_insert_with(BlockBuf::new_boxed).0;
                            // SAFETY: a `BlockOut` writes only initialized values.
                            unsafe { cast::as_uninit_mut(&mut bytes[..n * width]) }
                        }
                    };
                    Values::Copy {
                        input: BlockIn {
                            bytes: input,
                            len: n,
                            ptype: in_ptype,
                        },
                        out: BlockOut {
                            bytes: out,
                            len: n,
                            ptype: out_ptype,
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
                        in_ptype,
                        out_ptype
                    ));
                }
                let block = Block {
                    values,
                    validity: BlockValidity::new(start, n, validity, changed),
                };
                let written = op.process(start, block)?;
                vortex_ensure_eq!(written.len, n, "block op wrote the wrong number of rows");
                vortex_ensure_eq!(written.ptype, out_ptype, "block op wrote the wrong type");
            }
            if let (None, Target::Output(dst)) = (last, &mut target) {
                // With no ops the source's values are the output.
                let width = ptypes[0].byte_width();
                let input = match at {
                    At::Lent => lent.vortex_expect("a lent block").bytes,
                    _ => &scratch_bytes(&mut scratch[0])?[first * width..(first + n) * width],
                };
                dst[start * width..(start + n) * width].write_copy_of_slice(input);
            }
            start = rows.end;
        }
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

/// The output's validity: the input mask, unless an op changed a row.
fn output_validity(
    input: &Mask,
    changed: Option<BitBufferMut>,
    nullability: Nullability,
) -> VortexResult<Validity> {
    let changed = changed.map(BitBufferMut::freeze);
    if nullability == Nullability::NonNullable {
        let all_valid = match &changed {
            Some(bits) => bits.true_count() == bits.len(),
            None => input.all_true(),
        };
        vortex_ensure!(all_valid, "a non-nullable block program produced null rows");
        return Ok(Validity::NonNullable);
    }
    Ok(match changed {
        Some(bits) => Validity::from(bits),
        None => Validity::from_mask(input.clone(), nullability),
    })
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
