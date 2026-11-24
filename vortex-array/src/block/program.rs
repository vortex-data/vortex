// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A source and its ops, and the loop that runs them block by block.

use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::Range;

use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
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
use super::VALIDITY_BYTES;
use super::ValidityState;
use super::Values;
use super::Written;
use super::bits::copy_bits;
use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::builders::ArrayBuilder;
use crate::builders::PrimitiveBuilder;
use crate::dtype::DType;
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
        trace_op!(record_block_drive(self.ops.len(), self.len, false));
        let mut out_validity = OutputValidity::new(self.len);
        let in_place = (0..self.ops.len()).all(|i| self.in_place(i));
        if in_place && let Some(mut values) = self.source.take_buffer() {
            vortex_ensure_eq!(
                values.len(),
                self.len * self.ptypes[0].byte_width(),
                "an owned source buffer holds the program's rows"
            );
            // SAFETY: `values` holds `len` values of the source's type, and every op keeps the
            // width, so each block is overwritten where it lies.
            unsafe {
                self.run_blocks(
                    Target::Leaf(values.as_mut_ptr()),
                    validity,
                    &mut out_validity,
                )?
            };
            return Ok(PrimitiveArray::from_byte_buffer(
                values.freeze(),
                self.ptype(),
                out_validity.into_validity(nullability)?,
            ));
        }
        let ptype = self.ptype();
        match_each_native_ptype!(ptype, |T| {
            self.run_into_buffer::<T>(validity, nullability, out_validity, ctx)
        })
    }

    /// Run the program, appending its values and validity to `builder`, a primitive builder of the
    /// program's type.
    pub fn append_to(
        self,
        validity: &Mask,
        builder: &mut dyn ArrayBuilder,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let ptype = self.ptype();
        let DType::Primitive(builder_ptype, nullability) = builder.dtype() else {
            vortex_bail!(
                "a block program appends to a primitive builder, not {}",
                builder.dtype()
            );
        };
        vortex_ensure_eq!(
            *builder_ptype,
            ptype,
            "a block program appends its own type"
        );
        let nullability = *nullability;
        match_each_native_ptype!(ptype, |T| {
            let builder = builder
                .as_any_mut()
                .downcast_mut::<PrimitiveBuilder<T>>()
                .ok_or_else(|| vortex_err!("a primitive builder of {ptype}"))?;
            self.append_to_builder(validity, nullability, builder)
        })
    }

    fn run_into_buffer<T: NativePType>(
        mut self,
        validity: &Mask,
        nullability: Nullability,
        mut out_validity: OutputValidity,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<PrimitiveArray> {
        let len = self.len;
        let mut output = BufferMut::<T>::with_capacity_in(len, ctx.allocator().clone());
        let dst = output.spare_capacity_mut()[..len].as_mut_ptr();
        // SAFETY: `dst` has room for `len` values of `T`, the program's output type.
        unsafe { self.run_blocks(Target::Output(dst.cast()), validity, &mut out_validity)? };
        // SAFETY: `run_blocks` wrote every one of the `len` values.
        unsafe { output.set_len(len) };
        Ok(PrimitiveArray::new(
            output.freeze(),
            out_validity.into_validity(nullability)?,
        ))
    }

    fn append_to_builder<T: NativePType>(
        mut self,
        validity: &Mask,
        nullability: Nullability,
        builder: &mut PrimitiveBuilder<T>,
    ) -> VortexResult<()> {
        vortex_ensure_eq!(
            validity.len(),
            self.len,
            "validity covers the program's rows"
        );
        if self.len == 0 {
            return Ok(());
        }
        trace_op!(record_block_drive(self.ops.len(), self.len, true));
        let len = self.len;
        let mut out_validity = OutputValidity::new(len);
        builder.reserve_exact(len);
        let mut range = builder.uninit_range(len);
        // SAFETY: the range holds `len` slots, which `run_blocks` writes before `finish`.
        let dst: *mut MaybeUninit<T> = unsafe { range.slice_uninit_mut(0, len) }.as_mut_ptr();
        // SAFETY: `dst` has room for `len` values of `T`, the program's output type.
        unsafe { self.run_blocks(Target::Output(dst.cast()), validity, &mut out_validity)? };
        let mask = out_validity.into_mask(nullability)?;
        // SAFETY: `run_blocks` wrote every one of the `len` values, and `mask` covers them.
        unsafe {
            range.append_mask(&mask);
            range.finish();
        }
        Ok(())
    }

    /// Whether `ops[i]` may overwrite its input: it supports it and keeps the width.
    fn in_place(&self, i: usize) -> bool {
        self.ops[i].supports_in_place()
            && self.ptypes[i].byte_width() == self.ptypes[i + 1].byte_width()
    }

    /// Run the source through every op, block by block, into `target`, starting each block's
    /// validity from `validity` and recording the result in `out_validity`.
    ///
    /// Each block is the source's, in its scratch or lent, or the leaf's own memory. An op that
    /// keeps the width and allows it overwrites its block; any other op copies it into the other
    /// scratch block, and the last op writes `target`.
    ///
    /// # Safety
    ///
    /// For [`Target::Output`], the pointer must have room for `len` values of the program's output
    /// type. For [`Target::Leaf`], it must hold `len` initialized values of the source's type, and
    /// every op must be in place.
    unsafe fn run_blocks(
        &mut self,
        target: Target,
        validity: &Mask,
        out_validity: &mut OutputValidity,
    ) -> VortexResult<()> {
        let in_place: Vec<bool> = (0..self.ops.len()).map(|i| self.in_place(i)).collect();
        let (len, last) = (self.len, self.ops.len().checked_sub(1));
        let in_width = self.ptypes[0].byte_width();
        let out_width = self.ptype().byte_width();
        let offset = match target {
            Target::Leaf(_) => 0,
            Target::Output(_) => self.source.block_offset() % BLOCK_LEN,
        };
        // The source fills the first scratch block; the second is allocated when an op copies
        // between them.
        let mut scratch: [Option<Box<BlockBuf>>; 2] = [None, None];
        let mut bits = [0u8; VALIDITY_BYTES];
        let mut start = 0;
        while start < len {
            let rows = block_rows(start, len, offset);
            let n = rows.len();
            let mut state = validity_window(validity, rows.clone(), &mut bits);
            let (mut cur, mut at) = match target {
                // SAFETY: `start + n <= len`.
                Target::Leaf(base) => (unsafe { base.add(start * in_width) }, At::Leaf),
                Target::Output(_) => {
                    let source_scratch = scratch[0].get_or_insert_with(BlockBuf::new_boxed);
                    let mut block_validity = BlockValidity::new(n, &mut state, &mut bits);
                    match self
                        .source
                        .fill(rows.clone(), source_scratch, &mut block_validity)?
                    {
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
                            // The pointer comes from the program's own borrow of its scratch, so
                            // ops may write through it.
                            let base = scratch[0]
                                .as_deref_mut()
                                .vortex_expect("allocated above")
                                .as_mut_ptr();
                            // SAFETY: `offset + n <= BLOCK_LEN` values fit in the scratch.
                            (unsafe { base.add(offset * in_width) }, At::Scratch(0))
                        }
                        SourceBlock::Lent(block) => {
                            vortex_ensure_eq!(
                                block.len(),
                                n,
                                "block source emitted the wrong number of rows"
                            );
                            vortex_ensure_eq!(block.ptype(), self.ptypes[0]);
                            (block.ptr.cast_mut(), At::Lent)
                        }
                    }
                }
            };
            for (i, op) in self.ops.iter_mut().enumerate() {
                let (in_ptype, out_ptype) = (self.ptypes[i], self.ptypes[i + 1]);
                let dst = match target {
                    // SAFETY: `start + n <= len`, so the block fits in `dst`.
                    Target::Output(dst) => unsafe { dst.add(start * out_width) },
                    Target::Leaf(_) => cur,
                };
                let from = at;
                let values = if in_place[i] && at.writable() && (Some(i) != last || dst == cur) {
                    Values::InPlace(BlockInOut {
                        ptr: cur,
                        len: n,
                        in_ptype,
                        out_ptype,
                        _borrow: PhantomData,
                    })
                } else {
                    let (out, to) = if Some(i) == last {
                        (dst, At::Output)
                    } else {
                        let other = usize::from(at == At::Scratch(0));
                        let buf = scratch[other].get_or_insert_with(BlockBuf::new_boxed);
                        (buf.as_mut_ptr(), At::Scratch(other))
                    };
                    let values = Values::Copy {
                        input: BlockIn {
                            ptr: cur,
                            len: n,
                            ptype: in_ptype,
                            _borrow: PhantomData,
                        },
                        out: BlockOut {
                            ptr: out,
                            len: n,
                            ptype: out_ptype,
                            _borrow: PhantomData,
                        },
                    };
                    (cur, at) = (out, to);
                    values
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
                    validity: BlockValidity::new(n, &mut state, &mut bits),
                };
                check_written(&op.process(start, block)?, cur, n, out_ptype)?;
            }
            if let (None, Target::Output(dst)) = (last, target) {
                // With no ops the source's values are the output.
                // SAFETY: `cur` holds `n` values of the source's type, which `dst` has room for
                // from `start`, and the two never overlap.
                unsafe {
                    std::ptr::copy_nonoverlapping(cur, dst.add(start * in_width), n * in_width)
                };
            }
            out_validity.push(start, n, state, &bits);
            start = rows.end;
        }
        Ok(())
    }
}

/// Where a program writes its values.
#[derive(Clone, Copy)]
enum Target {
    /// Room for every value of the output type.
    Output(*mut u8),
    /// The source's own buffer, which every op overwrites in place.
    Leaf(*mut u8),
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

/// The validity of `rows`: known from `validity` when it is uniform, or copied into `bits`.
fn validity_window(
    validity: &Mask,
    rows: Range<usize>,
    bits: &mut [u8; VALIDITY_BYTES],
) -> ValidityState {
    match validity {
        Mask::AllTrue(_) => ValidityState::AllValid,
        Mask::AllFalse(_) => ValidityState::AllInvalid,
        Mask::Values(values) => {
            let buffer = values.bit_buffer();
            copy_bits(
                buffer.inner().as_slice(),
                buffer.offset() + rows.start,
                bits,
                0,
                rows.len(),
            );
            ValidityState::Bits
        }
    }
}

/// Check an op wrote the block it was handed.
fn check_written(
    written: &Written<'_>,
    ptr: *mut u8,
    len: usize,
    ptype: PType,
) -> VortexResult<()> {
    vortex_ensure!(
        std::ptr::eq(written.ptr, ptr),
        "block op returned a block other than the one it was handed"
    );
    vortex_ensure_eq!(written.len, len, "block op wrote the wrong number of rows");
    vortex_ensure_eq!(written.ptype, ptype, "block op wrote the wrong type");
    Ok(())
}

/// The validity of a program's output, gathered block by block. It holds no bitmap until a block
/// has a null row.
struct OutputValidity {
    len: usize,
    bits: Option<BitBufferMut>,
}

impl OutputValidity {
    fn new(len: usize) -> Self {
        Self { len, bits: None }
    }

    fn push(&mut self, start: usize, n: usize, state: ValidityState, bits: &[u8; VALIDITY_BYTES]) {
        let len = self.len;
        match state {
            // A bitmap starts all valid, so valid rows need no writing.
            ValidityState::AllValid => {}
            ValidityState::AllInvalid => self
                .bits
                .get_or_insert_with(|| BitBufferMut::new_set(len))
                .fill_range(start, start + n, false),
            ValidityState::Bits => {
                let out = self.bits.get_or_insert_with(|| BitBufferMut::new_set(len));
                let offset = out.offset();
                copy_bits(bits, 0, out.as_mut_slice(), offset + start, n);
            }
        }
    }

    fn into_validity(self, nullability: Nullability) -> VortexResult<Validity> {
        Ok(match (self.into_bits(nullability)?, nullability) {
            (None, Nullability::NonNullable) => Validity::NonNullable,
            (None, Nullability::Nullable) => Validity::AllValid,
            (Some(bits), _) => Validity::from(bits),
        })
    }

    fn into_mask(self, nullability: Nullability) -> VortexResult<Mask> {
        let len = self.len;
        Ok(match self.into_bits(nullability)? {
            None => Mask::new_true(len),
            Some(bits) => Mask::from_buffer(bits),
        })
    }

    /// The output bitmap, if a block had a null row. A non-nullable output must have none.
    fn into_bits(self, nullability: Nullability) -> VortexResult<Option<BitBuffer>> {
        let Some(bits) = self.bits.map(BitBufferMut::freeze) else {
            return Ok(None);
        };
        if nullability == Nullability::NonNullable {
            vortex_ensure_eq!(
                bits.true_count(),
                bits.len(),
                "a non-nullable block program produced null rows"
            );
            return Ok(None);
        }
        Ok(Some(bits))
    }
}
