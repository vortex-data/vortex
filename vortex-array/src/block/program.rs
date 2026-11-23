// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The block program a [`FusedChain`](super::FusedChain) builds, and the loops that run it.

use std::marker::PhantomData;
use std::ops::Range;

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use super::BLOCK_LEN;
use super::Block;
use super::BlockBuf;
use super::BlockIn;
use super::BlockInOut;
use super::BlockOp;
use super::BlockOut;
use super::BlockSource;
use super::SourceBlock;
use super::Written;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::trace_op;
use crate::validity::Validity;

/// The primitive type of an array the executor can fuse, or `None` if it is not primitive.
pub(crate) fn fusable_ptype(array: &ArrayRef) -> Option<PType> {
    match array.dtype() {
        DType::Primitive(ptype, _) => Some(*ptype),
        _ => None,
    }
}

/// A fused chain built from the leaf up: the leaf's source, then the ops of the parents above it.
pub(crate) struct BlockProgram {
    source: Box<dyn BlockSource>,
    ops: Vec<Box<dyn BlockOp>>,
    /// The input's type, then each op's output type.
    ptypes: Vec<PType>,
    len: usize,
}

impl BlockProgram {
    pub(crate) fn new(source: Box<dyn BlockSource>, ptype: PType, len: usize) -> Self {
        Self {
            source,
            ops: Vec::new(),
            ptypes: vec![ptype],
            len,
        }
    }

    /// Transform the program's values with `op`, which writes values of `ptype`.
    pub(crate) fn push(&mut self, op: Box<dyn BlockOp>, ptype: PType) {
        self.ops.push(op);
        self.ptypes.push(ptype);
    }

    pub(crate) fn ops(&self) -> usize {
        self.ops.len()
    }

    /// Run the program block by block into a primitive array with `validity`.
    pub(crate) fn run(
        mut self,
        validity: Validity,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<PrimitiveArray> {
        vortex_ensure!(
            !self.ops.is_empty(),
            "a block program needs at least one op"
        );
        let (len, out_ptype) = (self.len, *self.ptypes.last().vortex_expect("input type"));
        let mut source = self.source;
        // When every op keeps the width and the source owns its buffer outright, such as a
        // canonical leaf nothing else references, each block is overwritten where it lies.
        let in_place = (0..self.ops.len()).all(|i| in_place_op(&self.ops, &self.ptypes, i));
        if in_place && let Some(mut values) = source.take_buffer() {
            vortex_ensure_eq!(
                values.len(),
                len * self.ptypes[0].byte_width(),
                "an owned source buffer holds the program's rows"
            );
            // SAFETY: `values` holds `len` initialized values of `ptypes[0]`, and every op keeps
            // the width, so each block is overwritten in place.
            unsafe { run_blocks_in_place(&mut self.ops, &self.ptypes, len, values.as_mut_ptr())? };
            return Ok(PrimitiveArray::from_byte_buffer(
                values.freeze(),
                out_ptype,
                validity,
            ));
        }
        match_each_native_ptype!(out_ptype, |T| {
            let mut output = BufferMut::<T>::with_capacity_in(len, ctx.allocator().clone());
            let dst = output.spare_capacity_mut().as_mut_ptr().cast::<u8>();
            // SAFETY: `dst` has room for `len` values of `T`, the program's output type.
            unsafe { run_blocks(source.as_mut(), &mut self.ops, &self.ptypes, len, dst)? };
            // SAFETY: `run_blocks` wrote every one of the `len` values.
            unsafe { output.set_len(len) };
            Ok(PrimitiveArray::new(output.freeze(), validity))
        })
    }
}

/// Lends a canonical primitive array's values from its buffer.
pub(crate) struct PrimitiveSource<T> {
    values: Buffer<T>,
}

impl<T: NativePType> PrimitiveSource<T> {
    pub(crate) fn new(values: Buffer<T>) -> Self {
        Self { values }
    }
}

impl<T: NativePType> BlockSource for PrimitiveSource<T> {
    fn fill<'s>(
        &'s mut self,
        rows: Range<usize>,
        _scratch: &'s mut BlockBuf,
    ) -> VortexResult<SourceBlock<'s>> {
        Ok(SourceBlock::Lent(BlockIn::from_slice(
            &self.values.as_slice()[rows],
        )))
    }

    fn take_buffer(&mut self) -> Option<ByteBufferMut> {
        match std::mem::replace(&mut self.values, Buffer::empty()).try_into_mut() {
            Ok(values) => Some(values.into_byte_buffer()),
            Err(values) => {
                self.values = values;
                None
            }
        }
    }
}

/// The rows of the block starting at `start`: up to the next multiple of [`BLOCK_LEN`] past
/// `offset`, so every block lines up with the source's own blocks.
fn block_rows(start: usize, len: usize, offset: usize) -> Range<usize> {
    let to_boundary = BLOCK_LEN - (offset + start) % BLOCK_LEN;
    start..(start + to_boundary).min(len)
}

/// Whether `ops[i]` may overwrite its input: it supports it and keeps the width.
fn in_place_op(ops: &[Box<dyn BlockOp>], ptypes: &[PType], i: usize) -> bool {
    ops[i].supports_in_place() && ptypes[i].byte_width() == ptypes[i + 1].byte_width()
}

/// Run every op in place over `len` values of `ptypes[0]` at `values`.
///
/// # Safety
///
/// `values` must hold `len` initialized values of `ptypes[0]`, and every op must be in place.
unsafe fn run_blocks_in_place(
    ops: &mut [Box<dyn BlockOp>],
    ptypes: &[PType],
    len: usize,
    values: *mut u8,
) -> VortexResult<()> {
    let width = ptypes[0].byte_width();
    let mut start = 0;
    while start < len {
        let rows = block_rows(start, len, 0);
        let n = rows.len();
        // SAFETY: `start + n <= len`.
        let ptr = unsafe { values.add(start * width) };
        for (i, op) in ops.iter_mut().enumerate() {
            let block = Block::InPlace(BlockInOut {
                ptr,
                len: n,
                in_ptype: ptypes[i],
                out_ptype: ptypes[i + 1],
                _borrow: PhantomData,
            });
            if start == 0 {
                trace_op!(record_block_op(
                    i,
                    "in_place",
                    "leaf",
                    "leaf",
                    ptypes[i],
                    ptypes[i + 1]
                ));
            }
            check_written(&op.process(start, block)?, ptr, n, ptypes[i + 1])?;
        }
        start = rows.end;
    }
    Ok(())
}

/// How a trace names the memory a block lives in.
fn scratch_name(scratch: Option<usize>) -> &'static str {
    match scratch {
        None => "source",
        Some(0) => "scratch0",
        Some(_) => "scratch1",
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

/// Run `source` through `ops` into `dst`, overwriting scratch blocks in place where an op allows
/// it and copying between two scratch blocks where it does not. The last op writes `dst`.
///
/// # Safety
///
/// `dst` must have room for `len` values of `ptypes.last()`.
unsafe fn run_blocks(
    source: &mut dyn BlockSource,
    ops: &mut [Box<dyn BlockOp>],
    ptypes: &[PType],
    len: usize,
    dst: *mut u8,
) -> VortexResult<()> {
    // The source writes the first scratch block. The second is allocated when an op copies
    // between scratch blocks.
    let mut scratch = [Some(BlockBuf::new_boxed()), None];
    let in_place: Vec<bool> = (0..ops.len())
        .map(|i| in_place_op(ops, ptypes, i))
        .collect();
    let in_width = ptypes[0].byte_width();
    let out_width = ptypes[ops.len()].byte_width();
    let last = ops.len() - 1;
    let offset = source.block_offset() % BLOCK_LEN;
    let mut start = 0;
    while start < len {
        let rows = block_rows(start, len, offset);
        let n = rows.len();
        let source_scratch = scratch[0].as_deref_mut().vortex_expect("allocated above");
        // Pointers into scratch come from the executor's own mutable borrow of it, so ops may
        // write through them; a lent block is only ever read.
        let (mut cur, mut cur_scratch) = match source.fill(rows.clone(), source_scratch)? {
            SourceBlock::Scratch {
                offset,
                len: filled,
            } => {
                vortex_ensure_eq!(filled, n, "block source emitted the wrong number of rows");
                vortex_ensure!(
                    offset + filled <= BLOCK_LEN,
                    "block source overran its scratch"
                );
                let base = scratch[0]
                    .as_deref_mut()
                    .vortex_expect("allocated above")
                    .as_mut_ptr();
                // SAFETY: `offset + n <= BLOCK_LEN` values of at most 8 bytes fit in the scratch.
                (unsafe { base.add(offset * in_width) }, Some(0))
            }
            SourceBlock::Lent(block) => {
                vortex_ensure_eq!(
                    block.len(),
                    n,
                    "block source emitted the wrong number of rows"
                );
                vortex_ensure_eq!(block.ptype(), ptypes[0]);
                (block.ptr.cast_mut(), None)
            }
        };
        for (i, op) in ops.iter_mut().enumerate() {
            let out_ptype = ptypes[i + 1];
            let from = scratch_name(cur_scratch);
            let block = if i != last && in_place[i] && cur_scratch.is_some() {
                if start == 0 {
                    trace_op!(record_block_op(
                        i, "in_place", from, from, ptypes[i], out_ptype
                    ));
                }
                Block::InPlace(BlockInOut {
                    ptr: cur,
                    len: n,
                    in_ptype: ptypes[i],
                    out_ptype,
                    _borrow: PhantomData,
                })
            } else {
                let out = if i == last {
                    if start == 0 {
                        trace_op!(record_block_op(
                            i, "copy", from, "output", ptypes[i], out_ptype
                        ));
                    }
                    // SAFETY: `start + n <= len`, so the block fits in `dst`.
                    unsafe { dst.add(start * out_width) }
                } else {
                    let target = usize::from(cur_scratch == Some(0));
                    cur_scratch = Some(target);
                    if start == 0 {
                        let to = scratch_name(cur_scratch);
                        trace_op!(record_block_op(i, "copy", from, to, ptypes[i], out_ptype));
                    }
                    scratch[target]
                        .get_or_insert_with(BlockBuf::new_boxed)
                        .as_mut_ptr()
                };
                let copy = Block::Copy {
                    input: BlockIn {
                        ptr: cur,
                        len: n,
                        ptype: ptypes[i],
                        _borrow: PhantomData,
                    },
                    out: BlockOut {
                        ptr: out,
                        len: n,
                        ptype: out_ptype,
                        _borrow: PhantomData,
                    },
                };
                cur = out;
                copy
            };
            check_written(&op.process(start, block)?, cur, n, out_ptype)?;
        }
        start = rows.end;
    }
    Ok(())
}
