// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block-fused execution of primitive encoding chains.
//!
//! The iterative executor descends an encoding tree with `ExecuteSlot` frames. An encoding that
//! asks for its last child with [`ExecuteSpine`](crate::ExecutionStep::ExecuteSpine) promises that
//! once that child is canonical it only transforms the child's values row by row. When the
//! executor reaches a leaf, the spine frames on top of its stack are therefore a chain of parents
//! that each transform their child. Instead of executing the leaf to a full-length buffer,
//! popping, running the parent's kernel into another full-length buffer and so on, the executor
//! fuses the chain: the leaf emits blocks of at most [`BLOCK_LEN`] rows ([`BlockSource`]), each
//! parent transforms a block ([`BlockOp`]), and only the outermost parent writes full-length
//! output. Every intermediate stays in L1.
//!
//! An encoding opts in by requesting its spine with [`require_spine!`](crate::require_spine) and
//! implementing [`VTable::block_op`](crate::vtable::VTable::block_op) for that slot, and a leaf by
//! implementing [`VTable::block_source`](crate::vtable::VTable::block_source).

use std::marker::PhantomData;
use std::mem::MaybeUninit;

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_compute::lane_kernels::IndexedSinkExt;
use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_compute::lane_kernels::ReinterpretSink;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::patches::Patches;
use crate::validity::Validity;

#[cfg(any(test, feature = "_test-harness"))]
thread_local! {
    static FUSED_CHAINS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static IN_PLACE_CHAINS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The number of fused chains this thread decoded in place over their leaf's buffer, for tests.
#[cfg(any(test, feature = "_test-harness"))]
pub fn in_place_chain_count() -> usize {
    IN_PLACE_CHAINS.with(std::cell::Cell::get)
}

/// The number of chains the executor fused on this thread, for tests to check fusion fired.
#[cfg(any(test, feature = "_test-harness"))]
pub fn fused_chain_count() -> usize {
    FUSED_CHAINS.with(std::cell::Cell::get)
}

/// The most rows a block holds.
pub const BLOCK_LEN: usize = 1024;

/// Uninitialized scratch memory for one block of any primitive type.
#[repr(C, align(64))]
pub struct BlockBuf([MaybeUninit<u64>; BLOCK_LEN]);

impl BlockBuf {
    /// Allocate scratch for one block without initializing it.
    pub fn new_boxed() -> Box<Self> {
        // SAFETY: `BlockBuf` holds only `MaybeUninit` values, which need no initialization.
        unsafe { Box::<Self>::new_uninit().assume_init() }
    }

    /// The scratch as an uninitialized destination for `len` values of `T`.
    pub fn out<T: NativePType>(&mut self, len: usize) -> &mut [MaybeUninit<T>] {
        assert!(len <= BLOCK_LEN, "a block holds at most {BLOCK_LEN} rows");
        // SAFETY: the buffer holds `BLOCK_LEN` 8-byte values aligned to 64 bytes, so it fits
        // `len <= BLOCK_LEN` values of any primitive type of at most 8 bytes at their alignment.
        unsafe { std::slice::from_raw_parts_mut(self.0.as_mut_ptr().cast(), len) }
    }

    /// View the first `len` values of `T`, which the caller has written.
    ///
    /// # Safety
    ///
    /// The first `len` values of `T` must be initialized.
    pub unsafe fn view<T: NativePType>(&self, len: usize) -> BlockIn<'_> {
        assert!(len <= BLOCK_LEN, "a block holds at most {BLOCK_LEN} rows");
        BlockIn {
            ptr: self.0.as_ptr().cast(),
            len,
            ptype: T::PTYPE,
            _borrow: PhantomData,
        }
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.0.as_mut_ptr().cast()
    }

    fn contains(&self, ptr: *mut u8) -> bool {
        self.0.as_ptr_range().contains(&ptr.cast_const().cast())
    }
}

/// A block of initialized values of one primitive type.
#[derive(Clone, Copy)]
pub struct BlockIn<'a> {
    ptr: *const u8,
    len: usize,
    ptype: PType,
    _borrow: PhantomData<&'a [u8]>,
}

impl<'a> BlockIn<'a> {
    /// A block over initialized values.
    pub fn from_slice<T: NativePType>(values: &'a [T]) -> Self {
        assert!(
            values.len() <= BLOCK_LEN,
            "a block holds at most {BLOCK_LEN} rows"
        );
        Self {
            ptr: values.as_ptr().cast(),
            len: values.len(),
            ptype: T::PTYPE,
            _borrow: PhantomData,
        }
    }

    /// The number of rows in the block.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the block is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The type of the block's values.
    #[inline]
    pub fn ptype(&self) -> PType {
        self.ptype
    }

    /// The block's values.
    ///
    /// # Panics
    ///
    /// Panics if `T` is not the block's type.
    #[inline]
    pub fn as_slice<T: NativePType>(&self) -> &'a [T] {
        assert_eq!(T::PTYPE, self.ptype, "block holds {}", self.ptype);
        // SAFETY: the constructors guarantee `len` initialized values of `ptype` at `ptr`.
        unsafe { std::slice::from_raw_parts(self.ptr.cast(), self.len) }
    }
}

/// Uninitialized memory a [`BlockOp`] writes one block of values into.
pub struct BlockOut<'a> {
    ptr: *mut u8,
    len: usize,
    ptype: PType,
    _borrow: PhantomData<&'a mut [u8]>,
}

impl BlockOut<'_> {
    /// The number of rows the block must receive.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the block is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The destination for the block's values, every one of which the op must write.
    ///
    /// # Panics
    ///
    /// Panics if `T` is not the type the block must hold.
    #[inline]
    pub fn as_uninit<T: NativePType>(&mut self) -> &mut [MaybeUninit<T>] {
        assert_eq!(T::PTYPE, self.ptype, "block holds {}", self.ptype);
        // SAFETY: the executor hands out `len` writable values of `ptype` at `ptr`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.cast(), self.len) }
    }
}

/// A block an op overwrites: it reads each value as one type and writes it back as another of the
/// same width.
pub struct BlockInOut<'a> {
    ptr: *mut u8,
    len: usize,
    in_ptype: PType,
    out_ptype: PType,
    _borrow: PhantomData<&'a mut [u8]>,
}

impl<'a> BlockInOut<'a> {
    /// The number of rows in the block.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the block is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The block's values, for an op whose input and output types are both `T`.
    ///
    /// # Panics
    ///
    /// Panics if the block does not read and write `T`.
    #[inline]
    pub fn as_mut_slice<T: NativePType>(&mut self) -> &mut [T] {
        assert_eq!(T::PTYPE, self.in_ptype, "block holds {}", self.in_ptype);
        assert_eq!(
            T::PTYPE,
            self.out_ptype,
            "block must hold {}",
            self.out_ptype
        );
        // SAFETY: the executor hands out `len` initialized values of `in_ptype` at `ptr`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.cast(), self.len) }
    }

    /// Declare the block written, once every value holds its output.
    ///
    /// # Safety
    ///
    /// Every value must have been overwritten with a value of the output type.
    #[inline]
    pub unsafe fn assume_written(self) -> Written<'a> {
        Written {
            ptr: self.ptr,
            len: self.len,
            ptype: self.out_ptype,
            _borrow: PhantomData,
        }
    }
}

/// One block for a [`BlockOp`] to transform: from an input into separate memory, or in place.
pub enum Block<'a> {
    /// Read `input` and write `out`, which never alias.
    Copy {
        /// The child's values.
        input: BlockIn<'a>,
        /// Where the op's values go.
        out: BlockOut<'a>,
    },
    /// Read each value and overwrite it with the op's value. Only offered to ops that support it
    /// when their input and output types have the same width.
    InPlace(BlockInOut<'a>),
}

impl<'a> Block<'a> {
    /// The number of rows in the block.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Block::Copy { input, .. } => input.len(),
            Block::InPlace(block) => block.len(),
        }
    }

    /// Whether the block is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Write `f(value)` for every value, with the lane kernels level-wise execution uses:
    /// `map_into` from the input, or `map_into_in_place` over the block itself.
    ///
    /// # Panics
    ///
    /// Panics if the block does not read `S` and write `T`.
    #[inline]
    pub fn map<S: NativePType, T: NativePType>(self, f: impl Fn(S) -> T) -> Written<'a> {
        match self {
            Block::Copy { input, mut out } => {
                input.as_slice::<S>().map_into(out.as_uninit::<T>(), f);
                // SAFETY: `map_into` wrote every value of `out`.
                unsafe { Written::assume_init(out) }
            }
            Block::InPlace(block) => {
                assert_eq!(S::PTYPE, block.in_ptype, "block holds {}", block.in_ptype);
                assert_eq!(
                    T::PTYPE,
                    block.out_ptype,
                    "block must hold {}",
                    block.out_ptype
                );
                // SAFETY: the executor hands out `len` initialized values of `S` at `ptr`.
                let values =
                    unsafe { std::slice::from_raw_parts_mut(block.ptr.cast::<S>(), block.len) };
                ReinterpretSink::<S, T>::new(values).map_into_in_place(f);
                // SAFETY: `map_into_in_place` overwrote every value with a `T`.
                unsafe { block.assume_written() }
            }
        }
    }
}

/// Proof that every value of a block was written, which [`BlockOp::process`] returns.
pub struct Written<'a> {
    ptr: *mut u8,
    len: usize,
    ptype: PType,
    _borrow: PhantomData<&'a mut [u8]>,
}

impl<'a> Written<'a> {
    /// Declare a `Copy` output written, for ops that fill it themselves.
    ///
    /// # Safety
    ///
    /// Every value of `out` must be initialized.
    #[inline]
    pub unsafe fn assume_init(out: BlockOut<'a>) -> Self {
        Self {
            ptr: out.ptr,
            len: out.len,
            ptype: out.ptype,
            _borrow: PhantomData,
        }
    }

    /// The written values, for example to patch them.
    ///
    /// # Panics
    ///
    /// Panics if the block does not hold `T`.
    #[inline]
    pub fn as_mut_slice<T: NativePType>(&mut self) -> &mut [T] {
        assert_eq!(T::PTYPE, self.ptype, "block holds {}", self.ptype);
        // SAFETY: the constructors guarantee `len` initialized values of `ptype` at `ptr`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.cast(), self.len) }
    }
}

/// The bottom of a fused chain: emits an array's own values, block by block, in row order.
pub trait BlockSource: Send {
    /// Emit the next block of at most [`BLOCK_LEN`] rows, starting where the last one ended.
    ///
    /// The block holds values of the array's primitive type, written into `scratch` or borrowed
    /// from the source. The executor stops calling once the blocks cover the array.
    fn next_block<'s>(&'s mut self, scratch: &'s mut BlockBuf) -> VortexResult<BlockIn<'s>>;
}

/// A parent's transform of one block of its child's values into one block of its own.
pub trait BlockOp: Send {
    /// Write the parent's values for rows `start..start + block.len()`, from the child's values
    /// in its primitive type to the parent's in its own.
    ///
    /// Most ops only call [`Block::map`], which handles both shapes of [`Block`].
    fn process<'a>(&mut self, start: usize, block: Block<'a>) -> VortexResult<Written<'a>>;

    /// Whether the executor may hand this op [`Block::InPlace`] blocks. An op that reads values
    /// other than the one it is writing must return `false`.
    fn supports_in_place(&self) -> bool {
        true
    }
}

/// Patches resolved once and applied block by block in row order.
pub struct BlockPatches<T> {
    indices: Vec<usize>,
    values: Vec<T>,
    cursor: usize,
}

impl<T: NativePType> BlockPatches<T> {
    /// Execute `patches` and map each patch value with `map(row, value)`.
    pub fn try_new<S: NativePType>(
        patches: &Patches,
        ctx: &mut ExecutionCtx,
        map: impl Fn(usize, S) -> T,
    ) -> VortexResult<Self> {
        let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
        let offset = patches.offset();
        let indices: Vec<usize> = match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
            indices
                .as_slice::<I>()
                .iter()
                .map(|&index| AsPrimitive::<usize>::as_(index) - offset)
                .collect()
        });
        let values = indices
            .iter()
            .zip(values.as_slice::<S>())
            .map(|(&row, &value)| map(row, value))
            .collect();
        Ok(Self {
            indices,
            values,
            cursor: 0,
        })
    }

    /// Overwrite the patched rows of `block`, which holds the rows from `start`.
    ///
    /// Blocks must be applied in row order.
    #[inline]
    pub fn apply(&mut self, start: usize, block: &mut [T]) {
        let end = start + block.len();
        while let Some(&row) = self.indices.get(self.cursor) {
            if row >= end {
                break;
            }
            if row >= start {
                block[row - start] = self.values[self.cursor];
            }
            self.cursor += 1;
        }
    }
}

/// Emits a canonical primitive array's values straight from its buffer.
pub(crate) struct PrimitiveSource<T> {
    values: Buffer<T>,
    next: usize,
}

impl<T: NativePType> PrimitiveSource<T> {
    pub(crate) fn new(values: Buffer<T>) -> Self {
        Self { values, next: 0 }
    }
}

impl<T: NativePType> BlockSource for PrimitiveSource<T> {
    fn next_block<'s>(&'s mut self, _scratch: &'s mut BlockBuf) -> VortexResult<BlockIn<'s>> {
        let start = self.next;
        let end = (start + BLOCK_LEN).min(self.values.len());
        self.next = end;
        Ok(BlockIn::from_slice(&self.values.as_slice()[start..end]))
    }
}

/// The primitive type of a fusable array, or `None` if it is not primitive.
pub(crate) fn fusable_ptype(array: &ArrayRef) -> Option<PType> {
    match array.dtype() {
        DType::Primitive(ptype, _) => Some(*ptype),
        _ => None,
    }
}

/// The bottom of a chain the executor runs.
pub(crate) enum FusionInput {
    /// A leaf's block source.
    Source(Box<dyn BlockSource>),
    /// A canonical primitive array, which the chain may decode in place when the executor holds
    /// its only reference.
    Canonical(PrimitiveArray),
}

/// Run `input` through `ops`, from the leaf up, into `len` values of the last type in `ptypes`.
///
/// `ptypes[i]` is the type `ops[i]` reads and `ptypes[i + 1]` the type it writes.
pub(crate) fn run_fused(
    input: FusionInput,
    ops: &mut [Box<dyn BlockOp>],
    ptypes: &[PType],
    len: usize,
    validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    debug_assert_eq!(ptypes.len(), ops.len() + 1);
    #[cfg(any(test, feature = "_test-harness"))]
    FUSED_CHAINS.with(|count| count.set(count.get() + 1));
    let out_ptype = *ptypes.last().vortex_expect("at least one type");
    let mut source: Box<dyn BlockSource> = match input {
        FusionInput::Source(source) => source,
        FusionInput::Canonical(array) => {
            let in_place = (0..ops.len()).all(|i| in_place_op(ops, ptypes, i));
            match_each_native_ptype!(array.ptype(), |S| {
                match array.try_into_buffer_mut::<S>() {
                    Ok(mut values) if in_place => {
                        // SAFETY: `values` holds `len` initialized values of `ptypes[0]`, and
                        // every op keeps the width, so each block is overwritten in place.
                        unsafe {
                            run_blocks_in_place(ops, ptypes, len, values.as_mut_ptr().cast())?
                        };
                        #[cfg(any(test, feature = "_test-harness"))]
                        IN_PLACE_CHAINS.with(|count| count.set(count.get() + 1));
                        return Ok(PrimitiveArray::from_byte_buffer(
                            values.into_byte_buffer().freeze(),
                            out_ptype,
                            validity,
                        ));
                    }
                    Ok(values) => {
                        Box::new(PrimitiveSource::new(values.freeze())) as Box<dyn BlockSource>
                    }
                    Err(values) => Box::new(PrimitiveSource::new(values)),
                }
            })
        }
    };
    match_each_native_ptype!(out_ptype, |T| {
        let mut output = BufferMut::<T>::with_capacity_in(len, ctx.allocator().clone());
        let dst = output.spare_capacity_mut().as_mut_ptr().cast::<u8>();
        // SAFETY: `dst` has room for `len` values of `T`, the type of `ptypes.last()`.
        unsafe { run_blocks(source.as_mut(), ops, ptypes, len, dst)? };
        // SAFETY: `run_blocks` wrote every one of the `len` values.
        unsafe { output.set_len(len) };
        Ok(PrimitiveArray::new(output.freeze(), validity))
    })
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
        let n = BLOCK_LEN.min(len - start);
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
            check_written(&op.process(start, block)?, ptr, n, ptypes[i + 1])?;
        }
        start += n;
    }
    Ok(())
}

/// Check an op wrote the block it was handed.
fn check_written(
    written: &Written<'_>,
    ptr: *mut u8,
    len: usize,
    ptype: PType,
) -> VortexResult<()> {
    vortex_ensure!(
        std::ptr::eq(written.ptr, ptr) && written.len == len && written.ptype == ptype,
        "block op returned a block other than the one it was handed"
    );
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
    let mut scratch = [BlockBuf::new_boxed(), BlockBuf::new_boxed()];
    let in_place: Vec<bool> = (0..ops.len())
        .map(|i| in_place_op(ops, ptypes, i))
        .collect();
    let out_width = ptypes[ops.len()].byte_width();
    let last = ops.len() - 1;
    let mut start = 0;
    while start < len {
        let block = source.next_block(&mut scratch[0])?;
        let n = block.len();
        vortex_ensure!(
            n > 0 && start + n <= len,
            "block source emitted {n} rows at row {start} of {len}"
        );
        vortex_ensure_eq!(block.ptype(), ptypes[0]);
        let mut cur = block.ptr.cast_mut();
        // Which scratch block holds the current values, if any: a source may lend its own memory,
        // which ops must not overwrite.
        let mut cur_scratch = scratch.iter().position(|buf| buf.contains(cur));
        for (i, op) in ops.iter_mut().enumerate() {
            let out_ptype = ptypes[i + 1];
            let block = if i != last && in_place[i] && cur_scratch.is_some() {
                Block::InPlace(BlockInOut {
                    ptr: cur,
                    len: n,
                    in_ptype: ptypes[i],
                    out_ptype,
                    _borrow: PhantomData,
                })
            } else {
                let out = if i == last {
                    // SAFETY: `start + n <= len`, so the block fits in `dst`.
                    unsafe { dst.add(start * out_width) }
                } else {
                    let target = usize::from(cur_scratch == Some(0));
                    cur_scratch = Some(target);
                    scratch[target].as_mut_ptr()
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
        start += n;
    }
    Ok(())
}
