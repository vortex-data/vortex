// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block-fused execution of primitive encoding chains.
//!
//! The iterative executor trampolines down an encoding tree with `ExecuteSlot` frames. An
//! encoding that asks for its last child with [`ExecuteSpine`](crate::ExecutionStep::ExecuteSpine)
//! promises that once that child is canonical it only transforms the child's values row by row,
//! as its [`VTable::block_op`](crate::vtable::VTable::block_op) does.
//!
//! On the way back up, instead of executing a child into a full-length buffer and then the
//! parent's kernel into another, the executor builds a block program: the first child that is a
//! leaf, one whose [`VTable::block_source`](crate::vtable::VTable::block_source) emits its
//! values block by block (a canonical primitive array lends its buffer), starts it, and each spine parent above extends it with its [`BlockOp`],
//! popping the parent's frame. When the next parent cannot extend it, or at the root, the executor
//! drives the program right away, in place of the parents it reached: blocks of at most
//! [`BLOCK_LEN`] rows flow through every op in L1, and only the last op writes full-length
//! output. Execution then carries on as before. A chain is never left waiting, so it is a value of
//! the executor's loop rather than an array.

use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::Range;

use num_traits::AsPrimitive;
use vortex_buffer::ByteBufferMut;
use vortex_compute::lane_kernels::IndexedSinkExt;
use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_compute::lane_kernels::ReinterpretSink;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::match_each_unsigned_integer_ptype;
use crate::patches::Patches;

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

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.0.as_mut_ptr().cast()
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
#[non_exhaustive]
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

/// Where a [`BlockSource`] put the values of a block.
#[non_exhaustive]
pub enum SourceBlock<'s> {
    /// In the `scratch` the executor handed it, as `len` values of the source's type starting at
    /// value `offset`. The executor may overwrite them in place.
    ///
    /// Built with the unsafe [`SourceBlock::scratch`], since the executor reads those values as
    /// initialized.
    #[non_exhaustive]
    Scratch {
        /// The index of the first value in the scratch.
        offset: usize,
        /// The number of values.
        len: usize,
    },
    /// In memory the source owns, which ops never overwrite.
    Lent(BlockIn<'s>),
}

impl SourceBlock<'_> {
    /// Values the source wrote into its scratch: `len` values from value `offset`.
    ///
    /// # Safety
    ///
    /// The source wrote values `offset..offset + len` of the scratch it was handed, in its own
    /// type, during this call to [`BlockSource::fill`].
    pub unsafe fn scratch(offset: usize, len: usize) -> Self {
        Self::Scratch { offset, len }
    }
}

/// The bottom of a fused chain: emits an array's own values, block by block, in row order.
pub trait BlockSource: Send {
    /// Emit exactly the values of `rows`, at most [`BLOCK_LEN`] of them, written into `scratch`
    /// or lent from memory the source owns.
    ///
    /// The executor asks for consecutive ranges in row order, starting at 0, and stops once they
    /// cover the array. Values written into `scratch` belong to the executor once returned: it
    /// may overwrite them, so a source must not keep state in `scratch` between calls.
    fn fill<'s>(
        &'s mut self,
        rows: Range<usize>,
        scratch: &'s mut BlockBuf,
    ) -> VortexResult<SourceBlock<'s>>;

    /// The row at which this source would start its second block if it chose its own blocks,
    /// modulo [`BLOCK_LEN`]: a sliced BitPacked array's offset into its first chunk, so its blocks
    /// line up with its chunks. The executor cuts blocks at these boundaries.
    fn block_offset(&self) -> usize {
        0
    }

    /// The source's values as a buffer the executor may overwrite in place, if the source holds
    /// the only reference to them: a canonical array's own buffer. The executor asks at most once,
    /// before any [`fill`](Self::fill), and fills the source no more once it returns `Some`.
    fn take_buffer(&mut self) -> Option<ByteBufferMut> {
        None
    }
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
    /// The patched rows, in increasing order, with their values.
    patches: Vec<(usize, T)>,
    /// The first patch not yet applied.
    cursor: usize,
}

impl<T: NativePType> BlockPatches<T> {
    /// Execute `patches` and map each patch value with `map(row, value)`.
    ///
    /// `None` if a patch value is null: level-wise execution patches validity too, which blocks
    /// of values cannot, so the array should not fuse.
    pub fn try_new<S: NativePType>(
        patches: &Patches,
        ctx: &mut ExecutionCtx,
        map: impl Fn(usize, S) -> T,
    ) -> VortexResult<Option<Self>> {
        let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
        vortex_ensure_eq!(
            indices.len(),
            values.len(),
            "patch indices and values differ in length"
        );
        if !values.all_valid(ctx)? {
            return Ok(None);
        }
        let offset = patches.offset();
        let patches = match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
            indices
                .as_slice::<I>()
                .iter()
                .zip(values.as_slice::<S>())
                .map(|(&index, &value)| {
                    let row = AsPrimitive::<usize>::as_(index) - offset;
                    (row, map(row, value))
                })
                .collect()
        });
        Ok(Some(Self { patches, cursor: 0 }))
    }

    /// Overwrite the patched rows of `block`, which holds the rows from `start`.
    ///
    /// Blocks must be applied in row order.
    #[inline]
    pub fn apply(&mut self, start: usize, block: &mut [T]) {
        let end = start + block.len();
        while let Some(&(row, value)) = self.patches.get(self.cursor) {
            if row >= end {
                break;
            }
            if row >= start {
                block[row - start] = value;
            }
            self.cursor += 1;
        }
    }
}

mod chain;
mod program;
#[cfg(test)]
mod tests;

pub(crate) use chain::FusedChain;
pub(crate) use program::BlockProgram;
pub(crate) use program::PrimitiveSource;
pub(crate) use program::fusable_ptype;
