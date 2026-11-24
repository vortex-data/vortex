// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block-at-a-time execution of primitive encodings.
//!
//! A [`BlockProgram`] reads an array's values from a [`BlockSource`] a block at a time, at most
//! [`BLOCK_LEN`] rows, and passes each block through a list of [`BlockOp`]s while it is in L1.
//! Only the last op writes full-length memory: a new buffer, the source's own buffer when it owns
//! it outright, or the uninitialized tail of a builder.
//!
//! A [`Block`] holds its rows' values and their validity. Values move between scratch blocks when
//! an op changes their width; validity is one bit per row and is always updated in place, so an op
//! that does not change validity leaves it untouched.

use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::Range;

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_compute::lane_kernels::IndexedSinkExt;
use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_compute::lane_kernels::ReinterpretSink;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_mask::Mask;

use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::patches::Patches;

mod bits;
mod program;
#[cfg(test)]
mod tests;

pub use program::BlockProgram;

/// The most rows a block holds.
pub const BLOCK_LEN: usize = 1024;

/// The bytes of one block's validity: a bit per row.
const VALIDITY_BYTES: usize = BLOCK_LEN / 8;

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
        // SAFETY: the program hands out `len` writable values of `ptype` at `ptr`.
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
        // SAFETY: the program hands out `len` initialized values of `in_ptype` at `ptr`.
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

/// One block of values for a [`BlockOp`] to transform: from an input into separate memory, or in
/// place.
#[non_exhaustive]
pub enum Values<'a> {
    /// Read `input` and write `out`, which never alias.
    Copy {
        /// The input values.
        input: BlockIn<'a>,
        /// Where the op's values go.
        out: BlockOut<'a>,
    },
    /// Read each value and overwrite it with the op's value. Only offered to ops that support it
    /// when their input and output types have the same width.
    InPlace(BlockInOut<'a>),
}

impl<'a> Values<'a> {
    /// The number of rows in the block.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Values::Copy { input, .. } => input.len(),
            Values::InPlace(block) => block.len(),
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
            Values::Copy { input, mut out } => {
                input.as_slice::<S>().map_into(out.as_uninit::<T>(), f);
                // SAFETY: `map_into` wrote every value of `out`.
                unsafe { Written::assume_init(out) }
            }
            Values::InPlace(block) => {
                assert_eq!(S::PTYPE, block.in_ptype, "block holds {}", block.in_ptype);
                assert_eq!(
                    T::PTYPE,
                    block.out_ptype,
                    "block must hold {}",
                    block.out_ptype
                );
                // SAFETY: the program hands out `len` initialized values of `S` at `ptr`.
                let values =
                    unsafe { std::slice::from_raw_parts_mut(block.ptr.cast::<S>(), block.len) };
                ReinterpretSink::<S, T>::new(values).map_into_in_place(f);
                // SAFETY: `map_into_in_place` overwrote every value with a `T`.
                unsafe { block.assume_written() }
            }
        }
    }
}

/// How much of a block's validity is known without its bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValidityState {
    AllValid,
    AllInvalid,
    /// The bits hold each row's validity.
    Bits,
}

/// The validity of one block's rows, which ops update in place.
///
/// A block whose rows are all valid, or all null, holds no bits until an op changes one row.
pub struct BlockValidity<'a> {
    len: usize,
    state: &'a mut ValidityState,
    bits: &'a mut [u8; VALIDITY_BYTES],
}

impl<'a> BlockValidity<'a> {
    fn new(len: usize, state: &'a mut ValidityState, bits: &'a mut [u8; VALIDITY_BYTES]) -> Self {
        Self { len, state, bits }
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

    /// Whether every row of the block is valid.
    #[inline]
    pub fn all_valid(&self) -> bool {
        *self.state == ValidityState::AllValid
    }

    /// Whether `row`, counted from the block's first row, is valid.
    #[inline]
    pub fn is_valid(&self, row: usize) -> bool {
        assert!(row < self.len, "row {row} outside a block of {}", self.len);
        match self.state {
            ValidityState::AllValid => true,
            ValidityState::AllInvalid => false,
            ValidityState::Bits => self.bits[row / 8] & (1 << (row % 8)) != 0,
        }
    }

    /// Make `row`, counted from the block's first row, valid or null.
    #[inline]
    pub fn set(&mut self, row: usize, valid: bool) {
        assert!(row < self.len, "row {row} outside a block of {}", self.len);
        match (*self.state, valid) {
            (ValidityState::AllValid, true) | (ValidityState::AllInvalid, false) => return,
            (ValidityState::AllValid, false) => self.bits.fill(u8::MAX),
            (ValidityState::AllInvalid, true) => self.bits.fill(0),
            (ValidityState::Bits, _) => {}
        }
        *self.state = ValidityState::Bits;
        let (byte, bit) = (row / 8, 1 << (row % 8));
        if valid {
            self.bits[byte] |= bit;
        } else {
            self.bits[byte] &= !bit;
        }
    }
}

/// One block for a [`BlockOp`]: its rows' values, and their validity.
pub struct Block<'a> {
    /// The block's values.
    pub values: Values<'a>,
    /// The validity of the block's rows. An op that does not change validity leaves it as it is.
    pub validity: BlockValidity<'a>,
}

impl<'a> Block<'a> {
    /// The number of rows in the block.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the block is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Write `f(value)` for every value, leaving validity as it is. See [`Values::map`].
    #[inline]
    pub fn map<S: NativePType, T: NativePType>(self, f: impl Fn(S) -> T) -> Written<'a> {
        self.values.map(f)
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
    /// In the `scratch` the program handed it, as `len` values of the source's type starting at
    /// value `offset`. The program may overwrite them in place.
    ///
    /// Built with the unsafe [`SourceBlock::scratch`], since the program reads those values as
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

/// The start of a program: emits an array's own values, block by block, in row order.
pub trait BlockSource: Send {
    /// Emit exactly the values of `rows`, at most [`BLOCK_LEN`] of them, written into `scratch`
    /// or lent from memory the source owns.
    ///
    /// `validity` holds the rows' validity as the program was given it. A source that changes it,
    /// such as one applying patches, updates it here.
    ///
    /// The program asks for consecutive ranges in row order, starting at 0, and stops once they
    /// cover the array. Values written into `scratch` belong to the program once returned: it
    /// may overwrite them, so a source must not keep state in `scratch` between calls.
    fn fill<'s>(
        &'s mut self,
        rows: Range<usize>,
        scratch: &'s mut BlockBuf,
        validity: &mut BlockValidity<'_>,
    ) -> VortexResult<SourceBlock<'s>>;

    /// The row at which this source would start its second block if it chose its own blocks,
    /// modulo [`BLOCK_LEN`]: a sliced BitPacked array's offset into its first chunk, so its blocks
    /// line up with its chunks. The program cuts blocks at these boundaries.
    fn block_offset(&self) -> usize {
        0
    }

    /// The source's values as a buffer the program may overwrite in place, if the source holds
    /// the only reference to them: a canonical array's own buffer. The program asks at most once,
    /// before any [`fill`](Self::fill), and fills the source no more once it returns `Some`, so a
    /// source that returns one emits its values and validity unchanged.
    fn take_buffer(&mut self) -> Option<ByteBufferMut> {
        None
    }
}

/// A transform of one block of values into another: of their rows' values, and of their
/// validity where it changes it.
pub trait BlockOp: Send {
    /// Write the op's values for rows `start..start + block.len()`, and update their validity in
    /// `block.validity` if the op changes it.
    ///
    /// Most ops only call [`Block::map`], which handles both shapes of [`Values`].
    fn process<'a>(&mut self, start: usize, block: Block<'a>) -> VortexResult<Written<'a>>;

    /// Whether the program may hand this op [`Values::InPlace`] blocks. An op that reads values
    /// other than the one it is writing must return `false`.
    fn supports_in_place(&self) -> bool {
        true
    }
}

/// A canonical primitive array's values as a source: lends blocks of its buffer, and gives the
/// buffer itself to a program that decodes it in place when nothing else references it.
pub struct PrimitiveSource<T> {
    values: Buffer<T>,
}

impl<T: NativePType> PrimitiveSource<T> {
    /// A source over `values`.
    pub fn new(values: Buffer<T>) -> Self {
        Self { values }
    }
}

impl PrimitiveSource<()> {
    /// A source over the values of `array`, in its own type.
    pub fn boxed(array: &PrimitiveArray) -> Box<dyn BlockSource> {
        match_each_native_ptype!(array.ptype(), |T| { primitive_source::<T>(array) })
    }
}

fn primitive_source<T: NativePType>(array: &PrimitiveArray) -> Box<dyn BlockSource> {
    Box::new(PrimitiveSource::new(array.to_buffer::<T>()))
}

impl<T: NativePType> BlockSource for PrimitiveSource<T> {
    fn fill<'s>(
        &'s mut self,
        rows: Range<usize>,
        _scratch: &'s mut BlockBuf,
        _validity: &mut BlockValidity<'_>,
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

/// Patches resolved once and applied block by block in row order: each patched row takes the
/// patch's value and the patch's validity.
pub struct BlockPatches<T> {
    /// The patched rows, in increasing order, with their values and whether they are valid.
    patches: Vec<(usize, T, bool)>,
    /// The first patch not yet applied.
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
        vortex_ensure_eq!(
            indices.len(),
            values.len(),
            "patch indices and values differ in length"
        );
        let valid = values.validity()?.execute_mask(values.len(), ctx)?;
        let offset = patches.offset();
        let patches = match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
            collect_patches(
                indices.as_slice::<I>(),
                values.as_slice::<S>(),
                &valid,
                offset,
                &map,
            )
        });
        Ok(Self { patches, cursor: 0 })
    }

    /// Overwrite the patched rows of a block that holds the rows from `start`: their values in
    /// `values`, and their validity in `validity`.
    ///
    /// Blocks must be applied in row order.
    #[inline]
    pub fn apply(&mut self, start: usize, values: &mut [T], validity: &mut BlockValidity<'_>) {
        let end = start + values.len();
        while let Some(&(row, value, valid)) = self.patches.get(self.cursor) {
            if row >= end {
                break;
            }
            if row >= start {
                values[row - start] = value;
                validity.set(row - start, valid);
            }
            self.cursor += 1;
        }
    }
}

/// Each patch's row, `map(row, value)`, and validity.
fn collect_patches<I: AsPrimitive<usize>, S: Copy, T>(
    indices: &[I],
    values: &[S],
    valid: &Mask,
    offset: usize,
    map: &impl Fn(usize, S) -> T,
) -> Vec<(usize, T, bool)> {
    indices
        .iter()
        .zip(values)
        .enumerate()
        .map(|(i, (&index, &value))| {
            let row = index.as_() - offset;
            (row, map(row, value), valid.value(i))
        })
        .collect()
}
