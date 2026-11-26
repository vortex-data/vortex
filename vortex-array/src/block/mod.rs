// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block-at-a-time execution of primitive encodings.
//!
//! A [`BlockProgram`] reads an array's values from a [`BlockSource`] a block at a time, at most
//! [`BLOCK_LEN`] rows, and passes each block through a list of [`BlockOp`]s while it is in L1.
//! Only the last op writes full-length memory: a new buffer, or the source's own buffer when it
//! owns it outright.
//!
//! A [`Block`] holds its rows' values and their validity. Values move between two scratch blocks
//! when an op changes their width. Validity starts as the source's, and an op that changes it
//! edits the block's words; the output's validity is the source's mask unless an op edits a block,
//! so ops that leave validity alone pass the mask through untouched.
//!
//! Blocks are borrowed bytes and a [`BlockType`]. Viewing them as values is the job of `cast`, which
//! holds the runtime's reinterpretations of memory.

use std::cell::RefCell;
use std::mem::MaybeUninit;
use std::ops::Range;

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_compute::lane_kernels::IndexedSinkExt;
use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_mask::Mask;

use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::patches::Patches;

mod cast;
mod chain;
#[cfg(test)]
mod executor_tests;
mod program;
#[cfg(test)]
mod tests;
mod validity;
mod value;

pub(crate) use chain::FusedChain;
pub(crate) use chain::fusable_ptype;
pub use program::BlockProgram;
pub use validity::BlockValidity;
pub use value::BlockType;
pub use value::BlockValue;
pub(crate) use value::sealed;

/// The most rows a block holds.
pub const BLOCK_LEN: usize = 1024;

/// Scratch for one block of values of any primitive type.
#[repr(C, align(64))]
pub struct BlockBuf([u8; BLOCK_LEN * 8]);

thread_local! {
    /// Scratch that finished programs returned, so the next programs on this thread reuse it
    /// instead of zeroing new blocks.
    static SPARE_SCRATCH: RefCell<Vec<Box<BlockBuf>>> = const { RefCell::new(Vec::new()) };
}

impl BlockBuf {
    /// Scratch for one block, holding whatever a previous program left in it.
    pub(crate) fn take() -> Box<Self> {
        SPARE_SCRATCH
            .with_borrow_mut(Vec::pop)
            .unwrap_or_else(|| Box::new(Self([0; BLOCK_LEN * 8])))
    }

    /// Return scratch for a later program on this thread to [`take`](Self::take).
    pub(crate) fn give_back(scratch: Box<Self>) {
        SPARE_SCRATCH.with_borrow_mut(|spare| {
            if spare.len() < 2 {
                spare.push(scratch);
            }
        });
    }

    /// The scratch as `len` values of `T`, to fill.
    ///
    /// # Panics
    ///
    /// Panics if `len` is more than [`BLOCK_LEN`].
    pub fn values_mut<T: BlockValue>(&mut self, len: usize) -> &mut [T] {
        assert!(len <= BLOCK_LEN, "a block holds at most {BLOCK_LEN} rows");
        &mut cast::values_mut(&mut self.0)[..len]
    }
}

/// A block of values of one primitive type.
#[derive(Clone, Copy)]
pub struct BlockIn<'a> {
    bytes: &'a [u8],
    len: usize,
    ty: BlockType,
}

impl<'a> BlockIn<'a> {
    /// A block over `values`.
    pub(crate) fn from_slice<T: BlockValue>(values: &'a [T]) -> Self {
        assert!(
            values.len() <= BLOCK_LEN,
            "a block holds at most {BLOCK_LEN} rows"
        );
        Self {
            bytes: cast::bytes(values),
            len: values.len(),
            ty: T::BLOCK_TYPE,
        }
    }

    /// The block's values.
    ///
    /// # Panics
    ///
    /// Panics if `T` is not the block's type.
    #[inline]
    pub fn as_slice<T: BlockValue>(&self) -> &'a [T] {
        assert_eq!(T::BLOCK_TYPE, self.ty, "block holds {}", self.ty);
        cast::values(self.bytes)
    }
}

/// Memory a [`BlockOp`] writes one block of values into, row by row from the first.
pub struct BlockOut<'a> {
    bytes: &'a mut [MaybeUninit<u8>],
    len: usize,
    ty: BlockType,
    /// The rows written so far.
    filled: usize,
}

impl<'a> BlockOut<'a> {
    /// Write `f(value)` for each of `input` into the block's next `input.len()` rows.
    ///
    /// # Panics
    ///
    /// Panics if `T` is not the type the block must hold, or `input` runs past the block.
    #[inline]
    pub fn push_map<S: BlockValue, T: BlockValue>(&mut self, input: &[S], f: impl Fn(S) -> T) {
        assert_eq!(T::BLOCK_TYPE, self.ty, "block holds {}", self.ty);
        let rows = self.filled..self.filled + input.len();
        input.map_into(&mut cast::uninit_values_mut::<T>(self.bytes)[rows], f);
        self.filled += input.len();
    }

    /// The written block, once every row was pushed.
    #[inline]
    pub fn finish(self) -> VortexResult<Written<'a>> {
        vortex_ensure_eq!(self.filled, self.len, "a block op wrote too few rows");
        Ok(Written {
            // SAFETY: `push_map` wrote rows `0..filled`, which is every row.
            bytes: unsafe { cast::assume_init_mut(self.bytes) },
            len: self.len,
            ty: self.ty,
        })
    }

    /// Write the block with `f`, for an op whose kernel fills uninitialized memory itself.
    ///
    /// # Safety
    ///
    /// `f` must initialize every value it is handed.
    ///
    /// # Panics
    ///
    /// Panics if `T` is not the type the block must hold.
    #[inline]
    pub unsafe fn fill_with<T: BlockValue>(
        self,
        f: impl FnOnce(&mut [MaybeUninit<T>]),
    ) -> Written<'a> {
        assert_eq!(T::BLOCK_TYPE, self.ty, "block holds {}", self.ty);
        f(cast::uninit_values_mut::<T>(self.bytes));
        Written {
            // SAFETY: the caller guarantees `f` initialized every value.
            bytes: unsafe { cast::assume_init_mut(self.bytes) },
            len: self.len,
            ty: self.ty,
        }
    }
}

/// A block an op overwrites: it reads each value as one type and writes it back as another of the
/// same width.
pub struct BlockInOut<'a> {
    bytes: &'a mut [u8],
    len: usize,
    in_type: BlockType,
    out_type: BlockType,
}

impl<'a> BlockInOut<'a> {
    /// The block's values, for an op whose input and output types are both `T`.
    ///
    /// # Panics
    ///
    /// Panics if the block does not read and write `T`.
    #[inline]
    pub fn as_mut_slice<T: BlockValue>(&mut self) -> &mut [T] {
        assert_eq!(T::BLOCK_TYPE, self.in_type, "block holds {}", self.in_type);
        assert_eq!(
            T::BLOCK_TYPE,
            self.out_type,
            "block must hold {}",
            self.out_type
        );
        cast::values_mut(self.bytes)
    }

    /// The block, as the op has overwritten it.
    #[inline]
    pub fn written(self) -> Written<'a> {
        Written {
            bytes: self.bytes,
            len: self.len,
            ty: self.out_type,
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
    /// Write `f(value)` for every value, with the lane kernels level-wise execution uses:
    /// `map_into` from the input, or `map_into_in_place` over the block itself, writing each
    /// output's bits over its input when the type changes but the width does not.
    ///
    /// # Panics
    ///
    /// Panics if the block does not read `S` and write `T`.
    #[inline]
    pub fn map<S: BlockValue, T: BlockValue>(self, f: impl Fn(S) -> T) -> Written<'a> {
        match self {
            Values::Copy { input, mut out } => {
                out.push_map(input.as_slice::<S>(), f);
                out.finish()
                    .vortex_expect("the program sizes a block's output to its input")
            }
            Values::InPlace(block) => {
                assert_eq!(
                    S::BLOCK_TYPE,
                    block.in_type,
                    "block holds {}",
                    block.in_type
                );
                assert_eq!(
                    T::BLOCK_TYPE,
                    block.out_type,
                    "block must hold {}",
                    block.out_type
                );
                // Each value is overwritten with the bits of its output, which `written` then
                // views as `T`.
                cast::values_mut::<S>(block.bytes)
                    .map_into_in_place(|value| cast::bitcast::<T, S>(f(value)));
                block.written()
            }
        }
    }
}

/// One block for a [`BlockOp`]: its rows' values, and their validity.
pub struct Block<'a> {
    /// The block's values.
    pub values: Values<'a>,
    /// The validity of the block's rows. An op that does not change validity leaves it as it is.
    pub validity: BlockValidity<'a>,
    start: usize,
}

impl<'a> Block<'a> {
    /// The row of the block's first value.
    #[inline]
    pub fn start(&self) -> usize {
        self.start
    }

    /// Write `f(value)` for every value, leaving validity as it is. See [`Values::map`].
    #[inline]
    pub fn map<S: BlockValue, T: BlockValue>(self, f: impl Fn(S) -> T) -> Written<'a> {
        self.values.map(f)
    }
}

/// A block whose every value an op has written, which [`BlockOp::process`] returns.
pub struct Written<'a> {
    bytes: &'a mut [u8],
    len: usize,
    ty: BlockType,
}

impl Written<'_> {
    /// The written values, for example to patch them.
    ///
    /// # Panics
    ///
    /// Panics if the block does not hold `T`.
    #[inline]
    pub fn as_mut_slice<T: BlockValue>(&mut self) -> &mut [T] {
        assert_eq!(T::BLOCK_TYPE, self.ty, "block holds {}", self.ty);
        cast::values_mut(self.bytes)
    }
}

/// Where a [`BlockSource`] put the values of a block.
#[non_exhaustive]
pub enum SourceBlock<'s> {
    /// In the scratch the program handed it, as `len` values of the source's type starting at
    /// value `offset`. The program may overwrite them in place.
    Scratch {
        /// The index of the first value in the scratch.
        offset: usize,
        /// The number of values.
        len: usize,
    },
    /// In memory the source owns, which ops never overwrite.
    Lent(BlockIn<'s>),
}

/// The start of a program: emits an array's own values, block by block, in row order.
pub trait BlockSource: Send {
    /// The validity of the source's rows, which each block's starts from.
    fn validity(&self) -> &Mask;

    /// Emit exactly the values of `rows`, at most [`BLOCK_LEN`] of them, written into `scratch`
    /// or lent from memory the source owns. A source that changes validity, such as one applying
    /// patches, edits `validity`.
    ///
    /// The program asks for consecutive ranges in row order, starting at 0, and stops once they
    /// cover the array. Values written into `scratch` belong to the program once returned: it
    /// may overwrite them, so a source must not keep state in `scratch` between calls.
    fn fill<'s>(
        &'s mut self,
        rows: Range<usize>,
        scratch: &mut BlockBuf,
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
    /// Write the op's values for the block's rows, and edit their validity in `block.validity`
    /// if the op changes it.
    ///
    /// Most ops only call [`Block::map`], which handles both shapes of [`Values`].
    fn process<'a>(&mut self, block: Block<'a>) -> VortexResult<Written<'a>>;

    /// Whether the program may hand this op [`Values::InPlace`] blocks. An op that reads other
    /// rows of its block than the one it is writing must return `false`.
    fn supports_in_place(&self) -> bool {
        true
    }
}

/// A canonical primitive array's values as a source: lends blocks of its buffer, and gives the
/// buffer itself to a program that decodes it in place when nothing else references it.
pub struct PrimitiveSource<T> {
    values: Buffer<T>,
    validity: Mask,
}

impl<T: NativePType> PrimitiveSource<T> {
    /// A source over `values`, whose rows have `validity`.
    pub fn new(values: Buffer<T>, validity: Mask) -> Self {
        Self { values, validity }
    }
}

impl PrimitiveSource<()> {
    /// A source over the values and validity of `array`, in its own type.
    pub fn boxed(
        array: &PrimitiveArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Box<dyn BlockSource>> {
        let validity = array.validity()?.execute_mask(array.len(), ctx)?;
        Ok(match_each_native_ptype!(array.ptype(), |T| {
            primitive_source::<T>(array, validity)
        }))
    }
}

fn primitive_source<T: NativePType>(
    array: &PrimitiveArray,
    validity: Mask,
) -> Box<dyn BlockSource> {
    Box::new(PrimitiveSource::new(array.to_buffer::<T>(), validity))
}

impl<T: NativePType> BlockSource for PrimitiveSource<T> {
    fn validity(&self) -> &Mask {
        &self.validity
    }

    fn fill<'s>(
        &'s mut self,
        rows: Range<usize>,
        _scratch: &mut BlockBuf,
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

impl<T: BlockValue> BlockPatches<T> {
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
