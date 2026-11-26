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
//! when an op changes their width. Validity is read from the mask the program was given; an op
//! that changes a row's validity writes it into the output's bitmap, which exists only once a
//! row changes, so ops that leave validity alone pass the mask through untouched.
//!
//! Blocks are borrowed bytes and a [`PType`]. Viewing them as values is the job of `cast`, which
//! holds the runtime's reinterpretations of memory.

use std::mem::MaybeUninit;
use std::ops::Range;

use num_traits::AsPrimitive;
use vortex_buffer::BitBufferMut;
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
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::patches::Patches;

mod cast;
mod program;
#[cfg(test)]
mod tests;

pub use program::BlockProgram;

/// The most rows a block holds.
pub const BLOCK_LEN: usize = 1024;

/// Scratch for one block of values of any primitive type.
#[repr(C, align(64))]
pub struct BlockBuf([u8; BLOCK_LEN * 8]);

impl BlockBuf {
    /// Zeroed scratch for one block.
    pub(crate) fn new_boxed() -> Box<Self> {
        Box::new(Self([0; BLOCK_LEN * 8]))
    }

    /// The scratch as `len` values of `T`, to fill.
    ///
    /// # Panics
    ///
    /// Panics if `len` is more than [`BLOCK_LEN`].
    pub fn values_mut<T: NativePType>(&mut self, len: usize) -> &mut [T] {
        assert!(len <= BLOCK_LEN, "a block holds at most {BLOCK_LEN} rows");
        &mut cast::values_mut(&mut self.0)[..len]
    }
}

/// A block of values of one primitive type.
#[derive(Clone, Copy)]
pub struct BlockIn<'a> {
    bytes: &'a [u8],
    len: usize,
    ptype: PType,
}

impl<'a> BlockIn<'a> {
    /// A block over `values`.
    pub(crate) fn from_slice<T: NativePType>(values: &'a [T]) -> Self {
        assert!(
            values.len() <= BLOCK_LEN,
            "a block holds at most {BLOCK_LEN} rows"
        );
        Self {
            bytes: cast::bytes(values),
            len: values.len(),
            ptype: T::PTYPE,
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
        cast::values(self.bytes)
    }
}

/// Memory a [`BlockOp`] writes one block of values into, row by row from the first.
pub struct BlockOut<'a> {
    bytes: &'a mut [MaybeUninit<u8>],
    len: usize,
    ptype: PType,
    /// The rows written so far.
    filled: usize,
}

impl<'a> BlockOut<'a> {
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

    /// Write `f(value)` for each of `input` into the block's next `input.len()` rows.
    ///
    /// # Panics
    ///
    /// Panics if `T` is not the type the block must hold, or `input` runs past the block.
    #[inline]
    pub fn push_map<S: NativePType, T: NativePType>(&mut self, input: &[S], f: impl Fn(S) -> T) {
        assert_eq!(T::PTYPE, self.ptype, "block holds {}", self.ptype);
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
            ptype: self.ptype,
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
    pub unsafe fn fill_with<T: NativePType>(
        self,
        f: impl FnOnce(&mut [MaybeUninit<T>]),
    ) -> Written<'a> {
        assert_eq!(T::PTYPE, self.ptype, "block holds {}", self.ptype);
        f(cast::uninit_values_mut::<T>(self.bytes));
        Written {
            // SAFETY: the caller guarantees `f` initialized every value.
            bytes: unsafe { cast::assume_init_mut(self.bytes) },
            len: self.len,
            ptype: self.ptype,
        }
    }
}

/// A block an op overwrites: it reads each value as one type and writes it back as another of the
/// same width.
pub struct BlockInOut<'a> {
    bytes: &'a mut [u8],
    len: usize,
    in_ptype: PType,
    out_ptype: PType,
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
        cast::values_mut(self.bytes)
    }

    /// The block, as the op has overwritten it.
    #[inline]
    pub fn written(self) -> Written<'a> {
        Written {
            bytes: self.bytes,
            len: self.len,
            ptype: self.out_ptype,
        }
    }
}

/// One block of values for a [`BlockOp`] to transform: from an input into separate memory, or in
/// place.
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
    /// `map_into` from the input, or `map_into_in_place` over the block itself, writing each
    /// output's bits over its input when the type changes but the width does not.
    ///
    /// # Panics
    ///
    /// Panics if the block does not read `S` and write `T`.
    #[inline]
    pub fn map<S: NativePType, T: NativePType>(self, f: impl Fn(S) -> T) -> Written<'a> {
        match self {
            Values::Copy { input, mut out } => {
                out.push_map(input.as_slice::<S>(), f);
                out.finish()
                    .vortex_expect("the program sizes a block's output to its input")
            }
            Values::InPlace(block) => {
                assert_eq!(S::PTYPE, block.in_ptype, "block holds {}", block.in_ptype);
                assert_eq!(
                    T::PTYPE,
                    block.out_ptype,
                    "block must hold {}",
                    block.out_ptype
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

/// The validity of one block's rows, as the program's input mask gives it and as ops change it.
pub struct BlockValidity<'a> {
    /// The block's first row in the program.
    start: usize,
    len: usize,
    input: &'a Mask,
    /// The program's output validity, once a row's has changed.
    changed: &'a mut Option<BitBufferMut>,
}

impl<'a> BlockValidity<'a> {
    fn new(
        start: usize,
        len: usize,
        input: &'a Mask,
        changed: &'a mut Option<BitBufferMut>,
    ) -> Self {
        Self {
            start,
            len,
            input,
            changed,
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

    /// Whether `row`, counted from the block's first row, is valid.
    #[inline]
    pub fn is_valid(&self, row: usize) -> bool {
        assert!(row < self.len, "row {row} outside a block of {}", self.len);
        match &*self.changed {
            Some(bits) => bits.value(self.start + row),
            None => self.input.value(self.start + row),
        }
    }

    /// Make `row`, counted from the block's first row, valid or null.
    #[inline]
    pub fn set(&mut self, row: usize, valid: bool) {
        if self.is_valid(row) == valid {
            return;
        }
        let input = self.input;
        self.changed
            .get_or_insert_with(|| match input {
                Mask::AllTrue(len) => BitBufferMut::new_set(*len),
                Mask::AllFalse(len) => BitBufferMut::new_unset(*len),
                Mask::Values(values) => BitBufferMut::copy_from(values.bit_buffer()),
            })
            .set_to(self.start + row, valid);
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

/// A block whose every value an op has written, which [`BlockOp::process`] returns.
pub struct Written<'a> {
    bytes: &'a mut [u8],
    len: usize,
    ptype: PType,
}

impl Written<'_> {
    /// The written values, for example to patch them.
    ///
    /// # Panics
    ///
    /// Panics if the block does not hold `T`.
    #[inline]
    pub fn as_mut_slice<T: NativePType>(&mut self) -> &mut [T] {
        assert_eq!(T::PTYPE, self.ptype, "block holds {}", self.ptype);
        cast::values_mut(self.bytes)
    }
}

/// Where a [`BlockSource`] put the values of a block.
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
