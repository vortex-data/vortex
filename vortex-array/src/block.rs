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
use vortex_compute::lane_kernels::IndexedSourceExt;
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

    /// Write `f(value)` for every value of `input`, with the lane kernel level-wise execution maps
    /// whole buffers with, and return the written values.
    ///
    /// # Panics
    ///
    /// Panics if `input` does not hold `S`, the block does not hold `T`, or their lengths differ.
    #[inline]
    pub fn map_from<S: NativePType, T: NativePType>(
        &mut self,
        input: BlockIn<'_>,
        f: impl Fn(S) -> T,
    ) -> &mut [T] {
        let out = self.as_uninit::<T>();
        input.as_slice::<S>().map_into(out, f);
        // SAFETY: `map_into` wrote every value of `out`.
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast(), out.len()) }
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
    /// Write the parent's values for `input.len()` rows starting at row `start`.
    ///
    /// `input` holds the child's values for those rows, in the child's primitive type, and `out`
    /// receives the parent's, in the parent's primitive type. Every value of `out` must be written.
    fn process(&mut self, start: usize, input: BlockIn<'_>, out: BlockOut<'_>) -> VortexResult<()>;
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

/// Run `source` through `ops`, from the leaf up, into a buffer of `len` values of `out_ptype`.
///
/// `ptypes[i]` is the type `ops[i]` reads and `ptypes[i + 1]` the type it writes.
pub(crate) fn run_fused(
    source: &mut dyn BlockSource,
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
    match_each_native_ptype!(out_ptype, |T| {
        let mut output = BufferMut::<T>::with_capacity_in(len, ctx.allocator().clone());
        let dst = output.spare_capacity_mut().as_mut_ptr().cast::<u8>();
        // SAFETY: `dst` has room for `len` values of `T`, the type of `ptypes.last()`.
        unsafe { run_blocks(source, ops, ptypes, len, dst)? };
        // SAFETY: `run_blocks` wrote every one of the `len` values.
        unsafe { output.set_len(len) };
        Ok(PrimitiveArray::new(output.freeze(), validity))
    })
}

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
    let mut scratch = BlockBuf::new_boxed();
    let mut ping = BlockBuf::new_boxed();
    let mut pong = BlockBuf::new_boxed();
    let out_width = ptypes[ops.len()].byte_width();
    let last = ops.len() - 1;
    let mut start = 0;
    while start < len {
        let block = source.next_block(&mut scratch)?;
        let n = block.len();
        vortex_ensure!(
            n > 0 && start + n <= len,
            "block source emitted {n} rows at row {start} of {len}"
        );
        vortex_ensure_eq!(block.ptype(), ptypes[0]);
        let mut input = (block.ptr, block.ptype);
        for (i, op) in ops.iter_mut().enumerate() {
            let out_ptr = if i == last {
                // SAFETY: `start + n <= len`, so the block fits in `dst`.
                unsafe { dst.add(start * out_width) }
            } else if i % 2 == 0 {
                ping.as_mut_ptr()
            } else {
                pong.as_mut_ptr()
            };
            let out = BlockOut {
                ptr: out_ptr,
                len: n,
                ptype: ptypes[i + 1],
                _borrow: PhantomData,
            };
            let block_in = BlockIn {
                ptr: input.0,
                len: n,
                ptype: input.1,
                _borrow: PhantomData,
            };
            // Op `i` reads what op `i - 1` wrote (or the source's block) and writes the other
            // scratch buffer, so its input and output never alias.
            op.process(start, block_in, out)?;
            input = (out_ptr.cast_const(), ptypes[i + 1]);
        }
        start += n;
    }
    Ok(())
}

/// Converts each value with `as`, for casts that lose nothing.
struct Convert<S, T>(PhantomData<fn(S) -> T>);

impl<S, T> BlockOp for Convert<S, T>
where
    S: NativePType + AsPrimitive<T>,
    T: NativePType,
{
    fn process(
        &mut self,
        _start: usize,
        input: BlockIn<'_>,
        mut out: BlockOut<'_>,
    ) -> VortexResult<()> {
        out.map_from(input, |value: S| value.as_());
        Ok(())
    }
}

macro_rules! match_each_convertible_ptype {
    ($ptype:expr, | $T:ident | $body:expr) => {
        match $ptype {
            PType::U8 => {
                type $T = u8;
                $body
            }
            PType::U16 => {
                type $T = u16;
                $body
            }
            PType::U32 => {
                type $T = u32;
                $body
            }
            PType::U64 => {
                type $T = u64;
                $body
            }
            PType::I8 => {
                type $T = i8;
                $body
            }
            PType::I16 => {
                type $T = i16;
                $body
            }
            PType::I32 => {
                type $T = i32;
                $body
            }
            PType::I64 => {
                type $T = i64;
                $body
            }
            PType::F32 => {
                type $T = f32;
                $body
            }
            PType::F64 => {
                type $T = f64;
                $body
            }
            PType::F16 => None,
        }
    };
}

/// Whether a cast from `from` to `to` cannot fail and has a block op: a lossless primitive
/// conversion that does not drop nullability.
pub(crate) fn casts_by_block(from: &DType, to: &DType) -> bool {
    let (
        DType::Primitive(from_ptype, from_nullability),
        DType::Primitive(to_ptype, to_nullability),
    ) = (from, to)
    else {
        return false;
    };
    (to_nullability.is_nullable() || !from_nullability.is_nullable())
        && *from_ptype != PType::F16
        && *to_ptype != PType::F16
        && from_ptype.casts_losslessly_to(*to_ptype)
}

/// The op of a cast from `from` to `to` that cannot fail: a lossless primitive conversion that
/// does not drop nullability. Other casts must check their values, so they do not fuse.
pub(crate) fn cast_block_op(from: &DType, to: &DType) -> Option<Box<dyn BlockOp>> {
    if !casts_by_block(from, to) {
        return None;
    }
    let (DType::Primitive(from_ptype, _), DType::Primitive(to_ptype, _)) = (from, to) else {
        return None;
    };
    match_each_convertible_ptype!(from_ptype, |S| {
        match_each_convertible_ptype!(to_ptype, |T| {
            Some(Box::new(Convert::<S, T>(PhantomData)) as Box<dyn BlockOp>)
        })
    })
}
