// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block sources and ops for BitPacked and FoR, to start and extend a
//! [`BlockProgram`](vortex_array::block::BlockProgram).
//!
//! BitPacked emits blocks by unpacking one FastLanes chunk at a time ([`PackedSource`]), with the
//! same [`UnpackStrategy`] the level-wise unpack uses. FoR over BitPacked with one reference is a
//! source too, unpacking with FastLanes' fused unpack-and-add ([`FoRStrategy`]). FoR over any
//! other child adds its references to each block ([`AddReference`], [`AddReferences`]).

use std::ops::Range;

use fastlanes::FoR as FastLanesFoR;
use num_traits::AsPrimitive;
use num_traits::WrappingAdd;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Primitive;
use vortex_array::block::Block;
use vortex_array::block::BlockBuf;
use vortex_array::block::BlockOp;
use vortex_array::block::BlockPatches;
use vortex_array::block::BlockSource;
use vortex_array::block::BlockValidity;
use vortex_array::block::SourceBlock;
use vortex_array::block::Values;
use vortex_array::block::Written;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_buffer::Buffer;
use vortex_compute::lane_kernels::IndexedSinkExt;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::BitWidthsView;
use crate::FL_CHUNK_SIZE;
use crate::FoR;
use crate::FoRArrayExt;
use crate::FoRArraySlotsExt;
use crate::FoRReferences;
use crate::unpack_iter::BitPackingStrategy;
use crate::unpack_iter::UnpackStrategy;
use crate::unpack_iter::validate_packed;

/// BitPacked's block source, for arrays with one global bit width.
pub fn packed_source(
    bp: ArrayView<'_, BitPacked>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Box<dyn BlockSource>>> {
    match_each_integer_ptype!(bp.dtype().as_ptype(), |T| {
        packed_source_with::<T, _>(bp, BitPackingStrategy, |value| value, ctx)
    })
}

/// FoR's block source over a BitPacked child with one global reference: FastLanes' fused
/// unpack adds the reference to each chunk while it unpacks it, as FoR's own execute does. FoR's
/// execute also unpacks fused under per-chunk references that line up with BitPacked's chunks;
/// as a source those take the BitPacked leaf and FoR's per-chunk op instead.
pub fn for_block_source(
    array: ArrayView<'_, FoR>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Box<dyn BlockSource>>> {
    let (Some(bp), FoRReferences::Global(reference)) =
        (array.encoded().as_opt::<BitPacked>(), array.references())
    else {
        return Ok(None);
    };
    match_each_integer_ptype!(bp.dtype().as_ptype(), |T| {
        for_source_typed::<T>(bp, reference, ctx)
    })
}

fn for_source_typed<T>(
    bp: ArrayView<'_, BitPacked>,
    reference: &Scalar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Box<dyn BlockSource>>>
where
    T: PhysicalPType<Physical: FastLanesFoR> + WrappingAdd + AsPrimitive<T::Physical>,
{
    let reference = reference
        .as_primitive()
        .typed_value::<T>()
        .vortex_expect("FoR reference is non-null");
    packed_source_with::<T, _>(
        bp,
        FoRStrategy(reference.as_()),
        |value: T| value.wrapping_add(&reference),
        ctx,
    )
}

/// Unpacks a chunk and adds a reference in one pass.
struct FoRStrategy<P>(P);

impl<T: PhysicalPType<Physical: FastLanesFoR>> UnpackStrategy<T> for FoRStrategy<T::Physical> {
    unsafe fn unpack_chunk(
        &self,
        bit_width: usize,
        chunk: &[T::Physical],
        dst: &mut [T::Physical],
    ) {
        // SAFETY: the caller guarantees the sizes `unchecked_unfor_pack` requires.
        unsafe { T::Physical::unchecked_unfor_pack(bit_width, chunk, self.0, dst) }
    }
}

/// A source unpacking each chunk of `bp` with `strategy`, and patching its values mapped by
/// `patch`.
fn packed_source_with<T, S>(
    bp: ArrayView<'_, BitPacked>,
    strategy: S,
    patch: impl Fn(T) -> T,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Box<dyn BlockSource>>>
where
    T: PhysicalPType,
    S: UnpackStrategy<T> + Send + 'static,
{
    let BitWidthsView::Global(bit_width) = bp.bit_widths() else {
        return Ok(None);
    };
    let bit_width = usize::from(bit_width);
    let offset = usize::from(bp.offset());
    let packed = Buffer::<T::Physical>::from_byte_buffer(bp.packed().as_host().clone());
    validate_packed::<T>(packed.len(), bit_width, offset, bp.len())?;
    let patches = bp
        .patches()
        .map(|patches| BlockPatches::try_new(&patches, ctx, |_, value: T| patch(value)))
        .transpose()?;
    Ok(Some(Box::new(PackedSource::<T, S> {
        strategy,
        packed,
        bit_width,
        offset,
        patches,
    })))
}

/// Unpacks the FastLanes chunks under each requested range with its strategy.
struct PackedSource<T: PhysicalPType, S> {
    strategy: S,
    packed: Buffer<T::Physical>,
    bit_width: usize,
    /// The position of the first row within the first chunk.
    offset: usize,
    patches: Option<BlockPatches<T>>,
}

impl<T: PhysicalPType, S: UnpackStrategy<T>> PackedSource<T, S> {
    /// Unpack chunk `chunk` into `buf` and return its values.
    fn unpack<'o>(&self, chunk: usize, buf: &'o mut BlockBuf) -> &'o mut [T] {
        let elems_per_chunk = 128 * self.bit_width / size_of::<T>();
        let packed = &self.packed[chunk * elems_per_chunk..][..elems_per_chunk];
        let dst = buf.values_mut::<T::Physical>(FL_CHUNK_SIZE);
        // SAFETY: `packed` holds exactly one chunk at `bit_width`, and `dst` exactly one chunk.
        unsafe { self.strategy.unpack_chunk(self.bit_width, packed, dst) };
        buf.values_mut::<T>(FL_CHUNK_SIZE)
    }
}

impl<T: PhysicalPType, S: UnpackStrategy<T> + Send> BlockSource for PackedSource<T, S> {
    fn fill<'s>(
        &'s mut self,
        rows: Range<usize>,
        scratch: &mut BlockBuf,
        validity: &mut BlockValidity<'_>,
    ) -> VortexResult<SourceBlock<'s>> {
        let (first, end) = (self.offset + rows.start, self.offset + rows.end);
        let chunk = first / FL_CHUNK_SIZE;
        let chunk_start = chunk * FL_CHUNK_SIZE;
        // A program cuts its blocks at `block_offset`, so each lies in one chunk: unpack it and
        // point at the block's part of it.
        vortex_ensure!(
            end <= chunk_start + FL_CHUNK_SIZE,
            "rows {rows:?} cross a FastLanes chunk"
        );
        let offset = first - chunk_start;
        let block = &mut self.unpack(chunk, scratch)[offset..end - chunk_start];
        if let Some(patches) = &mut self.patches {
            patches.apply(rows.start, block, validity);
        }
        Ok(SourceBlock::Scratch {
            offset,
            len: rows.len(),
        })
    }

    fn block_offset(&self) -> usize {
        self.offset
    }
}

/// FoR's op over a child of any encoding: add the global reference to every value.
struct AddReference<T> {
    reference: T,
}

impl<T: NativePType + WrappingAdd> BlockOp for AddReference<T> {
    fn process<'a>(&mut self, _start: usize, block: Block<'a>) -> VortexResult<Written<'a>> {
        let reference = self.reference;
        Ok(block.map(|value: T| value.wrapping_add(&reference)))
    }
}

/// FoR's op over a child of any encoding: add each chunk's reference to its rows.
struct AddReferences<T> {
    references: Buffer<T>,
    offset: usize,
}

impl<T: NativePType + WrappingAdd> AddReferences<T> {
    /// Call `f(rows, reference)` for each run of the block's rows `start..start + len` that falls
    /// in one FoR chunk, with `rows` counted from the block's start.
    fn for_each_chunk(&self, start: usize, len: usize, mut f: impl FnMut(Range<usize>, T)) {
        let mut row = 0;
        while row < len {
            let position = self.offset + start + row;
            let run = (FL_CHUNK_SIZE - position % FL_CHUNK_SIZE).min(len - row);
            f(row..row + run, self.references[position / FL_CHUNK_SIZE]);
            row += run;
        }
    }
}

impl<T: NativePType + WrappingAdd> BlockOp for AddReferences<T> {
    fn process<'a>(&mut self, start: usize, block: Block<'a>) -> VortexResult<Written<'a>> {
        // A block may straddle a FoR chunk boundary, so add each chunk's reference to its rows.
        match block.values {
            Values::Copy { input, mut out } => {
                let input = input.as_slice::<T>();
                self.for_each_chunk(start, input.len(), |rows, reference| {
                    out.push_map(&input[rows], |v| v.wrapping_add(&reference));
                });
                out.finish()
            }
            Values::InPlace(mut block) => {
                let values = block.as_mut_slice::<T>();
                self.for_each_chunk(start, values.len(), |rows, reference| {
                    values[rows].map_into_in_place(|v| v.wrapping_add(&reference));
                });
                Ok(block.written())
            }
        }
    }
}

/// FoR's op over its encoded child of any encoding: add the global reference, or each chunk's
/// reference once the blocked references are canonical. `None` for a zero reference, which
/// changes nothing, and for blocked references not yet executed.
pub fn for_block_op(array: ArrayView<'_, FoR>) -> VortexResult<Option<Box<dyn BlockOp>>> {
    Ok(match_each_integer_ptype!(array.dtype().as_ptype(), |T| {
        for_block_op_typed::<T>(array)
    }))
}

fn for_block_op_typed<T: NativePType + WrappingAdd>(
    array: ArrayView<'_, FoR>,
) -> Option<Box<dyn BlockOp>> {
    match array.references() {
        FoRReferences::Global(reference) => {
            let reference = reference
                .as_primitive()
                .typed_value::<T>()
                .vortex_expect("FoR reference is non-null");
            // Level-wise execution returns the child as it is, without a copy.
            (reference != T::default())
                .then(|| Box::new(AddReference { reference }) as Box<dyn BlockOp>)
        }
        FoRReferences::Blocked(references) => references.as_opt::<Primitive>().map(|references| {
            Box::new(AddReferences {
                references: references.to_buffer::<T>(),
                offset: usize::from(array.offset()),
            }) as Box<dyn BlockOp>
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::block::BlockProgram;
    use vortex_array::block::PrimitiveSource;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::for_block_op;
    use super::for_block_source;
    use super::packed_source;
    use crate::BitPacked;
    use crate::FoR;
    use crate::FoRArraySlotsExt;
    use crate::bitpack_compress::bitpack_encode;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    /// Values below 1000, with every 97th above it and so patched at width 10, nullable or not.
    fn values(len: usize, nullable: bool) -> PrimitiveArray {
        let values = (0..len as u32).map(|i| {
            if i % 97 == 0 {
                5000 + i
            } else {
                (i * 7) % 1000
            }
        });
        if nullable {
            PrimitiveArray::from_option_iter(
                values.enumerate().map(|(i, v)| (i % 13 != 0).then_some(v)),
            )
        } else {
            PrimitiveArray::from_iter(values)
        }
    }

    /// `values` bit-packed at width 10, then sliced to `rows` if given.
    fn bitpacked(
        len: usize,
        nullable: bool,
        rows: Option<(usize, usize)>,
    ) -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        let bp = bitpack_encode(&values(len, nullable), 10, None, &mut ctx)?.into_array();
        Ok(match rows {
            Some((start, end)) => bp.slice(start..end)?.execute::<ArrayRef>(&mut ctx)?,
            None => bp,
        })
    }

    /// `depth` FoR levels over `leaf`, with references 1000, 1001, and so on.
    fn for_stack(leaf: ArrayRef, depth: usize) -> VortexResult<ArrayRef> {
        let mut array = leaf;
        for level in 0..depth {
            array = FoR::try_new(array, Scalar::from(1_000u32 + level as u32))?.into_array();
        }
        Ok(array)
    }

    /// The FoR levels of `array` from the innermost out, and the array below them.
    fn levels(array: &ArrayRef) -> (Vec<ArrayRef>, ArrayRef) {
        let mut levels = Vec::new();
        let mut current = array.clone();
        while let Some(for_) = current.as_opt::<FoR>() {
            let encoded = for_.encoded().clone();
            levels.push(current);
            current = encoded;
        }
        levels.reverse();
        (levels, current)
    }

    /// A program for a FoR stack: its leaf's source, then an op for each FoR level above the leaf.
    /// With `fused_leaf`, the innermost FoR over BitPacked is the leaf, through FoR's fused unpack.
    fn program(array: &ArrayRef, fused_leaf: bool) -> VortexResult<(BlockProgram, ArrayRef)> {
        let mut ctx = SESSION.create_execution_ctx();
        let (levels, bottom) = levels(array);
        let (leaf, ops) = match (fused_leaf, levels.split_first()) {
            (true, Some((innermost, above))) => (innermost.clone(), above.to_vec()),
            _ => (bottom, levels),
        };
        let source = if let Some(for_) = leaf.as_opt::<FoR>() {
            for_block_source(for_, &mut ctx)?
        } else if let Some(bp) = leaf.as_opt::<BitPacked>() {
            packed_source(bp, &mut ctx)?
        } else {
            Some(PrimitiveSource::boxed(
                &leaf.clone().execute::<PrimitiveArray>(&mut ctx)?,
            ))
        }
        .vortex_expect("a leaf with a source");
        let ptype = leaf.dtype().as_ptype();
        let mut program = BlockProgram::new(source, ptype, leaf.len());
        for level in &ops {
            let op = for_block_op(level.as_::<FoR>())?.vortex_expect("a FoR op");
            program.push(op, level.dtype().as_ptype());
        }
        Ok((program, leaf))
    }

    /// Run `array`'s program and check it against executing `array` level-wise.
    fn assert_program_matches(array: &ArrayRef, fused_leaf: bool) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let (program, leaf) = program(array, fused_leaf)?;
        let validity = leaf.validity()?.execute_mask(leaf.len(), &mut ctx)?;
        let output = program.run(&validity, array.dtype().nullability(), &mut ctx)?;
        let expected = array.clone().execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(output, expected, &mut ctx);
        Ok(())
    }

    /// FoR stacks over BitPacked, with patches and nulls, from the start of a chunk or part way
    /// in, with FoR's fused unpack as the leaf or BitPacked's unpack under FoR ops.
    #[rstest]
    fn for_stacks_match_level_wise(
        #[values(1, 2, 3, 8)] depth: usize,
        #[values(false, true)] nullable: bool,
        #[values(None, Some((517, 4013)))] rows: Option<(usize, usize)>,
        #[values(false, true)] fused_leaf: bool,
    ) -> VortexResult<()> {
        let array = for_stack(bitpacked(5000, nullable, rows)?, depth)?;
        assert_program_matches(&array, fused_leaf)
    }

    #[test]
    fn signed_values_with_negative_references() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0..3000i64).map(|i| (i * 31) % 4000));
        let bp = bitpack_encode(&values, 12, None, &mut ctx)?.into_array();
        let inner = FoR::try_new(bp, Scalar::from(-1_000_000i64))?.into_array();
        let outer = FoR::try_new(inner, Scalar::from(-7i64))?.into_array();
        assert_program_matches(&outer, true)?;
        assert_program_matches(&outer, false)
    }

    /// Per-chunk references over a BitPacked child that starts part way into a chunk: FoR has no
    /// fused source then, so BitPacked's unpack is the leaf and FoR adds each chunk's reference.
    #[test]
    fn blocked_references_over_misaligned_bitpacked() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0..4096u32).map(|i| i % 300));
        let bp = bitpack_encode(&values, 9, None, &mut ctx)?
            .into_array()
            .slice(100..3172)?
            .execute::<ArrayRef>(&mut ctx)?;
        let references = PrimitiveArray::new(buffer![10u32, 20, 30], Validity::NonNullable);
        let array = FoR::try_new_chunked(bp, references.into_array(), 0)?.into_array();
        assert!(for_block_source(array.as_::<FoR>(), &mut ctx)?.is_none());
        assert_program_matches(&array, false)?;
        let outer = FoR::try_new(array, Scalar::from(5u32))?.into_array();
        assert_program_matches(&outer, false)
    }

    /// A FoR stack over a canonical leaf, decoded in place when nothing else holds the leaf.
    #[test]
    fn for_stack_over_a_canonical_leaf() -> VortexResult<()> {
        let leaf = PrimitiveArray::from_iter((0..3000u32).map(|i| i * 3)).into_array();
        assert_program_matches(&for_stack(leaf, 3)?, false)
    }

    /// A zero reference changes nothing, so FoR has no op for it.
    #[test]
    fn zero_reference_has_no_op() -> VortexResult<()> {
        let array = FoR::try_new(bitpacked(3000, false, None)?, Scalar::from(0u32))?;
        assert!(for_block_op(array.as_view())?.is_none());
        Ok(())
    }
}
