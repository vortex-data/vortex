// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block-fused execution for BitPacked and FoR. See [`vortex_array::block`].
//!
//! BitPacked emits blocks by unpacking one FastLanes chunk at a time ([`PackedSource`]), with the
//! same [`UnpackStrategy`] the level-wise unpack uses. FoR over BitPacked with one reference is a
//! source too, unpacking with FastLanes' fused unpack-and-add ([`FoRStrategy`]). FoR over any
//! other child adds its references to each block ([`AddReference`], [`AddReferences`]).

use std::mem::MaybeUninit;
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
use vortex_array::block::SourceBlock;
use vortex_array::block::Written;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::Buffer;
use vortex_compute::lane_kernels::IndexedSinkExt;
use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::BitWidthsView;
use crate::FL_CHUNK_SIZE;
use crate::FoR;
use crate::FoRArrayExt;
use crate::FoRArraySlotsExt;
use crate::FoRReferences;
use crate::FoRSlots;
use crate::unpack_iter::BitPackingStrategy;
use crate::unpack_iter::UnpackStrategy;
use crate::unpack_iter::unpack_chunk_into;
use crate::unpack_iter::validate_packed;

/// BitPacked's block source, for arrays with one global bit width.
pub(crate) fn packed_source(
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
pub(crate) fn for_block_source(
    array: ArrayView<'_, FoR>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Box<dyn BlockSource>>> {
    let (Some(bp), FoRReferences::Global(reference)) =
        (array.encoded().as_opt::<BitPacked>(), array.references())
    else {
        return Ok(None);
    };
    match_each_integer_ptype!(bp.dtype().as_ptype(), |T| {
        let reference = reference
            .as_primitive()
            .typed_value::<T>()
            .vortex_expect("FoR reference is non-null");
        packed_source_with::<T, _>(
            bp,
            FoRStrategy(reference.as_()),
            |value: T| value.wrapping_add(reference),
            ctx,
        )
    })
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
    let patches = match bp.patches() {
        Some(patches) => match BlockPatches::try_new(&patches, ctx, |_, value: T| patch(value))? {
            Some(patches) => Some(patches),
            None => return Ok(None),
        },
        None => None,
    };
    Ok(Some(Box::new(PackedSource::<T, S> {
        strategy,
        packed,
        bit_width,
        offset,
        patches,
        straddle: None,
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
    /// Scratch for ranges that straddle two chunks, allocated on first use.
    straddle: Option<Box<BlockBuf>>,
}

impl<T: PhysicalPType, S: UnpackStrategy<T>> PackedSource<T, S> {
    /// Unpack chunk `chunk` into `buf` and return its values.
    fn unpack<'o>(&self, chunk: usize, buf: &'o mut BlockBuf) -> &'o mut [T] {
        let out = <&mut [MaybeUninit<T>; FL_CHUNK_SIZE]>::try_from(buf.out::<T>(FL_CHUNK_SIZE))
            .ok()
            .vortex_expect("a block holds one FastLanes chunk");
        unpack_chunk_into(&self.strategy, &self.packed, self.bit_width, chunk, out)
    }
}

impl<T: PhysicalPType, S: UnpackStrategy<T> + Send> BlockSource for PackedSource<T, S> {
    fn fill<'s>(
        &'s mut self,
        rows: Range<usize>,
        scratch: &'s mut BlockBuf,
    ) -> VortexResult<SourceBlock<'s>> {
        let (first, end) = (self.offset + rows.start, self.offset + rows.end);
        let chunk = first / FL_CHUNK_SIZE;
        let chunk_start = chunk * FL_CHUNK_SIZE;
        let (block, offset): (&mut [T], usize) = if end <= chunk_start + FL_CHUNK_SIZE {
            // On the source's own grid every range lies in one chunk: unpack it and point at the
            // range's part of it.
            let values = self.unpack(chunk, scratch);
            (
                &mut values[first - chunk_start..end - chunk_start],
                first - chunk_start,
            )
        } else {
            // A range straddling two chunks is assembled from both.
            let mut straddle = self.straddle.take().unwrap_or_else(BlockBuf::new_boxed);
            let out = scratch.out::<T>(rows.len());
            let mut written = 0;
            for chunk in chunk..=(end - 1) / FL_CHUNK_SIZE {
                let chunk_start = chunk * FL_CHUNK_SIZE;
                let values = self.unpack(chunk, &mut straddle);
                let part = &values[first.max(chunk_start) - chunk_start
                    ..end.min(chunk_start + FL_CHUNK_SIZE) - chunk_start];
                out[written..written + part.len()].write_copy_of_slice(part);
                written += part.len();
            }
            self.straddle = Some(straddle);
            // SAFETY: the chunks' parts cover the `rows.len()` values of `out`.
            (
                unsafe { std::mem::transmute::<&mut [MaybeUninit<T>], &mut [T]>(out) },
                0,
            )
        };
        if let Some(patches) = &mut self.patches {
            patches.apply(rows.start, block);
        }
        // SAFETY: the chunk or the straddle holds the `rows.len()` values from `offset`.
        Ok(unsafe { SourceBlock::scratch(offset, rows.len()) })
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
        match block {
            Block::Copy { input, mut out } => {
                let (input, values) = (input.as_slice::<T>(), out.as_uninit::<T>());
                self.for_each_chunk(start, input.len(), |rows, reference| {
                    input[rows.clone()].map_into(&mut values[rows], |v| v.wrapping_add(&reference));
                });
                // SAFETY: the chunk runs cover the block, and each run's values were written.
                Ok(unsafe { Written::assume_init(out) })
            }
            Block::InPlace(mut block) => {
                let values = block.as_mut_slice::<T>();
                self.for_each_chunk(start, values.len(), |rows, reference| {
                    values[rows].map_into_in_place(|v| v.wrapping_add(&reference));
                });
                // SAFETY: the chunk runs cover the block, and each run's values were overwritten.
                Ok(unsafe { block.assume_written() })
            }
            _ => vortex_bail!("FoR adds per-chunk references to copied or in-place blocks only"),
        }
    }
}

/// FoR's op over a child of any encoding, once its blocked references, if any, are executed.
pub(crate) fn for_block_op(
    array: ArrayView<'_, FoR>,
    slot_idx: usize,
) -> VortexResult<Option<Box<dyn BlockOp>>> {
    if slot_idx != FoRSlots::ENCODED {
        return Ok(None);
    }
    match_each_integer_ptype!(array.dtype().as_ptype(), |T| {
        Ok(match array.references() {
            FoRReferences::Global(reference) => {
                let reference = reference
                    .as_primitive()
                    .typed_value::<T>()
                    .vortex_expect("FoR reference is non-null");
                // A zero reference changes nothing, which level-wise execution returns without
                // copying, so it does not fuse.
                (reference != T::default())
                    .then(|| Box::new(AddReference { reference }) as Box<dyn BlockOp>)
            }
            FoRReferences::Blocked(references) => match references.as_opt::<Primitive>() {
                Some(references) => Some(Box::new(AddReferences {
                    references: references.to_buffer::<T>(),
                    offset: usize::from(array.offset()),
                }) as Box<dyn BlockOp>),
                None => None,
            },
        })
    })
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::block::BLOCK_LEN;
    use vortex_array::block::BlockBuf;
    use vortex_array::block::SourceBlock;
    use vortex_array::scalar::Scalar;
    use vortex_array::test_harness::fusion;
    use vortex_array::test_harness::fusion::fused_chains;
    use vortex_array::test_harness::trace::trace_op;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::for_block_source;
    use super::packed_source;
    use crate::BitPacked;
    use crate::FoR;
    use crate::bitpack_compress::bitpack_encode;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    /// Execute `array` with and without fusion, check the results agree, and how many chains
    /// fused.
    fn assert_fuses_like_execute(array: &ArrayRef, chains: usize) -> VortexResult<()> {
        let lines = fusion_trace(array.clone())?;
        assert_eq!(
            fused_chains(&lines),
            chains,
            "fused chains for\n{}",
            array.display_tree()
        );
        Ok(())
    }

    /// `depth` FoR levels over a BitPacked leaf at width 10, whose values above 1000 are patched.
    fn for_stack(len: usize, depth: usize, nullable: bool) -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = (0..len as u32).map(|i| {
            if i % 97 == 0 {
                5000 + i
            } else {
                (i * 7) % 1000
            }
        });
        let values = if nullable {
            PrimitiveArray::from_option_iter(
                values.enumerate().map(|(i, v)| (i % 13 != 0).then_some(v)),
            )
        } else {
            PrimitiveArray::from_iter(values)
        };
        let mut array = bitpack_encode(&values, 10, None, &mut ctx)?.into_array();
        for level in 0..depth {
            array = FoR::try_new(array, Scalar::from(1_000u32 + level as u32))?.into_array();
        }
        Ok(array)
    }

    /// FoR over BitPacked alone runs FoR's own fused kernel, so it fuses nothing. A slice of it
    /// does: the slice pushes into FoR, so FoR waits on a Slice child, which resolves into a
    /// BitPacked leaf part way into a chunk that FoR's op is fused over.
    #[rstest]
    #[case::fused_for_bp_alone(1, 0)]
    #[case::two_levels(2, 1)]
    #[case::three_levels(3, 1)]
    #[case::eight_levels(8, 1)]
    fn for_stacks(#[case] depth: usize, #[case] chains: usize) -> VortexResult<()> {
        for nullable in [false, true] {
            let array = for_stack(5000, depth, nullable)?;
            assert_fuses_like_execute(&array, chains)?;
            assert_fuses_like_execute(&array.slice(517..4013)?, 1)?;
        }
        Ok(())
    }

    #[test]
    fn signed_with_negative_references() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0..3000i64).map(|i| (i * 31) % 4000));
        let bp = bitpack_encode(&values, 12, None, &mut ctx)?.into_array();
        let inner = FoR::try_new(bp, Scalar::from(-1_000_000i64))?.into_array();
        let outer = FoR::try_new(inner, Scalar::from(-7i64))?.into_array();
        assert_fuses_like_execute(&outer, 1)?;
        assert_fuses_like_execute(&outer.slice(1..2999)?, 1)
    }

    #[test]
    fn blocked_references_over_misaligned_bitpacked() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0..4096u32).map(|i| i % 300));
        // Slicing the BitPacked leaf by itself leaves it at offset 100 while the FoR chunks start
        // at offset 0, so FoR cannot fuse the unpack: the leaf unpacks and FoR adds per block.
        let bp = bitpack_encode(&values, 9, None, &mut ctx)?
            .into_array()
            .slice(100..3172)?;
        let references = PrimitiveArray::new(buffer![10u32, 20, 30], Validity::NonNullable);
        let array = FoR::try_new_chunked(bp, references.into_array(), 0)?.into_array();
        assert_fuses_like_execute(&array, 1)?;
        let outer = FoR::try_new(array, Scalar::from(5u32))?.into_array();
        assert_fuses_like_execute(&outer, 1)
    }

    #[test]
    fn canonical_leaf_needs_two_parents() -> VortexResult<()> {
        let leaf = PrimitiveArray::from_iter(0..3000u32).into_array();
        let one = FoR::try_new(leaf, Scalar::from(1u32))?.into_array();
        assert_fuses_like_execute(&one, 0)?;
        let two = FoR::try_new(one, Scalar::from(2u32))?.into_array();
        assert_fuses_like_execute(&two, 1)?;
        let three = FoR::try_new(two, Scalar::from(3u32))?.into_array();
        assert_fuses_like_execute(&three, 1)
    }

    /// The fusion steps of executing `array`, checked against execution without fusion.
    fn fusion_trace(array: ArrayRef) -> VortexResult<String> {
        fusion::fusion_trace(array, &SESSION)
    }

    /// FoR over BitPacked alone runs FoR's own fused kernel. Above it, each FoR asks for its
    /// child as its spine, so the executor descends to FoR over BitPacked. On the way back up that
    /// leaf starts a block program, each FoR extends it with its op, and the root drives it once:
    /// FoR over BitPacked unpacks into scratch, ops that keep the width add their reference in
    /// place, and the last op writes the output.
    #[rstest]
    #[case::fused_for_bp_alone(1, "")]
    #[case::two_levels(
        2,
        "ExecuteSpine slot=0 parent=fastlanes.for.v2(u32, len=3000) child=fastlanes.for.v2(u32, len=3000)
fuse start array=fastlanes.for.v2(u32, len=3000)
fuse extend parent=fastlanes.for.v2(u32, len=3000) ops=1
drive ops=1 len=3000
block_op 0 copy u32->u32 scratch0 -> output"
    )]
    #[case::four_levels(
        4,
        "ExecuteSpine slot=0 parent=fastlanes.for.v2(u32, len=3000) child=fastlanes.for.v2(u32, len=3000)
ExecuteSpine slot=0 parent=fastlanes.for.v2(u32, len=3000) child=fastlanes.for.v2(u32, len=3000)
ExecuteSpine slot=0 parent=fastlanes.for.v2(u32, len=3000) child=fastlanes.for.v2(u32, len=3000)
fuse start array=fastlanes.for.v2(u32, len=3000)
fuse extend parent=fastlanes.for.v2(u32, len=3000) ops=1
fuse extend parent=fastlanes.for.v2(u32, len=3000) ops=2
fuse extend parent=fastlanes.for.v2(u32, len=3000) ops=3
drive ops=3 len=3000
block_op 0 in_place u32->u32 scratch0 -> scratch0
block_op 1 in_place u32->u32 scratch0 -> scratch0
block_op 2 copy u32->u32 scratch0 -> output"
    )]
    fn for_stack_schedule(#[case] depth: usize, #[case] expected: &str) -> VortexResult<()> {
        assert_eq!(fusion_trace(for_stack(3000, depth, false)?)?, expected);
        Ok(())
    }

    /// Three FoRs over a primitive array: the innermost FoR runs its own kernel into a buffer the
    /// executor owns, which is the leaf of the two FoRs above, decoded block by block in place.
    /// A shared input leaf is only read, so it keeps its values.
    #[test]
    fn canonical_leaf_schedule() -> VortexResult<()> {
        let build = || -> VortexResult<ArrayRef> {
            let mut array = PrimitiveArray::from_iter(0..3000u32).into_array();
            for reference in 1..=3u32 {
                array = FoR::try_new(array, Scalar::from(reference))?.into_array();
            }
            Ok(array)
        };
        let expected = "ExecuteSpine slot=0 parent=fastlanes.for.v2(u32, len=3000) child=fastlanes.for.v2(u32, len=3000)
ExecuteSpine slot=0 parent=fastlanes.for.v2(u32, len=3000) child=fastlanes.for.v2(u32, len=3000)
fuse start array=vortex.primitive(u32, len=3000)
fuse extend parent=fastlanes.for.v2(u32, len=3000) ops=1
fuse extend parent=fastlanes.for.v2(u32, len=3000) ops=2
drive ops=2 len=3000
block_op 0 in_place u32->u32 leaf -> leaf
block_op 1 in_place u32->u32 leaf -> leaf";

        assert_eq!(fusion_trace(build()?)?, expected);
        let shared = build()?;
        assert_eq!(fusion_trace(shared.clone())?, expected);
        assert_arrays_eq!(shared, build()?, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    /// FoR with per-chunk references that do not line up with its BitPacked child cannot run the
    /// fused unpack, so the child is the source and FoR's op adds each chunk's reference.
    #[test]
    fn misaligned_blocked_references_schedule() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0..4096u32).map(|i| i % 300));
        let bp = bitpack_encode(&values, 9, None, &mut ctx)?
            .into_array()
            .slice(100..3172)?;
        let references = PrimitiveArray::new(buffer![10u32, 20, 30], Validity::NonNullable);
        let array = FoR::try_new_chunked(bp, references.into_array(), 0)?.into_array();
        assert_eq!(
            fusion_trace(array)?,
            "ExecuteSpine slot=0 parent=fastlanes.for.v2(u32, len=3072) child=fastlanes.bitpacked.v2(u32, len=3072)
fuse start array=fastlanes.bitpacked.v2(u32, len=3072)
fuse extend parent=fastlanes.for.v2(u32, len=3072) ops=1
drive ops=1 len=3072
block_op 0 copy u32->u32 scratch0 -> output"
        );
        Ok(())
    }

    /// A spine child's validity can be lazy over its own children: Dict's is a dict of its codes
    /// over the values' validity. The chain takes the validity of the executed leaf, so fusing
    /// decodes the codes no more often than level-wise execution does.
    #[test]
    fn lazy_child_validity_executes_once() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let codes = PrimitiveArray::from_iter((0..3000u32).map(|i| i % 7));
        let codes = bitpack_encode(&codes, 3, None, &mut ctx)?.into_array();
        let values =
            PrimitiveArray::from_option_iter((0..7u32).map(|i| (i != 3).then_some(i * 100)));
        let dict = DictArray::try_new(codes, values.into_array())?.into_array();
        let array = FoR::try_new(dict, Scalar::from(5u32))?.into_array();
        let packed_executions = |fusion: bool| -> VortexResult<usize> {
            let mut ctx = SESSION.create_execution_ctx().with_block_fusion(fusion);
            let traced = trace_op(|| array.clone().execute::<PrimitiveArray>(&mut ctx))?;
            Ok(traced
                .trace
                .to_string()
                .lines()
                .filter(|line| {
                    line.trim().starts_with("execute_until ")
                        && line.contains("root=fastlanes.bitpacked")
                })
                .count())
        };
        assert_eq!(packed_executions(false)?, 1);
        assert_eq!(packed_executions(true)?, 1);
        assert_fuses_like_execute(&array, 1)
    }

    /// FoR with a zero reference has no op, since level-wise execution returns its child as is.
    /// The executor asks for the op before the leaf's source, so the patched FoR over BitPacked
    /// below builds no source, and the frame is off for the rest of the execution.
    #[test]
    fn parent_without_op_builds_no_source() -> VortexResult<()> {
        let array = FoR::try_new(for_stack(3000, 1, true)?, Scalar::from(0u32))?.into_array();
        assert_eq!(
            fusion_trace(array.clone())?,
            "ExecuteSpine slot=0 parent=fastlanes.for.v2(u32?, len=3000) child=fastlanes.for.v2(u32?, len=3000)"
        );
        assert_fuses_like_execute(&array, 0)
    }

    /// A packed source fills any in-order ranges, including ones that straddle FastLanes chunks,
    /// which the executor only asks for when a grid other than the source's own is in use.
    #[rstest]
    #[case::plain(false)]
    #[case::fused_reference(true)]
    fn packed_source_fills_any_ranges(#[case] with_reference: bool) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // A sliced BitPacked array, starting part way into its first chunk, with patches.
        let bp = for_stack(5000, 0, false)?
            .slice(517..4800)?
            .execute::<ArrayRef>(&mut ctx)?;
        assert!(bp.is::<BitPacked>());
        let array = if with_reference {
            FoR::try_new(bp, Scalar::from(1000u32))?.into_array()
        } else {
            bp
        };
        let expected = array.clone().execute::<PrimitiveArray>(&mut ctx)?;
        let mut source = if with_reference {
            for_block_source(array.as_::<FoR>(), &mut ctx)?
        } else {
            packed_source(array.as_::<BitPacked>(), &mut ctx)?
        }
        .vortex_expect("a source");
        let mut scratch = BlockBuf::new_boxed();
        let mut values = Vec::new();
        for rows in [0..300, 300..1324, 1324..1500, 1500..2900, 2900..array.len()]
            .into_iter()
            .flat_map(|rows| {
                // At most one block per call.
                (rows.start..rows.end)
                    .step_by(BLOCK_LEN)
                    .map(move |start| start..(start + BLOCK_LEN).min(rows.end))
            })
        {
            match source.fill(rows.clone(), &mut scratch)? {
                SourceBlock::Scratch { offset, len, .. } => {
                    assert_eq!(len, rows.len());
                    let filled = scratch.out::<u32>(offset + len);
                    // SAFETY: the source wrote these values.
                    values.extend(filled[offset..].iter().map(|v| unsafe { v.assume_init() }));
                }
                SourceBlock::Lent(block) => values.extend_from_slice(block.as_slice::<u32>()),
                _ => unreachable!(),
            }
        }
        assert_eq!(values, expected.as_slice::<u32>());
        Ok(())
    }
}
