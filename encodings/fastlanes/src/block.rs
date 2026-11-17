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
use vortex_array::block::BlockIn;
use vortex_array::block::BlockOp;
use vortex_array::block::BlockPatches;
use vortex_array::block::BlockSource;
use vortex_array::block::Written;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_compute::lane_kernels::IndexedSinkExt;
use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

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
/// unpack adds the reference to each chunk while it unpacks it, as FoR's own execute does.
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
    let elems_per_chunk = 128 * bit_width / size_of::<T>();
    let packed = bp.packed().as_host().clone();
    vortex_ensure_eq!(
        packed.len(),
        (offset + bp.len()).div_ceil(FL_CHUNK_SIZE) * elems_per_chunk * size_of::<T>(),
        "packed buffer does not hold every chunk"
    );
    let patches = bp
        .patches()
        .map(|patches| BlockPatches::try_new(&patches, ctx, |_, value: T| patch(value)))
        .transpose()?;
    Ok(Some(Box::new(PackedSource::<T, S> {
        strategy,
        packed,
        bit_width,
        elems_per_chunk,
        offset,
        len: bp.len(),
        chunk: 0,
        patches,
    })))
}

/// Unpacks one FastLanes chunk per block with its strategy.
struct PackedSource<T, S> {
    strategy: S,
    packed: ByteBuffer,
    bit_width: usize,
    elems_per_chunk: usize,
    /// The position of the first row within the first chunk.
    offset: usize,
    len: usize,
    /// The next chunk to unpack.
    chunk: usize,
    patches: Option<BlockPatches<T>>,
}

impl<T: PhysicalPType, S: UnpackStrategy<T> + Send> BlockSource for PackedSource<T, S> {
    fn next_block<'s>(&'s mut self, scratch: &'s mut BlockBuf) -> VortexResult<BlockIn<'s>> {
        let chunk_start = self.chunk * FL_CHUNK_SIZE;
        let first = self.offset.max(chunk_start);
        let end = (chunk_start + FL_CHUNK_SIZE).min(self.offset + self.len);
        // SAFETY: `ByteBuffer` is aligned for the packed integers it was built from, and the
        // constructor checked it holds `elems_per_chunk` integers for every chunk.
        let packed = unsafe {
            std::slice::from_raw_parts(
                self.packed
                    .as_ptr()
                    .cast::<T::Physical>()
                    .add(self.chunk * self.elems_per_chunk),
                self.elems_per_chunk,
            )
        };
        self.chunk += 1;
        let out = scratch.out::<T::Physical>(FL_CHUNK_SIZE);
        // SAFETY: the unpack writes all `FL_CHUNK_SIZE` values of `out`, which holds one chunk,
        // from `packed`, which holds one chunk at `bit_width`.
        let values = unsafe {
            let out =
                std::mem::transmute::<&mut [MaybeUninit<T::Physical>], &mut [T::Physical]>(out);
            self.strategy.unpack_chunk(self.bit_width, packed, out);
            // `T` and `T::Physical` have the same size and alignment.
            std::mem::transmute::<&mut [T::Physical], &mut [T]>(out)
        };
        let block = &mut values[first - chunk_start..end - chunk_start];
        if let Some(patches) = &mut self.patches {
            patches.apply(first - self.offset, block);
        }
        Ok(BlockIn::from_slice(block))
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
                Some(Box::new(AddReference { reference }) as Box<dyn BlockOp>)
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
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::block::fused_chain_count;
    use vortex_array::block::in_place_chain_count;
    use vortex_array::scalar::Scalar;
    use vortex_array::test_harness::trace::trace_op;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::FoR;
    use crate::bitpack_compress::bitpack_encode;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    /// Execute `array` with and without fusion and check the results agree, and how many chains
    /// fused.
    fn assert_fuses_like_execute(array: &ArrayRef, chains: usize) -> VortexResult<()> {
        let expected = array.clone().execute::<PrimitiveArray>(
            &mut SESSION.create_execution_ctx().with_block_fusion(false),
        )?;
        let before = fused_chain_count();
        let fused = array.clone().execute::<PrimitiveArray>(
            &mut SESSION.create_execution_ctx().with_block_fusion(true),
        )?;
        assert_eq!(
            fused_chain_count() - before,
            chains,
            "fused chains for\n{}",
            array.display_tree()
        );
        assert_arrays_eq!(fused, expected, &mut SESSION.create_execution_ctx());
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

    /// A canonical leaf the executor holds the only reference to is decoded in its own buffer.
    /// One that is shared is read and decoded into a new buffer.
    #[test]
    fn canonical_leaf_decodes_in_place_when_owned() -> VortexResult<()> {
        let build = || -> VortexResult<ArrayRef> {
            let leaf = PrimitiveArray::from_iter((0..3000u32).map(|i| i * 3)).into_array();
            let one = FoR::try_new(leaf, Scalar::from(1u32))?.into_array();
            Ok(FoR::try_new(one, Scalar::from(2u32))?.into_array())
        };
        let mut ctx = SESSION.create_execution_ctx();
        let expected = build()?.execute::<PrimitiveArray>(
            &mut SESSION.create_execution_ctx().with_block_fusion(false),
        )?;

        let before = in_place_chain_count();
        let owned = build()?.execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(in_place_chain_count() - before, 1);
        assert_arrays_eq!(owned, expected, &mut ctx);

        let shared = build()?;
        let before = in_place_chain_count();
        let copied = shared.clone().execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(in_place_chain_count() - before, 0);
        // The input still holds its own values.
        assert_eq!(shared.len(), copied.len());
        assert_arrays_eq!(copied, expected, &mut ctx);
        Ok(())
    }

    /// The fusion steps in the trace of executing `array`: the spine requests, each fused chain,
    /// and how each of its ops ran over the first block.
    fn fusion_trace(array: ArrayRef) -> VortexResult<String> {
        let mut ctx = SESSION.create_execution_ctx();
        let traced = trace_op(|| array.execute::<PrimitiveArray>(&mut ctx))?;
        Ok(traced
            .trace
            .to_string()
            .lines()
            .map(str::trim)
            .filter(|line| {
                line.starts_with("ExecuteSpine")
                    || line.starts_with("fuse ")
                    || line.starts_with("block_op")
            })
            .collect::<Vec<_>>()
            .join("\n"))
    }

    /// FoR over BitPacked alone runs FoR's own fused kernel. Above it, each FoR asks for its
    /// child as its spine, and the chain fuses as soon as the leaf can emit blocks: FoR over
    /// BitPacked unpacks into scratch, ops that keep the width add their reference in place, and
    /// the last op writes the output. With two FoRs waiting, the leaf is FoR(FoR(BitPacked)),
    /// whose own op runs over its child's source.
    #[rstest]
    #[case::fused_for_bp_alone(1, "")]
    #[case::two_levels(
        2,
        "ExecuteSpine slot=0 parent=fastlanes.for.v2(u32, len=3000) child=fastlanes.for.v2(u32, len=3000)
fuse top=fastlanes.for.v2(u32, len=3000) input=source frames=1 ops=1
block_op 0 copy u32->u32 scratch0 -> output"
    )]
    #[case::four_levels(
        4,
        "ExecuteSpine slot=0 parent=fastlanes.for.v2(u32, len=3000) child=fastlanes.for.v2(u32, len=3000)
ExecuteSpine slot=0 parent=fastlanes.for.v2(u32, len=3000) child=fastlanes.for.v2(u32, len=3000)
fuse top=fastlanes.for.v2(u32, len=3000) input=source frames=2 ops=3
block_op 0 in_place u32->u32 scratch0 -> scratch0
block_op 1 in_place u32->u32 scratch0 -> scratch0
block_op 2 copy u32->u32 scratch0 -> output"
    )]
    fn for_stack_schedule(#[case] depth: usize, #[case] expected: &str) -> VortexResult<()> {
        assert_eq!(fusion_trace(for_stack(3000, depth, false)?)?, expected);
        Ok(())
    }

    /// Three FoRs over a canonical leaf. A leaf another owner shares is copied into scratch by
    /// the first op; one the executor owns is decoded block by block in its own buffer.
    #[test]
    fn canonical_leaf_schedule() -> VortexResult<()> {
        let build = || -> VortexResult<ArrayRef> {
            let mut array = PrimitiveArray::from_iter(0..3000u32).into_array();
            for reference in 1..=3u32 {
                array = FoR::try_new(array, Scalar::from(reference))?.into_array();
            }
            Ok(array)
        };
        let frames = "ExecuteSpine slot=0 parent=fastlanes.for.v2(u32, len=3000) child=fastlanes.for.v2(u32, len=3000)
ExecuteSpine slot=0 parent=fastlanes.for.v2(u32, len=3000) child=fastlanes.for.v2(u32, len=3000)
fuse top=fastlanes.for.v2(u32, len=3000) input=canonical frames=2 ops=3";

        let shared = build()?;
        assert_eq!(
            fusion_trace(shared.clone())?,
            format!(
                "{frames}
block_op 0 copy u32->u32 source -> scratch0
block_op 1 in_place u32->u32 scratch0 -> scratch0
block_op 2 copy u32->u32 scratch0 -> output"
            )
        );
        assert_eq!(shared.len(), 3000);

        assert_eq!(
            fusion_trace(build()?)?,
            format!(
                "{frames}
block_op 0 in_place u32->u32 leaf -> leaf
block_op 1 in_place u32->u32 leaf -> leaf
block_op 2 in_place u32->u32 leaf -> leaf"
            )
        );
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
fuse top=fastlanes.for.v2(u32, len=3072) input=source frames=1 ops=1
block_op 0 copy u32->u32 scratch0 -> output"
        );
        Ok(())
    }
}
