// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streaming chunked decompression for bit-packed arrays.
//!
//! This backs [`VTable::decompress_chunks`](vortex_array::vtable::VTable::decompress_chunks) for
//! [`BitPacked`]: each FastLanes block is unpacked straight into the sink's destination when it
//! offers one, else into a cache-resident scratch buffer, patches falling in that block are
//! applied in place, and the block is handed to the sink, so the array is never materialized in
//! full.

use fastlanes::BitPacking;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::chunk_iter::ChunkMut;
use vortex_array::chunk_iter::ChunkPatches;
use vortex_array::chunk_iter::ChunkSink;
use vortex_array::chunk_iter::ScratchChunk;
use vortex_array::chunk_iter::emit_with;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_integer_ptype;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::FL_CHUNK_SIZE;
use crate::unpack_iter::for_each_packed_chunk;

pub(crate) fn decompress_chunks(
    array: ArrayView<'_, BitPacked>,
    ctx: &mut ExecutionCtx,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()> {
    match_each_integer_ptype!(array.dtype().as_ptype(), |T| {
        let patches = match array.patches() {
            None => ChunkPatches::new(Vec::new()),
            Some(patches) => ChunkPatches::try_from_patches(&patches, ctx, |_, value: T| value)?,
        };
        let bit_width = array.bit_width() as usize;
        stream_packed_chunks::<T>(array, patches, sink, |_, packed, out| {
            // SAFETY: `packed` holds one chunk at `bit_width`, and `out` holds a full chunk.
            unsafe { BitPacking::unchecked_unpack(bit_width, packed, out) }
        })
    })
}

/// Stream every chunk of `bp`, unpacked by `unpack(chunk, packed, out)` and then patched.
///
/// `chunk` counts from the first chunk of `bp`, and `out` holds a whole chunk. Whole chunks unpack
/// straight into the sink's destination when it offers one; a partial first or last chunk unpacks
/// into scratch and only its rows are forwarded.
pub(crate) fn stream_packed_chunks<T>(
    bp: ArrayView<'_, BitPacked>,
    mut patches: ChunkPatches<T>,
    sink: &mut dyn ChunkSink,
    unpack: impl Fn(usize, &[T::Physical], &mut [T::Physical]),
) -> VortexResult<()>
where
    T: PhysicalPType<Physical: BitPacking>,
{
    let len = bp.len();
    if len == 0 {
        return Ok(());
    }
    let offset = usize::from(bp.offset());
    let mut scratch = ScratchChunk::<T>::new();
    let mut result = Ok(());
    for_each_packed_chunk::<T, _>(
        bp.packed_slice::<T::Physical>(),
        bp.bit_width() as usize,
        offset,
        len,
        |packed, range| {
            if result.is_err() {
                return;
            }
            let chunk = range.start / FL_CHUNK_SIZE;
            // `range` counts from the start of the first chunk, and the rows start at `offset`.
            let skip = offset.saturating_sub(range.start);
            let rows = range.start + skip - offset..range.end - offset;
            let start = rows.start;
            result = if rows.len() == FL_CHUNK_SIZE {
                emit_with(sink, T::PTYPE, rows, &mut scratch, |out| {
                    unpack(chunk, packed, physical_mut(out));
                    patches.apply(out, start);
                    Ok(())
                })
            } else {
                let scratch = scratch.values();
                unpack(chunk, packed, physical_mut(scratch));
                let values = &mut scratch[skip..skip + rows.len()];
                patches.apply(values, start);
                sink.accept(ChunkMut::new(values), rows)
            };
        },
    )?;
    result
}

/// View values as their physical type, which the FastLanes kernels write.
fn physical_mut<T: PhysicalPType>(values: &mut [T]) -> &mut [T::Physical] {
    // SAFETY: `T::Physical` is `T` with the same size and alignment, and every bit pattern is
    // valid for both.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), values.len()) }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::chunk_iter::ChunkMut;
    use vortex_array::dtype::NativePType;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::BitPackedArrayExt;
    use crate::FoRArraySlotsExt;
    use crate::FoRData;
    use crate::bitpack_compress::bitpack_encode;

    pub(super) static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    fn collect_chunks<T: NativePType>(array: &vortex_array::ArrayRef) -> VortexResult<Vec<T>> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut out = Vec::with_capacity(array.len());
        array.decompress_chunks_or_materialize(&mut ctx, &mut |chunk: ChunkMut<'_>,
                                                                _range: std::ops::Range<
            usize,
        >|
         -> VortexResult<()> {
            out.extend_from_slice(chunk.as_slice::<T>());
            Ok(())
        })?;
        Ok(out)
    }

    fn assert_chunks_match_execute<T: NativePType>(
        array: vortex_array::ArrayRef,
    ) -> VortexResult<()> {
        let chunked = collect_chunks::<T>(&array)?;
        let mut ctx = SESSION.create_execution_ctx();
        let expected = array.execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(chunked.as_slice(), expected.as_slice::<T>());
        Ok(())
    }

    #[test]
    fn bitpacked_chunks_match_execute() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0..5000u32).map(|i| i % 900));
        let bp = bitpack_encode(&values, 10, None, &mut ctx)?;
        let array = bp.into_array();
        assert!(array.supports_decompress_chunks());
        assert_chunks_match_execute::<u32>(array)
    }

    #[test]
    fn bitpacked_chunks_with_patches() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter(
            (0..5000u32).map(|i| if i % 700 == 0 { 100_000 + i } else { i % 900 }),
        );
        let bp = bitpack_encode(&values, 10, None, &mut ctx)?;
        assert!(bp.patches().is_some());
        assert_chunks_match_execute::<u32>(bp.into_array())
    }

    #[test]
    fn bitpacked_chunks_sliced() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter(
            (0..5000u32).map(|i| if i % 700 == 0 { 100_000 + i } else { i % 900 }),
        );
        let bp = bitpack_encode(&values, 10, None, &mut ctx)?.into_array();
        // Slice crossing chunk boundaries with a non-zero offset.
        let sliced = bp.slice(517..4013)?;
        // The slice may no longer be a BitPacked array head, but chunked iteration must still
        // stream correct values via whatever encodings the slice resolves to.
        assert_chunks_match_execute::<u32>(sliced)
    }

    #[test]
    fn for_over_bitpacked_chunks() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0..5000i64).map(|i| 1_000_000 + (i * 7) % 800));
        let for_array = FoRData::encode(values, &mut ctx)?;
        assert!(
            for_array.encoded().as_opt::<crate::BitPacked>().is_some()
                || for_array
                    .encoded()
                    .as_opt::<vortex_array::arrays::Primitive>()
                    .is_some()
        );
        assert_chunks_match_execute::<i64>(for_array.into_array())
    }

    /// FoR over a BitPacked child streams through the fused unpack kernel that adds the
    /// reference, including patch handling.
    #[test]
    fn for_over_bitpacked_fused_chunks_with_patches() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let deltas = PrimitiveArray::from_iter(
            (0..5000u32).map(|i| if i % 700 == 0 { 100_000 + i } else { i % 900 }),
        );
        let bp = bitpack_encode(&deltas, 10, None, &mut ctx)?;
        assert!(bp.patches().is_some());
        let for_array = crate::FoR::try_new(
            bp.into_array(),
            vortex_array::scalar::Scalar::from(1_000_000u32),
        )?;
        assert_chunks_match_execute::<u32>(for_array.into_array())
    }

    #[test]
    fn fallback_primitive_chunks() -> VortexResult<()> {
        let values = PrimitiveArray::from_iter(0..3000i32).into_array();
        assert_chunks_match_execute::<i32>(values)
    }

    #[test]
    fn empty_array_chunks() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter(Vec::<u32>::new());
        let bp = bitpack_encode(&values, 0, None, &mut ctx)?;
        let chunked = collect_chunks::<u32>(&bp.into_array())?;
        assert!(chunked.is_empty());
        Ok(())
    }
}

#[cfg(test)]
mod executor_tests {
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::chunk_iter::execute_via_chunks;
    use vortex_array::chunk_iter::without_chunked_execute;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

    use super::tests::SESSION;
    use crate::FoR;
    use crate::bitpack_compress::bitpack_encode;

    /// `depth` nested FoR levels over a nullable BitPacked leaf.
    fn for_stack(depth: usize, ctx: &mut vortex_array::ExecutionCtx) -> VortexResult<ArrayRef> {
        let values = Buffer::from_iter((0..5000i32).map(|i| i % 900));
        let validity = Validity::from_iter((0..5000).map(|i| i % 7 != 0));
        let mut array =
            bitpack_encode(&PrimitiveArray::new(values, validity), 10, None, ctx)?.into_array();
        for _ in 0..depth {
            // The innermost level fuses with the BitPacked unpack; every level above it adds its
            // reference to the chunks streamed up from below.
            array = FoR::try_new(array, Scalar::from(-1_000i32))?.into_array();
        }
        Ok(array)
    }

    /// Streaming a tree to canonical, directly or through the executor's shortcut, must match
    /// level-wise execution exactly, including validity, at every stack depth.
    #[rstest::rstest]
    #[case::single(1)]
    #[case::pair(2)]
    #[case::deep(5)]
    fn execute_via_chunks_matches_levelwise(#[case] depth: usize) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = for_stack(depth, &mut ctx)?;
        assert!(array.supports_decompress_chunks());

        let streaming = execute_via_chunks(&array, &mut ctx)?;
        let executed = array.clone().execute::<PrimitiveArray>(&mut ctx)?;
        let levelwise = without_chunked_execute(|| array.execute::<PrimitiveArray>(&mut ctx))?;

        assert_arrays_eq!(streaming, levelwise, &mut ctx);
        assert_arrays_eq!(executed, levelwise, &mut ctx);
        Ok(())
    }

    /// The executor only takes the streaming shortcut once a tree is deep enough for it to pay
    /// off, and never for cardinality-changing roots.
    #[test]
    fn executor_depth_rule() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        assert!(!for_stack(0, &mut ctx)?.should_execute_via_chunks());
        assert!(!for_stack(1, &mut ctx)?.should_execute_via_chunks());
        assert!(for_stack(2, &mut ctx)?.should_execute_via_chunks());
        assert!(for_stack(8, &mut ctx)?.should_execute_via_chunks());

        // A Filter root streams, but never through the executor: its level-wise kernel already
        // decodes into the output and compacts in place, so there is no intermediate to remove.
        let filtered = for_stack(0, &mut ctx)?
            .filter(vortex_mask::Mask::from_iter((0..5000).map(|i| i % 3 == 0)))?;
        assert!(filtered.supports_decompress_chunks());
        assert!(!filtered.should_execute_via_chunks());

        // Filter push-down can sink a filter beneath a deep stack; the levels above it still
        // stream, because each of them would otherwise materialize a full intermediate.
        let deep_filtered = for_stack(8, &mut ctx)?
            .filter(vortex_mask::Mask::from_iter((0..5000).map(|i| i % 3 == 0)))?;
        assert!(deep_filtered.should_execute_via_chunks());
        Ok(())
    }
}

#[cfg(test)]
mod filter_tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::chunk_iter::execute_via_chunks;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use super::tests::SESSION;
    use crate::bitpack_compress::bitpack_encode;

    /// Filter over BitPacked must stream to the same canonical result as level-wise execution,
    /// across selectivities and mask shapes that straddle the 1024-element block boundary.
    #[rstest]
    #[case::sparse(97)]
    #[case::medium(7)]
    #[case::dense(2)]
    #[case::every_row(1)]
    fn filter_over_bitpacked_streams_like_execute(#[case] keep_every: usize) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let len = 5000;
        let values = PrimitiveArray::from_iter((0..len as u32).map(|i| i % 900));
        let bp = bitpack_encode(&values, 10, None, &mut ctx)?.into_array();

        // A run-structured mask: keeps a 3-wide run every `keep_every` rows, so runs cross
        // block boundaries at 1024/2048/... for several of these cases.
        let mask = Mask::from_iter((0..len).map(|i| (i % keep_every) < 3));
        let filtered = bp.filter(mask)?;
        assert!(filtered.supports_decompress_chunks());

        let streaming = execute_via_chunks(&filtered, &mut ctx)?;
        let levelwise = filtered.execute::<PrimitiveArray>(&mut ctx)?;

        assert_eq!(streaming.len(), levelwise.len());
        assert_arrays_eq!(streaming, levelwise, &mut ctx);
        Ok(())
    }

    #[test]
    fn filter_all_false_and_all_true_over_bitpacked() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0..3000u32).map(|i| i % 900));
        let bp = bitpack_encode(&values, 10, None, &mut ctx)?.into_array();

        let none = bp.filter(Mask::new_false(3000))?;
        assert_eq!(none.execute::<PrimitiveArray>(&mut ctx)?.len(), 0);

        let all = bp.filter(Mask::new_true(3000))?;
        let all = all.execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(all, values, &mut ctx);
        Ok(())
    }
}
