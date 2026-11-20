// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decompression of a [`Patched`] array over a [`BitPacked`] array, with each chunk's patches
//! applied as soon as the chunk is unpacked.

use std::mem::MaybeUninit;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Patched;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::patched::PATCH_CHUNK_SIZE;
use vortex_array::arrays::patched::PatchLayout;
use vortex_array::arrays::patched::PatchedArrayExt;
use vortex_array::arrays::patched::PatchedArraySlotsExt;
use vortex_array::arrays::patched::PatchedSlots;
use vortex_array::arrays::patched::patch_validity;
use vortex_array::arrays::patched::scatter;
use vortex_array::builders::PrimitiveBuilder;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::match_each_integer_ptype;
use vortex_array::matcher::Matcher;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::BitWidthsView;
use crate::FL_CHUNK_SIZE;
use crate::bitpacking::array::bitpack_decompress::unpack_into_primitive_builder;
use crate::unpack_iter::BitPacked as BitPackedUnpack;

/// Executes a [`Patched`] parent of a [`BitPacked`] array in one pass.
///
/// Unfused, the bit-packed array is first unpacked in full and the patches are then scattered
/// over the result, reading the output a second time. This kernel instead applies the patches of
/// each 1024-element chunk immediately after unpacking it, while the chunk is still in cache.
#[derive(Debug)]
pub(crate) struct BitPackedPatchedKernel;

impl ExecuteParentKernel<BitPacked> for BitPackedPatchedKernel {
    type Parent = Patched;

    fn execute_parent(
        &self,
        array: ArrayView<'_, BitPacked>,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != PatchedSlots::INNER {
            return Ok(None);
        }
        // Patches inside the bit-packed array would have to be applied first.
        if array.patches().is_some() {
            return Ok(None);
        }
        if !matches!(array.bit_widths(), BitWidthsView::Global(_)) {
            return Ok(None);
        }
        // The chunks of the bit-packed array line up with the patch chunks only when both are
        // offset by the same amount into their first chunk.
        if usize::from(array.offset()) != parent.offset() {
            return Ok(None);
        }

        let values = parent
            .patch_values()
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        let layout = parent.execute_layout(ctx)?;
        let layout = layout.layout()?;

        let len = array.len();

        // Sparse patches are in position order with a single lane. When they cannot change the
        // validity, decode exactly as the bit-packed array does and then overwrite the patched
        // positions: the decoder runs uninterrupted, and the few patches cost almost nothing.
        if layout.n_lanes() == 1
            && array.validity()?.definitely_no_nulls()
            && values.as_ref().validity()?.definitely_no_nulls()
        {
            let result = match_each_integer_ptype!(array.dtype().as_ptype(), |T| {
                let mut builder = PrimitiveBuilder::<T>::with_capacity_in(
                    array.dtype().nullability(),
                    len,
                    ctx.allocator(),
                );
                unpack_into_primitive_builder::<T>(array, &mut builder, ctx)?;
                scatter(builder.values_mut(), values.as_slice::<T>(), &layout);
                builder.finish_into_primitive()
            });
            return Ok(Some(result.into_array()));
        }

        let validity = patch_validity(
            array.validity()?,
            values.as_ref().validity()?,
            values.len(),
            &layout,
            ctx,
        )?;
        let validity = validity.execute_mask(len, ctx)?;

        let result = match_each_integer_ptype!(array.dtype().as_ptype(), |T| {
            let mut builder = PrimitiveBuilder::<T>::with_capacity_in(
                array.dtype().nullability(),
                len,
                ctx.allocator(),
            );
            if len > 0 {
                let mut uninit_range = builder.uninit_range(len);
                // SAFETY: every value of the range is initialized by the decode below.
                unsafe { uninit_range.append_mask(&validity) };
                // SAFETY: the decode writes a value to every slot of this range.
                let output = unsafe { uninit_range.slice_uninit_mut(0, len) };
                decode_patched::<T>(array, values.as_slice::<T>(), &layout, output)?;
                // SAFETY: the values and validity of all `len` positions have been set.
                unsafe { uninit_range.finish() };
            }
            builder.finish_into_primitive()
        });

        Ok(Some(result.into_array()))
    }
}

/// The number of chunks unpacked before their patches are applied: 8 chunks of 64-bit values
/// take 64KiB, which stays in cache.
const PATCH_BATCH_CHUNKS: usize = 8;

/// Unpack `array` into `output`, writing the patch values over each chunk once it is unpacked.
fn decode_patched<T: BitPackedUnpack>(
    array: ArrayView<'_, BitPacked>,
    values: &[T],
    layout: &PatchLayout<'_>,
    output: &mut [MaybeUninit<T>],
) -> VortexResult<()> {
    assert_eq!(FL_CHUNK_SIZE, PATCH_CHUNK_SIZE);
    assert!(layout.n_patches() <= values.len());

    let offset = layout.offset();
    let end = offset + layout.len();
    let indices = layout.indices();

    // Patches are applied a batch of chunks at a time: the decoder unpacks each batch in a tight
    // loop and stops after the last chunk of every batch that has patches, while the batch is
    // still in cache. Stopping after every chunk costs more than it saves when most chunks have
    // a patch.
    let n_chunks = layout.n_chunks();
    let batch_ends: Vec<usize> = (0..n_chunks.div_ceil(PATCH_BATCH_CHUNKS))
        .filter(|&batch| {
            let start = batch * PATCH_BATCH_CHUNKS;
            let end = (start + PATCH_BATCH_CHUNKS).min(n_chunks);
            layout.chunk_patches(start).start != layout.chunk_patches(end - 1).end
        })
        .map(|batch| ((batch + 1) * PATCH_BATCH_CHUNKS).min(n_chunks) - 1)
        .collect();

    let mut scratch = [const { MaybeUninit::<T>::uninit() }; FL_CHUNK_SIZE];
    let mut chunks = array.unpacked_chunks::<T>(&mut scratch)?;
    chunks.decode_into_with(output, &batch_ends, |last_chunk, output| {
        let first_chunk = last_chunk - last_chunk % PATCH_BATCH_CHUNKS;
        for chunk in first_chunk..=last_chunk {
            let patches = layout.chunk_patches(chunk);
            let chunk_start = chunk * PATCH_CHUNK_SIZE;
            if chunk_start >= offset && chunk_start + PATCH_CHUNK_SIZE <= end {
                let output = &mut output[chunk_start - offset..][..PATCH_CHUNK_SIZE];
                for patch in patches {
                    output[usize::from(indices[patch]) & (PATCH_CHUNK_SIZE - 1)]
                        .write(values[patch]);
                }
            } else {
                for patch in patches {
                    let position =
                        chunk_start + (usize::from(indices[patch]) & (PATCH_CHUNK_SIZE - 1));
                    if position >= offset && position < end {
                        output[position - offset].write(values[patch]);
                    }
                }
            }
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Patched;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::patched::PatchedArraySlotsExt;
    use vortex_array::arrays::patched::PatchedSlots;
    use vortex_array::assert_arrays_eq;
    use vortex_array::kernel::ExecuteParentKernel;
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::BitPackedPatchedKernel;
    use crate::BitPacked;
    use crate::bitpack_compress::bitpack_encode;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    /// Values that fit 6 bits, with every 97th value too wide and nulls every 13th value.
    fn data(len: u32, nullable: bool) -> PrimitiveArray {
        let value = |i: u32| if i % 97 == 5 { 1_000_000 + i } else { i % 64 };
        if nullable {
            PrimitiveArray::from_option_iter((0..len).map(|i| (i % 13 != 0).then(|| value(i))))
        } else {
            PrimitiveArray::from_iter((0..len).map(value))
        }
    }

    fn patched(array: &PrimitiveArray) -> VortexResult<vortex_array::arrays::PatchedArray> {
        let mut ctx = SESSION.create_execution_ctx();
        let packed = bitpack_encode(array, 6, None, &mut ctx)?;
        let mut parts = BitPacked::into_parts(packed);
        let patches = parts
            .patches
            .take()
            .expect("values wider than 6 bits are patches");
        let crate::BitWidths::Global(bit_width) = parts.bit_widths else {
            unreachable!("encoded with a global width")
        };
        let inner = BitPacked::try_new(
            parts.packed,
            array.ptype(),
            parts.validity,
            None,
            bit_width,
            parts.len,
            parts.offset,
        )?;
        Patched::from_array_and_patches(inner.into_array(), &patches, &mut ctx)
    }

    #[rstest]
    #[case(5000, false, 0..5000)]
    #[case(5000, true, 0..5000)]
    #[case(5000, true, 3..4999)]
    #[case(5000, false, 1030..1100)]
    #[case(100, true, 0..100)]
    #[case(2048, false, 1024..2048)]
    fn fused_decode_matches_reference(
        #[case] len: u32,
        #[case] nullable: bool,
        #[case] range: std::ops::Range<usize>,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = data(len, nullable);
        let expected = array.clone().into_array().slice(range.clone())?;
        let patched = patched(&array)?
            .into_array()
            .slice(range)?
            .optimize_ctx(&SESSION)?;
        let patched = patched.as_::<Patched>();

        let inner = patched.inner().as_::<BitPacked>();
        let fused = BitPackedPatchedKernel
            .execute_parent(inner, patched, PatchedSlots::INNER, &mut ctx)?
            .expect("kernel applies to aligned bit-packed children");
        assert_arrays_eq!(fused, expected, &mut ctx);

        // Execution through the session uses the kernel too.
        assert_arrays_eq!(patched.array().clone(), expected, &mut ctx);
        Ok(())
    }
}
