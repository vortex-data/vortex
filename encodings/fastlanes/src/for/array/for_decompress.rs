// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use fastlanes::FoR;
use num_traits::PrimInt;
use num_traits::WrappingAdd;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builders::PrimitiveBuilder;
use vortex_array::chunk_iter::ChunkMut;
use vortex_array::chunk_iter::ChunkSink;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::dtype::UnsignedPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::FoRArray;
use crate::bitpack_decompress;
use crate::bitpacking::chunked_decompress;
use crate::r#for::array::FoRArrayExt;
use crate::r#for::array::FoRArraySlotsExt;
use crate::unpack_iter::UnpackStrategy;
use crate::unpack_iter::UnpackedChunks;

/// FoR unpacking strategy that applies a reference value during unpacking.
struct FoRStrategy<T> {
    reference: T,
}

impl<T: PhysicalPType<Physical = T> + FoR> UnpackStrategy<T> for FoRStrategy<T> {
    #[inline(always)]
    unsafe fn unpack_chunk(
        &self,
        bit_width: usize,
        chunk: &[T::Physical],
        dst: &mut [T::Physical],
    ) {
        // SAFETY: Caller ensures chunk and dst have correct sizes.
        unsafe {
            FoR::unchecked_unfor_pack(bit_width, chunk, self.reference, dst);
        }
    }
}

pub fn decompress(array: &FoRArray, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    let ptype = array.ptype();

    // Try to do fused unpack.
    if array.reference_scalar().dtype().is_unsigned_int()
        && let Some(bp) = array.encoded().as_opt::<BitPacked>()
    {
        return match_each_unsigned_integer_ptype!(array.ptype(), |T| {
            fused_decompress::<T>(array, bp, ctx)
        });
    }

    // TODO(ngates): Do we need this to be into_encoded() somehow?
    let encoded = array.encoded().clone().execute::<PrimitiveArray>(ctx)?;
    let validity = encoded.validity()?;

    Ok(match_each_integer_ptype!(ptype, |T| {
        let min = array
            .reference_scalar()
            .as_primitive()
            .typed_value::<T>()
            .vortex_expect("reference must be non-null");
        if min == 0 {
            encoded
        } else {
            PrimitiveArray::new(
                decompress_primitive(encoded.into_buffer::<T>(), min),
                validity,
            )
        }
    }))
}

pub(crate) fn fused_decompress<
    T: PhysicalPType<Physical = T> + UnsignedPType + FoR + WrappingAdd,
>(
    for_: &FoRArray,
    bp: ArrayView<'_, BitPacked>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let ref_ = for_
        .reference_scalar()
        .as_primitive()
        .as_::<T>()
        .vortex_expect("cannot be null");

    let strategy = FoRStrategy { reference: ref_ };

    // Create [`UnpackedChunks`] with FoR strategy.
    let mut unpacked = UnpackedChunks::try_new_with_strategy(
        strategy,
        bp.packed().as_host().clone(),
        bp.bit_width() as usize,
        bp.offset() as usize,
        bp.len(),
    )?;

    let mut builder = PrimitiveBuilder::<T>::with_capacity(
        for_.reference_scalar().dtype().nullability(),
        bp.len(),
    );
    let mut uninit_range = builder.uninit_range(bp.len());
    unsafe {
        // Append a dense null Mask.
        uninit_range.append_mask(&bp.validity()?.execute_mask(bp.as_ref().len(), ctx)?);
    }

    // SAFETY: `decode_into` will initialize all values in this range.
    let uninit_slice = unsafe { uninit_range.slice_uninit_mut(0, bp.len()) };

    // Decode all chunks (initial, full, and trailer) in one call.
    unpacked.decode_into(uninit_slice);

    if let Some(patches) = bp.patches() {
        bitpack_decompress::apply_patches_to_uninit_range(
            &mut uninit_range,
            &patches,
            ctx,
            |v: T| v.wrapping_add(&ref_),
        )?;
    };

    // SAFETY: We have set a correct validity mask via `append_mask` with `array.len()` values and
    // initialized the same number of values needed via `decode_into`.
    unsafe {
        uninit_range.finish();
    }

    Ok(builder.finish_into_primitive())
}

/// Whether [`decompress_chunks`] can stream: either the fused FoR+BitPacked path applies, or the
/// encoded child itself supports streaming (generic composition).
pub(crate) fn supports_decompress_chunks(array: ArrayView<'_, crate::FoR>) -> bool {
    (array.reference_scalar().dtype().is_unsigned_int() && array.encoded().is::<BitPacked>())
        || array.encoded().supports_decompress_chunks()
}

/// Streaming chunked decompression for FoR arrays.
///
/// When the child is a [`BitPacked`] array (the common layout), this streams through the fused
/// [`FoRStrategy`] unpack kernel — the reference is folded into `unchecked_unfor_pack` itself,
/// exactly like [`fused_decompress`], so there is no extra add pass at all.
///
/// Otherwise FoR composes generically over any encoded child: each chunk the child streams up is
/// shifted by the reference value in place (one pass over an L1-resident chunk) and forwarded to
/// the downstream sink. The adapter lives on this stack frame — no heap state is added on the
/// way down.
pub(crate) fn decompress_chunks(
    array: ArrayView<'_, crate::FoR>,
    ctx: &mut ExecutionCtx,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()> {
    // Fused fast path, mirroring the dispatch in `decompress`.
    if array.reference_scalar().dtype().is_unsigned_int()
        && let Some(bp) = array.encoded().as_opt::<BitPacked>()
    {
        return match_each_unsigned_integer_ptype!(array.ptype(), |T| {
            fused_decompress_chunks::<T>(array, bp, ctx, sink)
        });
    }

    match_each_integer_ptype!(array.ptype(), |T| {
        let reference = array
            .reference_scalar()
            .as_primitive()
            .typed_value::<T>()
            .vortex_expect("reference must be non-null");
        if reference == 0 {
            array.encoded().decompress_chunks(ctx, sink)
        } else {
            let mut adapter = AddReferenceSink {
                reference,
                inner: sink,
            };
            array.encoded().decompress_chunks(ctx, &mut adapter)
        }
    })
}

/// Stream FoR-over-BitPacked chunks through the fused unpack kernel: each FastLanes block is
/// unpacked with the reference added by `unchecked_unfor_pack`, patched in place (patch values
/// get the reference applied up front), and handed to the sink.
fn fused_decompress_chunks<T: PhysicalPType<Physical = T> + UnsignedPType + FoR + WrappingAdd>(
    for_: ArrayView<'_, crate::FoR>,
    bp: ArrayView<'_, BitPacked>,
    ctx: &mut ExecutionCtx,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()> {
    if bp.as_ref().is_empty() {
        return Ok(());
    }

    let ref_ = for_
        .reference_scalar()
        .as_primitive()
        .as_::<T>()
        .vortex_expect("cannot be null");

    let mut unpacked = UnpackedChunks::try_new_with_strategy(
        FoRStrategy { reference: ref_ },
        bp.packed().as_host().clone(),
        bp.bit_width() as usize,
        bp.offset() as usize,
        bp.len(),
    )?;

    let patch_list = match bp.patches() {
        None => Vec::new(),
        Some(patches) => {
            chunked_decompress::build_patch_list(&patches, ctx, |v: T| v.wrapping_add(&ref_))?
        }
    };

    chunked_decompress::stream_unpacked_chunks(&mut unpacked, &patch_list, sink)
}

struct AddReferenceSink<'a, T> {
    reference: T,
    inner: &'a mut dyn ChunkSink,
}

impl<T: NativePType + WrappingAdd> ChunkSink for AddReferenceSink<'_, T> {
    #[inline]
    fn accept(&mut self, mut chunk: ChunkMut<'_>, row_range: Range<usize>) -> VortexResult<()> {
        for v in chunk.as_slice_mut::<T>() {
            *v = v.wrapping_add(&self.reference);
        }
        self.inner.accept(chunk, row_range)
    }
}

fn decompress_primitive<T: NativePType + WrappingAdd + PrimInt>(
    values: Buffer<T>,
    min: T,
) -> Buffer<T> {
    values
        .map_each_in_place(move |v| v.wrapping_add(&min))
        .freeze()
}
