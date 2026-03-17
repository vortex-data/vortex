// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::build_views::BinaryView;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_array::vtable::ValidityHelper;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;

use crate::FSSTArray;
use crate::decompressor::OptimizedDecompressor;

pub(super) fn canonicalize_fsst(
    array: &FSSTArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let (buffers, views) = fsst_decode_views(array, 0, ctx)?;
    // SAFETY: FSST already validates the bytes for binary/UTF-8. We build views directly on
    //  top of them, so the view pointers will all be valid.
    Ok(unsafe {
        VarBinViewArray::new_unchecked(
            views,
            Arc::from(buffers),
            array.dtype().clone(),
            array.codes().validity().clone(),
        )
        .into_array()
    })
}

pub(crate) fn fsst_decode_views(
    fsst_array: &FSSTArray,
    start_buf_index: u32,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(Vec<ByteBuffer>, Buffer<BinaryView>)> {
    let bytes = fsst_array.codes().sliced_bytes();

    let uncompressed_lens_array = fsst_array
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;

    // Single pass over lengths: compute total_size for decompression buffer capacity.
    #[allow(clippy::cast_possible_truncation)]
    let total_size: usize = match_each_integer_ptype!(uncompressed_lens_array.ptype(), |P| {
        uncompressed_lens_array
            .as_slice::<P>()
            .iter()
            .map(|x| *x as usize)
            .sum()
    });

    // Bulk-decompress the entire string heap in one call.
    let decompressor = OptimizedDecompressor::new(
        fsst_array.symbols().as_slice(),
        fsst_array.symbol_lengths().as_slice(),
    );
    let mut uncompressed_bytes = ByteBufferMut::with_capacity(total_size + 7);
    let len =
        decompressor.decompress_into(bytes.as_slice(), uncompressed_bytes.spare_capacity_mut());
    unsafe { uncompressed_bytes.set_len(len) };

    // Build views directly from the typed lengths slice — no intermediate Vec<usize> allocation.
    match_each_integer_ptype!(uncompressed_lens_array.ptype(), |P| {
        Ok(build_views_fast(
            start_buf_index,
            uncompressed_bytes,
            uncompressed_lens_array.as_slice::<P>(),
        ))
    })
}

/// Optimized view builder for FSST decompression.
///
/// Unlike the general-purpose `build_views`, this version:
/// - Inlines the view construction (avoids `#[inline(never)]` `make_view` call per string)
/// - Skips buffer splitting (FSST data fits in one buffer)
/// - Uses raw pointer writes to construct views directly
/// - Generic over the length type to avoid an intermediate `Vec<usize>` allocation
#[allow(clippy::cast_possible_truncation)]
fn build_views_fast<P: NativePType + AsPrimitive<usize>>(
    buf_index: u32,
    bytes: ByteBufferMut,
    lens: &[P],
) -> (Vec<ByteBuffer>, Buffer<BinaryView>) {
    let mut views = BufferMut::<BinaryView>::with_capacity(lens.len());
    let src = bytes.as_slice();
    let mut offset: usize = 0;

    for &raw_len in lens {
        let len: usize = raw_len.as_();
        // SAFETY: we reserved the right capacity in `with_capacity` above.
        unsafe {
            let view = make_view_inline(src, offset, len, buf_index);
            views.push_unchecked(view);
        }
        offset += len;
    }

    debug_assert_eq!(offset, src.len(), "lengths must sum to total buffer size");

    let buffers = if bytes.is_empty() {
        Vec::new()
    } else {
        vec![bytes.freeze()]
    };

    (buffers, views.freeze())
}

/// Inline view construction — avoids the `#[inline(never)]` overhead of `BinaryView::make_view`.
///
/// Constructs the 16-byte view directly via `u128` to bypass private field access.
/// Layout (little-endian):
/// - Inlined (len <= 12): [size:u32][data:12 bytes]
/// - Reference (len > 12): [size:u32][prefix:4 bytes][buf_index:u32][offset:u32]
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
unsafe fn make_view_inline(src: &[u8], offset: usize, len: usize, buf_index: u32) -> BinaryView {
    debug_assert!(offset + len <= src.len());

    if len <= BinaryView::MAX_INLINED_SIZE {
        // Inlined: zero 16 bytes, write size at byte 0, copy data at byte 4.
        let mut bytes = [0u8; 16];
        bytes[..4].copy_from_slice(&(len as u32).to_le_bytes());
        // SAFETY: len <= 12, and src[offset..offset+len] is valid.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr().add(offset), bytes.as_mut_ptr().add(4), len);
        }
        BinaryView::from(u128::from_le_bytes(bytes))
    } else {
        // Reference: size + 4-byte prefix + buffer index + offset.
        let mut bytes = [0u8; 16];
        bytes[..4].copy_from_slice(&(len as u32).to_le_bytes());
        // SAFETY: len > 12 so there are at least 4 bytes at src[offset..].
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr().add(offset), bytes.as_mut_ptr().add(4), 4);
        }
        bytes[8..12].copy_from_slice(&buf_index.to_le_bytes());
        bytes[12..16].copy_from_slice(&(offset as u32).to_le_bytes());
        BinaryView::from(u128::from_le_bytes(bytes))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rand::Rng;
    use rand::SeedableRng;
    use rand::prelude::StdRng;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::accessor::ArrayAccessor;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::builders::ArrayBuilder;
    use vortex_array::builders::VarBinViewBuilder;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::fsst_compress;
    use crate::fsst_train_compressor;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    fn make_data() -> (VarBinArray, Vec<Option<Vec<u8>>>) {
        const STRING_COUNT: usize = 1000;
        let mut rng = StdRng::seed_from_u64(0);
        let mut strings = Vec::with_capacity(STRING_COUNT);

        for _ in 0..STRING_COUNT {
            if rng.random_bool(0.9) {
                strings.push(None)
            } else {
                // Generate a random string with length around `avg_len`. The number of possible
                // characters within the random string is defined by `unique_chars`.
                let len = 10 * rng.random_range(50..=150) / 100;
                strings.push(Some(
                    (0..len)
                        .map(|_| rng.random_range(b'a'..=b'z') as char)
                        .collect::<String>()
                        .into_bytes(),
                ));
            }
        }

        (
            VarBinArray::from_iter(
                strings
                    .clone()
                    .into_iter()
                    .map(|opt_s| opt_s.map(Vec::into_boxed_slice)),
                DType::Binary(Nullability::Nullable),
            ),
            strings,
        )
    }

    fn make_data_chunked() -> (ChunkedArray, Vec<Option<Vec<u8>>>) {
        #[allow(clippy::type_complexity)]
        let (arr_vec, data_vec): (Vec<ArrayRef>, Vec<Vec<Option<Vec<u8>>>>) = (0..10)
            .map(|_| {
                let (array, data) = make_data();
                let compressor = fsst_train_compressor(&array);
                (fsst_compress(&array, &compressor).into_array(), data)
            })
            .unzip();

        (
            ChunkedArray::from_iter(arr_vec),
            data_vec.into_iter().flatten().collect(),
        )
    }

    #[test]
    fn test_to_canonical() -> VortexResult<()> {
        let (chunked_arr, data) = make_data_chunked();

        let mut builder =
            VarBinViewBuilder::with_capacity(chunked_arr.dtype().clone(), chunked_arr.len());
        chunked_arr.append_to_builder(&mut builder, &mut SESSION.create_execution_ctx())?;

        {
            let arr = builder.finish_into_canonical().into_varbinview();
            let res1 =
                arr.with_iterator(|iter| iter.map(|b| b.map(|v| v.to_vec())).collect::<Vec<_>>());
            assert_eq!(data, res1);
        };

        {
            let arr2 = chunked_arr.to_varbinview();
            let res2 =
                arr2.with_iterator(|iter| iter.map(|b| b.map(|v| v.to_vec())).collect::<Vec<_>>());
            assert_eq!(data, res2)
        };
        Ok(())
    }
}
