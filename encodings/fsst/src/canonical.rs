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

    // Compute exact total size for the decompression buffer.
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
    let mut uncompressed_bytes = ByteBufferMut::with_capacity(total_size + VIEW_BUILD_PADDING);
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

/// Minimum padding (in bytes) required after the logical end of the source buffer
/// for safe 16-byte unaligned reads in `make_view_inline`.
pub const VIEW_BUILD_PADDING: usize = 16;

/// Optimized view builder for FSST decompression.
///
/// Unlike the general-purpose `build_views`, this version:
/// - Inlines the view construction (avoids `#[inline(never)]` `make_view` call per string)
/// - Skips buffer splitting (FSST data fits in one buffer)
/// - Uses raw pointer writes to construct views directly
/// - Generic over the length type to avoid an intermediate `Vec<usize>` allocation
///
/// # Safety requirement
///
/// `bytes` must have at least [`VIEW_BUILD_PADDING`] bytes of allocated capacity
/// beyond the logical length, to allow safe 16-byte unaligned reads at any offset.
#[allow(clippy::cast_possible_truncation)]
pub fn build_views_fast<P: NativePType + AsPrimitive<usize>>(
    buf_index: u32,
    bytes: ByteBufferMut,
    lens: &[P],
) -> (Vec<ByteBuffer>, Buffer<BinaryView>) {
    let mut views = BufferMut::<BinaryView>::with_capacity(lens.len());
    let src = bytes.as_slice().as_ptr();
    let mut offset: usize = 0;

    for &raw_len in lens {
        let len: usize = raw_len.as_();
        // SAFETY: we reserved the right capacity in `with_capacity` above,
        // and the source buffer has VIEW_BUILD_PADDING bytes of padding.
        unsafe {
            let view = make_view_inline(src, offset, len, buf_index);
            views.push_unchecked(view);
        }
        offset += len;
    }

    let buffers = if bytes.is_empty() {
        Vec::new()
    } else {
        vec![bytes.freeze()]
    };

    (buffers, views.freeze())
}

/// Byte masks for zeroing out trailing bytes when constructing inlined views.
/// `INLINE_MASKS[n]` keeps the lowest `n` bytes of a `u128`.
#[allow(clippy::cast_possible_truncation)]
const INLINE_MASKS: [u128; 13] = {
    let mut table = [0u128; 13];
    let mut i = 1usize;
    while i <= 12 {
        table[i] = (1u128 << (i as u32 * 8)) - 1;
        i += 1;
    }
    table
};

/// Inline view construction — avoids the `#[inline(never)]` overhead of `BinaryView::make_view`.
///
/// For inlined views (len <= 12): performs a single 16-byte unaligned read from the source,
/// masks to `len` bytes, shifts into position, and ORs in the length — no zero-init or
/// variable-length copy needed.
///
/// For reference views (len > 12): reads a 4-byte prefix and constructs the view directly
/// via arithmetic.
///
/// # Safety
///
/// The source buffer must have at least 16 bytes of readable memory from `offset`
/// (i.e., padding after the logical end). The caller must ensure `offset + len <= src.len()`.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
unsafe fn make_view_inline(
    src: *const u8,
    offset: usize,
    len: usize,
    buf_index: u32,
) -> BinaryView {
    if len <= BinaryView::MAX_INLINED_SIZE {
        // Read 16 bytes from source (buffer has >=16 bytes padding, so this is safe).
        // Mask to keep only `len` bytes, shift into data position (bytes 4-15),
        // and OR in the length at bytes 0-3.
        let raw = unsafe { src.add(offset).cast::<u128>().read_unaligned() };
        let masked = raw & INLINE_MASKS[len];
        BinaryView::from((len as u128) | (masked << 32))
    } else {
        // Reference view: [size:u32][prefix:4 bytes][buf_index:u32][offset:u32]
        let prefix = unsafe { src.add(offset).cast::<u32>().read_unaligned() };
        BinaryView::from(
            (len as u128)
                | ((prefix as u128) << 32)
                | ((buf_index as u128) << 64)
                | ((offset as u128) << 96),
        )
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
