// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbin::VarBinArrayExt;
use vortex_array::arrays::varbinview::build_views::BinaryView;
use vortex_array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex_array::arrays::varbinview::build_views::build_views;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;

use crate::FSST;
use crate::FSSTArrayExt;

pub(super) fn canonicalize_fsst(
    array: ArrayView<'_, FSST>,
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
            array.codes().validity()?,
        )
        .into_array()
    })
}

pub(crate) fn fsst_decode_views(
    fsst_array: ArrayView<'_, FSST>,
    start_buf_index: u32,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(Vec<ByteBuffer>, Buffer<BinaryView>)> {
    // FSSTArray has two child arrays:
    //  1. A VarBinArray, which holds the string heap of the compressed codes.
    //  2. An uncompressed_lengths primitive array, storing the length of each original
    //     string element.
    // To speed up canonicalization, we can decompress the entire string-heap in a single
    // call. We then turn our uncompressed_lengths into an offsets buffer
    // necessary for a VarBinViewArray and construct the canonical array.
    let bytes = fsst_array.codes().sliced_bytes();

    let uncompressed_lens_array = fsst_array
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;

    #[expect(clippy::cast_possible_truncation)]
    let total_size: usize = match_each_integer_ptype!(uncompressed_lens_array.ptype(), |P| {
        uncompressed_lens_array
            .as_slice::<P>()
            .iter()
            .map(|x| *x as usize)
            .sum()
    });

    // Bulk-decompress the entire array.
    let decompressor = fsst_array.decompressor();
    let mut uncompressed_bytes = ByteBufferMut::with_capacity(total_size + 7);
    let len =
        decompressor.decompress_into(bytes.as_slice(), uncompressed_bytes.spare_capacity_mut());
    unsafe { uncompressed_bytes.set_len(len) };

    // Directly create the binary views.
    match_each_integer_ptype!(uncompressed_lens_array.ptype(), |P| {
        Ok(build_views(
            start_buf_index,
            MAX_BUFFER_LEN,
            uncompressed_bytes,
            uncompressed_lens_array.as_slice::<P>(),
        ))
    })
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rand::RngExt;
    use rand::SeedableRng;
    use rand::prelude::StdRng;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::accessor::ArrayAccessor;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::arrays::VarBinViewArray;
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
        let mut ctx = SESSION.create_execution_ctx();
        #[expect(clippy::type_complexity)]
        let (arr_vec, data_vec): (Vec<ArrayRef>, Vec<Vec<Option<Vec<u8>>>>) = (0..10)
            .map(|_| {
                let (array, data) = make_data();
                let compressor = fsst_train_compressor(&array);
                (
                    fsst_compress(&array, array.len(), array.dtype(), &compressor, &mut ctx)
                        .into_array(),
                    data,
                )
            })
            .unzip();

        (
            ChunkedArray::from_iter(arr_vec),
            data_vec.into_iter().flatten().collect(),
        )
    }

    #[test]
    fn test_to_canonical() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let (chunked_arr, data) = make_data_chunked();

        let mut builder =
            VarBinViewBuilder::with_capacity(chunked_arr.dtype().clone(), chunked_arr.len());
        chunked_arr
            .clone()
            .into_array()
            .append_to_builder(&mut builder, &mut ctx)?;

        {
            let arr = builder.finish_into_canonical().into_varbinview();
            let res1 =
                arr.with_iterator(|iter| iter.map(|b| b.map(|v| v.to_vec())).collect::<Vec<_>>());
            assert_eq!(data, res1);
        };

        {
            let arr2 = chunked_arr
                .as_array()
                .clone()
                .execute::<VarBinViewArray>(&mut ctx)?;
            let res2 =
                arr2.with_iterator(|iter| iter.map(|b| b.map(|v| v.to_vec())).collect::<Vec<_>>());
            assert_eq!(data, res2)
        };
        Ok(())
    }

    #[test]
    fn test_append_after_in_progress_buffer() -> VortexResult<()> {
        let dtype = DType::Binary(Nullability::NonNullable);
        let mut builder = VarBinViewBuilder::with_capacity(dtype.clone(), 2);
        builder.append_value(b"long enough!!!");

        let varbin = VarBinArray::from_iter(
            [Some(b"long enough too".to_vec().into_boxed_slice())],
            dtype,
        );
        let mut ctx = SESSION.create_execution_ctx();
        let fsst_array = fsst_compress(
            &varbin,
            varbin.len(),
            varbin.dtype(),
            &fsst_train_compressor(&varbin),
            &mut ctx,
        )
        .into_array();
        fsst_array.append_to_builder(&mut builder, &mut ctx)?;

        let _result = builder.finish_into_varbinview();
        Ok(())
    }
}
