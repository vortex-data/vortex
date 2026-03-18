// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::build_views::BinaryView;
use vortex_array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex_array::arrays::varbinview::build_views::build_views;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::vtable::ValidityHelper;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;

use crate::FSSTArray;

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

    // Fast path: try to downcast uncompressed_lengths directly to PrimitiveArray,
    // avoiding the execute() overhead when it's already in primitive form.
    // The compression path always writes i32 lengths, so this succeeds in the common case.
    if let Some(parray) = fsst_array.uncompressed_lengths().as_opt::<Primitive>() {
        return decompress_and_build_views(
            &fsst_array.decompressor(),
            bytes.as_slice(),
            parray,
            start_buf_index,
        );
    }

    // Slow path: lengths are compressed, need to execute to get PrimitiveArray.
    let uncompressed_lens_array = fsst_array
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;

    decompress_and_build_views(
        &fsst_array.decompressor(),
        bytes.as_slice(),
        &uncompressed_lens_array,
        start_buf_index,
    )
}

/// Core decompress + view building, split out to avoid duplicating logic between
/// the fast (direct downcast) and slow (execute) paths.
#[inline]
fn decompress_and_build_views(
    decompressor: &fsst::Decompressor<'_>,
    compressed: &[u8],
    lens_array: &PrimitiveArray,
    start_buf_index: u32,
) -> VortexResult<(Vec<ByteBuffer>, Buffer<BinaryView>)> {
    match_each_integer_ptype!(lens_array.ptype(), |P| {
        let lens = lens_array.as_slice::<P>();
        #[allow(clippy::cast_possible_truncation)]
        let total_size: usize = lens.iter().map(|x| *x as usize).sum();

        let mut uncompressed_bytes = ByteBufferMut::with_capacity(total_size + 7);
        let len = decompressor.decompress_into(compressed, uncompressed_bytes.spare_capacity_mut());
        unsafe { uncompressed_bytes.set_len(len) };

        Ok(build_views(
            start_buf_index,
            MAX_BUFFER_LEN,
            uncompressed_bytes,
            lens,
        ))
    })
}

/// Batch-decompress multiple FSST chunks into a single contiguous buffer,
/// producing one combined VarBinViewArray.
///
/// Compared to decompressing chunks individually (one allocation per chunk),
/// this performs a single allocation for the entire decompressed output. The
/// resulting VarBinViewArray references at most one data buffer, which is
/// beneficial for downstream consumers that benefit from fewer buffer references.
pub fn fsst_batch_decode(
    chunks: &[&FSSTArray],
    ctx: &mut ExecutionCtx,
) -> VortexResult<(Vec<ByteBuffer>, Buffer<BinaryView>)> {
    if chunks.is_empty() {
        return Ok((Vec::new(), BufferMut::<BinaryView>::empty().freeze()));
    }

    if chunks.len() == 1 {
        return fsst_decode_views(chunks[0], 0, ctx);
    }

    // Phase 1: Resolve all length arrays and compute total decompressed size.
    let mut resolved: Vec<(PrimitiveArray, usize)> = Vec::with_capacity(chunks.len());
    let mut total_decompressed_size: usize = 0;
    let mut total_elements: usize = 0;

    for chunk in chunks {
        let parray = resolve_primitive_lengths(chunk, ctx)?;
        #[allow(clippy::cast_possible_truncation)]
        let chunk_size: usize = match_each_integer_ptype!(parray.ptype(), |P| {
            parray.as_slice::<P>().iter().map(|x| *x as usize).sum()
        });
        total_decompressed_size += chunk_size;
        total_elements += chunk.len();
        resolved.push((parray, chunk_size));
    }

    // Phase 2: Allocate a single buffer and decompress each chunk into it sequentially.
    let mut combined_buf = ByteBufferMut::with_capacity(total_decompressed_size + 7);
    let mut all_lens: BufferMut<i32> = BufferMut::with_capacity(total_elements);

    for (chunk, (parray, _chunk_size)) in chunks.iter().zip(resolved.iter()) {
        let compressed = chunk.codes().sliced_bytes();
        let decompressor = chunk.decompressor();

        // Decompress this chunk, appending to combined_buf.
        let decompressed_len =
            decompressor.decompress_into(compressed.as_slice(), combined_buf.spare_capacity_mut());
        let prev_len = combined_buf.len();
        unsafe { combined_buf.set_len(prev_len + decompressed_len) };

        // Collect this chunk's lengths as i32 into the combined lens buffer.
        // Fast path: when lengths are already i32 (the common case from compression),
        // use memcpy via extend_from_slice instead of per-element push with cast.
        if parray.ptype() == PType::I32 {
            all_lens.extend_from_slice(parray.as_slice::<i32>());
        } else {
            match_each_integer_ptype!(parray.ptype(), |P| {
                for &l in parray.as_slice::<P>() {
                    #[allow(clippy::cast_possible_truncation, clippy::unnecessary_cast)]
                    all_lens.push(l as i32);
                }
            });
        }
    }

    // Phase 3: Build views over the single combined buffer.
    Ok(build_views(
        0,
        MAX_BUFFER_LEN,
        combined_buf,
        all_lens.as_slice(),
    ))
}

/// Resolve the uncompressed_lengths child to a PrimitiveArray, using direct downcast
/// when possible to avoid execute() overhead.
fn resolve_primitive_lengths(
    chunk: &FSSTArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    if let Some(p) = chunk.uncompressed_lengths().as_opt::<Primitive>() {
        return Ok(p.clone());
    }
    chunk
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(ctx)
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
    fn test_batch_decode() -> VortexResult<()> {
        use vortex_array::accessor::ArrayAccessor;

        // Create non-nullable test data so validity is simple.
        let strings: Vec<Option<Box<[u8]>>> = (0..100)
            .map(|i| {
                Some(
                    format!("string_{i}_padding")
                        .into_bytes()
                        .into_boxed_slice(),
                )
            })
            .collect();
        let array = VarBinArray::from_iter(strings, DType::Binary(Nullability::NonNullable));

        // Create 3 FSST chunks from the same data.
        let compressor = fsst_train_compressor(&array);
        let chunk1 = fsst_compress(&array, &compressor);
        let chunk2 = fsst_compress(&array, &compressor);
        let chunk3 = fsst_compress(&array, &compressor);

        let chunks: Vec<&crate::FSSTArray> = vec![&chunk1, &chunk2, &chunk3];

        let (buffers, views) =
            super::fsst_batch_decode(&chunks, &mut SESSION.create_execution_ctx())?;

        let arr = unsafe {
            vortex_array::arrays::VarBinViewArray::new_unchecked(
                views,
                std::sync::Arc::from(buffers),
                DType::Binary(Nullability::NonNullable),
                vortex_array::validity::Validity::NonNullable,
            )
        };

        assert_eq!(arr.len(), 300);

        // Verify each chunk's data is correct.
        let result: Vec<Option<Vec<u8>>> =
            arr.with_iterator(|iter| iter.map(|b| b.map(|v| v.to_vec())).collect());
        for chunk_idx in 0..3 {
            for i in 0..100 {
                let expected = format!("string_{i}_padding").into_bytes();
                assert_eq!(
                    result[chunk_idx * 100 + i].as_deref(),
                    Some(expected.as_slice())
                );
            }
        }

        Ok(())
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
