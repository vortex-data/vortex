// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streaming chunked decompression for sparse primitive arrays: each chunk is the fill value with
//! the patches that fall in it written over, so the full-length fill buffer is never written.

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::chunk_iter::ChunkPatches;
use vortex_array::chunk_iter::ChunkSink;
use vortex_array::chunk_iter::stream_from_fn;
use vortex_array::match_each_native_ptype;
use vortex_error::VortexResult;

use crate::Sparse;
use crate::SparseExt;

pub(crate) fn supports_decompress_chunks(_array: ArrayView<'_, Sparse>) -> bool {
    true
}

pub(crate) fn decompress_chunks(
    array: ArrayView<'_, Sparse>,
    ctx: &mut ExecutionCtx,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()> {
    let patches = array.patches();
    match_each_native_ptype!(array.dtype().as_ptype(), |T| {
        // A null fill streams unspecified (but initialized) values, as validity is not streamed.
        let fill = array
            .fill_scalar()
            .as_primitive()
            .typed_value::<T>()
            .unwrap_or_default();
        let mut patches = ChunkPatches::try_from_patches(&patches, ctx, |_, value: T| value)?;
        stream_from_fn(array.len(), sink, |chunk: &mut [T], rows| {
            chunk.fill(fill);
            patches.apply(chunk, rows.start);
            Ok(())
        })
    })
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_array::test_harness::assert_streams_like_execute;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Sparse;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[rstest]
    #[case::fill(Scalar::from(Some(7i64)))]
    #[case::null_fill(Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable)))]
    fn sparse_streams_like_execute(#[case] fill: Scalar) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let indices = buffer![0u32, 1023, 1024, 2047, 3000, 4999].into_array();
        let values = vortex_array::arrays::PrimitiveArray::from_option_iter([
            Some(1i64),
            Some(2),
            None,
            Some(4),
            Some(5),
            Some(6),
        ])
        .into_array();
        let array = Sparse::try_new(indices, values, 5000, fill)?.into_array();
        assert_streams_like_execute(&array, &mut ctx)?;
        // Slicing leaves the patches with an offset.
        assert_streams_like_execute(&array.slice(1000..4500)?, &mut ctx)
    }
}
