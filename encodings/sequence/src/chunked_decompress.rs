// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streaming chunked decompression for sequence arrays: each chunk is generated in place.

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::chunk_iter::ChunkSink;
use vortex_array::chunk_iter::stream_from_fn;
use vortex_array::match_each_integer_ptype;
use vortex_error::VortexResult;

use crate::Sequence;
use crate::eval;

pub(crate) fn supports_decompress_chunks(_array: ArrayView<'_, Sequence>) -> bool {
    true
}

pub(crate) fn decompress_chunks(
    array: ArrayView<'_, Sequence>,
    _ctx: &mut ExecutionCtx,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()> {
    match_each_integer_ptype!(array.dtype().as_ptype(), |O| {
        let (base, multiplier) = array.wrapping_parts::<O>()?;
        stream_from_fn(array.len(), sink, |chunk: &mut [O], rows| {
            let mut value = eval::wrapping_value(base, multiplier, rows.start);
            for out in chunk {
                *out = value;
                value = value.wrapping_add(multiplier);
            }
            Ok(())
        })
    })
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::dtype::Nullability;
    use vortex_array::test_harness::assert_streams_like_execute;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Sequence;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    fn sequence_streams_like_execute() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array =
            Sequence::try_new_typed(-1_000i64, 3, Nullability::Nullable, 5000)?.into_array();
        assert_streams_like_execute(&array, &mut ctx)?;
        assert_streams_like_execute(&array.slice(517..4013)?, &mut ctx)?;

        let array =
            Sequence::try_new_typed(-30_000i16, 20, Nullability::NonNullable, 3000)?.into_array();
        assert_streams_like_execute(&array, &mut ctx)
    }
}
