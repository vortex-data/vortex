// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streaming chunked decompression for chunked arrays: each chunk streams in turn, with its rows
//! shifted by the chunk's offset.

use std::marker::PhantomData;
use std::ops::Range;

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::Chunked;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::chunk_iter::ChunkMut;
use crate::chunk_iter::ChunkSink;
use crate::chunk_iter::ChunkValue;
use crate::chunk_iter::ScratchChunk;
use crate::chunk_iter::ValueType;
use crate::chunk_iter::emit_with;
use crate::dtype::BigCast;
use crate::dtype::DType;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::match_each_decimal_value_type;

pub(super) fn decompress_chunks_type(array: ArrayView<'_, Chunked>) -> Option<ValueType> {
    if !array
        .iter_chunks()
        .all(|chunk| chunk.supports_decompress_chunks())
    {
        return None;
    }
    match array.dtype() {
        // Executing a single chunk executes the chunk, and executing several builds the smallest
        // type for the precision, whatever each chunk stores.
        DType::Decimal(decimal_dtype, _) if array.nchunks() != 1 => {
            Some(DecimalType::smallest_decimal_value_type(decimal_dtype).into())
        }
        DType::Decimal(..) => array.chunk(0).decompress_chunks_type(),
        dtype => ValueType::primitive(dtype),
    }
}

pub(super) fn decompress_chunks(
    array: ArrayView<'_, Chunked>,
    ctx: &mut ExecutionCtx,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()> {
    // Decimal chunks may store their values in another type than the array streams.
    let decimal_type = match array.dtype() {
        DType::Decimal(..) => decompress_chunks_type(array).and_then(ValueType::decimal_type),
        _ => None,
    };
    let mut adapter = OffsetSink {
        start: 0,
        inner: sink,
    };
    for chunk in array.non_empty_chunks() {
        let chunk_type = decimal_type
            .and_then(|_| chunk.decompress_chunks_type())
            .and_then(ValueType::decimal_type);
        match (chunk_type, decimal_type) {
            // Convert the chunk's values, as appending it to a builder of the array's type would.
            (Some(from), Some(to)) if from != to => {
                match_each_decimal_value_type!(from, |F| {
                    match_each_decimal_value_type!(to, |T| {
                        let mut convert = ConvertDecimalSink::<F, T> {
                            scratch: ScratchChunk::new(),
                            inner: &mut adapter,
                            _from: PhantomData,
                        };
                        chunk.decompress_child_chunks(ctx, &mut convert)?;
                    })
                });
            }
            _ => chunk.decompress_child_chunks(ctx, &mut adapter)?,
        }
        adapter.start += chunk.len();
    }
    Ok(())
}

/// Converts each chunk of decimal values stored as `F` to `T`.
struct ConvertDecimalSink<'a, F, T> {
    scratch: ScratchChunk<T>,
    inner: &'a mut dyn ChunkSink,
    _from: PhantomData<F>,
}

impl<F, T> ChunkSink for ConvertDecimalSink<'_, F, T>
where
    F: NativeDecimalType + ChunkValue,
    T: NativeDecimalType + ChunkValue,
{
    fn accept(&mut self, chunk: ChunkMut<'_>, rows: Range<usize>) -> VortexResult<()> {
        let values = chunk.as_slice::<F>();
        emit_with(
            &mut *self.inner,
            T::VALUE_TYPE,
            rows,
            &mut self.scratch,
            |out| {
                for (out, &value) in out.iter_mut().zip(values) {
                    // Every valid value fits the precision, so only a null's unspecified value can
                    // fail to convert.
                    *out = <T as BigCast>::from(value).unwrap_or_default();
                }
                Ok(())
            },
        )
    }
}

/// Shifts the rows of a chunk's stream to the chunk's position in the array.
struct OffsetSink<'a> {
    start: usize,
    inner: &'a mut dyn ChunkSink,
}

impl ChunkSink for OffsetSink<'_> {
    #[inline]
    fn accept(&mut self, chunk: ChunkMut<'_>, rows: Range<usize>) -> VortexResult<()> {
        self.inner
            .accept(chunk, self.start + rows.start..self.start + rows.end)
    }

    fn destination(&mut self, rows: Range<usize>) -> Option<ChunkMut<'_>> {
        self.inner
            .destination(self.start + rows.start..self.start + rows.end)
    }

    fn accept_written(&mut self, rows: Range<usize>) -> VortexResult<()> {
        self.inner
            .accept_written(self.start + rows.start..self.start + rows.end)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::test_harness::assert_streams_like_execute;

    #[test]
    fn chunked_streams_like_execute() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let chunks = [
            PrimitiveArray::from_option_iter((0..1500i32).map(|i| (i % 3 != 0).then_some(i)))
                .into_array(),
            PrimitiveArray::from_option_iter(Vec::<Option<i32>>::new()).into_array(),
            ConstantArray::new(Some(-7i32), 2100).into_array(),
            PrimitiveArray::new(buffer![1i32, 2, 3], crate::validity::Validity::AllValid)
                .into_array(),
        ];
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let array = ChunkedArray::try_new(chunks, dtype)?.into_array();
        assert_streams_like_execute(&array, &mut ctx)
    }
}
