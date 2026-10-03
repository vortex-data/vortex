// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streaming chunked decompression for ZigZag arrays: each chunk of unsigned integers streamed up
//! from the child is decoded into signed integers in place.

use std::marker::PhantomData;
use std::ops::Range;

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::chunk_iter::ChunkMut;
use vortex_array::chunk_iter::ChunkSink;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use zigzag::ZigZag as ExternalZigZag;

use crate::ZigZag;
use crate::ZigZagArraySlotsExt;

pub(crate) fn supports_decompress_chunks(array: ArrayView<'_, ZigZag>) -> bool {
    array.encoded().supports_decompress_chunks()
}

pub(crate) fn decompress_chunks(
    array: ArrayView<'_, ZigZag>,
    ctx: &mut ExecutionCtx,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()> {
    // Each decode reads the unsigned encoding from the bits of a signed value of the same width.
    match array.dtype().as_ptype() {
        PType::I8 => decompress_chunks_typed(array, ctx, sink, |v: i8| i8::decode(v as u8)),
        PType::I16 => decompress_chunks_typed(array, ctx, sink, |v: i16| i16::decode(v as u16)),
        PType::I32 => decompress_chunks_typed(array, ctx, sink, |v: i32| i32::decode(v as u32)),
        PType::I64 => decompress_chunks_typed(array, ctx, sink, |v: i64| i64::decode(v as u64)),
        ptype => vortex_bail!("ZigZag can only decode into signed integers, got {ptype}"),
    }
}

fn decompress_chunks_typed<T: NativePType>(
    array: ArrayView<'_, ZigZag>,
    ctx: &mut ExecutionCtx,
    sink: &mut dyn ChunkSink,
    decode: impl Fn(T) -> T,
) -> VortexResult<()> {
    let mut adapter = DecodeSink {
        decode,
        inner: sink,
        _value: PhantomData,
    };
    array.encoded().decompress_chunks(ctx, &mut adapter)
}

/// Decodes each chunk of unsigned integers into the signed integers of the same width in place.
struct DecodeSink<'a, T, F> {
    decode: F,
    inner: &'a mut dyn ChunkSink,
    _value: PhantomData<T>,
}

impl<T, F> ChunkSink for DecodeSink<'_, T, F>
where
    T: NativePType,
    F: Fn(T) -> T,
{
    #[inline]
    fn accept(&mut self, chunk: ChunkMut<'_>, rows: Range<usize>) -> VortexResult<()> {
        let mut chunk = chunk.retype::<T>();
        for value in chunk.as_slice_mut::<T>() {
            *value = (self.decode)(*value);
        }
        self.inner.accept(chunk, rows)
    }

    /// The child writes unsigned encodings into the signed destination, decoded in place after.
    fn destination(&mut self, rows: Range<usize>) -> Option<ChunkMut<'_>> {
        self.inner
            .destination(rows)
            .map(|chunk| chunk.retype_to(T::PTYPE.to_unsigned()))
    }

    fn accept_written(&mut self, rows: Range<usize>) -> VortexResult<()> {
        let mut chunk = self
            .inner
            .destination(rows.clone())
            .ok_or_else(|| vortex_err!("ZigZag's destination for rows {rows:?} is gone"))?
            .retype::<T>();
        for value in chunk.as_slice_mut::<T>() {
            *value = (self.decode)(*value);
        }
        self.inner.accept_written(rows)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::test_harness::assert_streams_like_execute;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::zigzag_encode;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    fn zigzag_streams_like_execute() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_option_iter(
            (-2500i32..2500).map(|i| (i % 11 != 0).then_some(i * 37)),
        );
        let array = zigzag_encode(values.as_view())?.into_array();
        assert_streams_like_execute(&array, &mut ctx)?;

        let values = PrimitiveArray::from_iter([i8::MIN, -1, 0, 1, i8::MAX]);
        assert_streams_like_execute(&zigzag_encode(values.as_view())?.into_array(), &mut ctx)
    }
}
