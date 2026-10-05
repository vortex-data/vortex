// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streaming chunked decompression for decimal byte parts.
//!
//! Without lower parts, the most significant part's integers are the decimal's values, so its
//! stream is forwarded as is. With lower parts, the most significant part streams and each chunk
//! is assembled with the lower parts' words for its rows, straight into the output when
//! materializing, so the most significant part is never materialized. The lower parts, full
//! 64-bit words that rarely compress, are read from their executed buffers.

use std::marker::PhantomData;
use std::ops::BitOr;
use std::ops::Range;
use std::ops::Shl;

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::chunk_iter::ChunkMut;
use vortex_array::chunk_iter::ChunkSink;
use vortex_array::chunk_iter::ChunkValue;
use vortex_array::chunk_iter::ScratchChunk;
use vortex_array::chunk_iter::ValueType;
use vortex_array::chunk_iter::emit_with;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::i256;
use vortex_array::match_each_signed_integer_ptype;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::DecimalByteParts;
use super::DecimalBytePartsArraySlotsExt;
use super::LOWER_PART_DTYPE;
use super::assemble::assemble_wide_decimal_value;

pub(super) fn decompress_chunks_type(array: ArrayView<'_, DecimalByteParts>) -> Option<ValueType> {
    let msp_type = array.msp().decompress_chunks_type()?;
    // As executing assembles them: in the most significant part's type, or 128 or 256 bits.
    Some(match array.lower_parts().len() {
        0 => msp_type,
        1 => ValueType::I128,
        _ => ValueType::I256,
    })
}

pub(super) fn decompress_chunks(
    array: ArrayView<'_, DecimalByteParts>,
    ctx: &mut ExecutionCtx,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()> {
    let msp = array.msp();
    if array.lower_parts().is_empty() {
        return msp.decompress_child_chunks(ctx, sink);
    }
    // Widened to whole words, as executing assembles them.
    let lower = array
        .lower_parts()
        .iter()
        .map(|part| part.cast(LOWER_PART_DTYPE)?.execute::<PrimitiveArray>(ctx))
        .collect::<VortexResult<Vec<_>>>()?;
    match_each_signed_integer_ptype!(msp.dtype().as_ptype(), |Msp| {
        match lower.as_slice() {
            [first] => assemble_stream::<Msp, i128, 1>(array, [first.as_slice::<u64>()], ctx, sink),
            [first, second] => assemble_stream::<Msp, i256, 2>(
                array,
                [first.as_slice::<u64>(), second.as_slice::<u64>()],
                ctx,
                sink,
            ),
            [first, second, third] => assemble_stream::<Msp, i256, 3>(
                array,
                [
                    first.as_slice::<u64>(),
                    second.as_slice::<u64>(),
                    third.as_slice::<u64>(),
                ],
                ctx,
                sink,
            ),
            _ => vortex_bail!("expected between one and three lower parts"),
        }
    })
}

/// Stream the most significant part of `array`, assembling each chunk with `lower`.
fn assemble_stream<Msp, T, const K: usize>(
    array: ArrayView<'_, DecimalByteParts>,
    lower: [&[u64]; K],
    ctx: &mut ExecutionCtx,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()>
where
    Msp: NativePType + Into<i64>,
    T: Assemble,
{
    let mut adapter = AssembleSink::<Msp, T, K> {
        lower,
        scratch: ScratchChunk::new(),
        inner: sink,
        _msp: PhantomData,
    };
    array.msp().decompress_child_chunks(ctx, &mut adapter)
}

/// A wide decimal integer assembled from a signed most significant part and lower words.
trait Assemble:
    ChunkValue
    + NativeDecimalType
    + From<i64>
    + From<u64>
    + Shl<usize, Output = Self>
    + BitOr<Output = Self>
{
}

impl Assemble for i128 {}
impl Assemble for i256 {}

/// Assembles each chunk of most significant parts with the lower words of its rows.
struct AssembleSink<'a, Msp, T, const K: usize> {
    lower: [&'a [u64]; K],
    scratch: ScratchChunk<T>,
    inner: &'a mut dyn ChunkSink,
    _msp: PhantomData<Msp>,
}

impl<Msp, T, const K: usize> ChunkSink for AssembleSink<'_, Msp, T, K>
where
    Msp: NativePType + Into<i64>,
    T: Assemble,
{
    #[inline]
    fn accept(&mut self, chunk: ChunkMut<'_>, rows: Range<usize>) -> VortexResult<()> {
        let msp = chunk.as_slice::<Msp>();
        let lower = self.lower.map(|part| &part[rows.clone()]);
        emit_with(
            &mut *self.inner,
            T::VALUE_TYPE,
            rows,
            &mut self.scratch,
            |out| {
                for (row, (out, &msp)) in out.iter_mut().zip(msp).enumerate() {
                    *out = assemble_wide_decimal_value(msp.into(), lower.map(|part| part[row]));
                }
                Ok(())
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::SliceArray;
    use vortex_array::chunk_iter::ValueType;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::i256;
    use vortex_array::test_harness::assert_streams_like_execute;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

    use crate::DecimalByteParts;
    use crate::decimal_byte_parts::DecimalBytePartsArraySlotsExt;

    const LEN: usize = 3000;

    fn validity(nullable: bool) -> Validity {
        if nullable {
            Validity::from_iter((0..LEN).map(|i| i % 11 != 4))
        } else {
            Validity::NonNullable
        }
    }

    /// Encode `values` as byte parts, the most significant part re-wrapped in a lazy slice so it
    /// streams through a wrapper too.
    fn encode(decimal: DecimalArray) -> VortexResult<ArrayRef> {
        let mut ctx = array_session().create_execution_ctx();
        let parts = DecimalByteParts::encode(&decimal, &mut ctx)?;
        let msp = parts.msp().clone();
        let msp = SliceArray::new(msp.clone(), 0..msp.len()).into_array();
        Ok(DecimalByteParts::try_new_with_lower_parts(
            msp,
            parts.lower_parts().to_vec(),
            decimal.decimal_dtype(),
        )?
        .into_array())
    }

    #[rstest]
    fn narrow_streams_like_execute(#[values(false, true)] nullable: bool) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let values = (0..LEN as i64)
            .map(|i| (i - 1500) * 7919)
            .collect::<Buffer<i64>>();
        let array = encode(DecimalArray::new(
            values,
            DecimalDType::new(18, 2),
            validity(nullable),
        ))?;
        // The most significant part's integers are the values, in whatever type stores them.
        assert!(matches!(
            array.decompress_chunks_type(),
            Some(ValueType::Primitive(_))
        ));
        assert_streams_like_execute(&array, &mut ctx)?;
        assert_streams_like_execute(&array.slice(517..2900)?, &mut ctx)
    }

    #[rstest]
    fn wide_streams_like_execute(#[values(false, true)] nullable: bool) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let i128s = (0..LEN as i128)
            .map(|i| (i - 1500) * 10i128.pow(30) + i * 7919)
            .collect::<Buffer<i128>>();
        let array = encode(DecimalArray::new(
            i128s,
            DecimalDType::new(38, 2),
            validity(nullable),
        ))?;
        assert_eq!(array.decompress_chunks_type(), Some(ValueType::I128));
        assert_streams_like_execute(&array, &mut ctx)?;
        assert_streams_like_execute(&array.slice(517..2900)?, &mut ctx)?;

        let i256s = (0..LEN as i64)
            .map(|i| i256::from_parts(u128::MAX - i as u128 * 7919, (i - 1500) as i128))
            .collect::<Buffer<i256>>();
        let array = encode(DecimalArray::new(
            i256s,
            DecimalDType::new(76, 2),
            validity(nullable),
        ))?;
        assert_eq!(array.decompress_chunks_type(), Some(ValueType::I256));
        assert_streams_like_execute(&array, &mut ctx)
    }

    /// Chunks stored in different types stream converted to the chunked array's type.
    #[test]
    fn chunked_mixed_types_stream_like_execute() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dtype = DecimalDType::new(18, 2);
        let narrow = encode(DecimalArray::new(
            (0..1500i16).map(|i| i - 700).collect::<Buffer<i16>>(),
            dtype,
            Validity::NonNullable,
        ))?;
        let wide = encode(DecimalArray::new(
            (0..1700i64)
                .map(|i| i * 1_000_000_007)
                .collect::<Buffer<i64>>(),
            dtype,
            Validity::NonNullable,
        ))?;
        let chunked = ChunkedArray::try_new(
            vec![narrow, wide],
            DType::Decimal(dtype, Nullability::NonNullable),
        )?;
        assert_streams_like_execute(&chunked.into_array(), &mut ctx)
    }
}
