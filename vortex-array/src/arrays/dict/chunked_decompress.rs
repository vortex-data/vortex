// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streaming chunked decompression for dictionaries of primitive values.
//!
//! The codes stream up from their own encoding (typically bit-packed), and each chunk of codes is
//! gathered from the dictionary while L1-resident, straight into the output when materializing,
//! so the codes are never materialized.

use std::marker::PhantomData;
use std::ops::Range;

use num_traits::AsPrimitive;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::Dict;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::DictArraySlotsExt;
use crate::chunk_iter::ChunkMut;
use crate::chunk_iter::ChunkSink;
use crate::chunk_iter::ScratchChunk;
use crate::chunk_iter::emit_with;
use crate::dtype::NativePType;
use crate::match_each_integer_ptype;
use crate::match_each_native_ptype;

pub(super) fn supports_decompress_chunks(array: ArrayView<'_, Dict>) -> bool {
    array.codes().supports_decompress_chunks()
}

pub(super) fn decompress_chunks(
    array: ArrayView<'_, Dict>,
    ctx: &mut ExecutionCtx,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()> {
    let codes = array.codes();
    let values = array.values().clone().execute::<PrimitiveArray>(ctx)?;
    // Null codes may hold any value, so a code that is out of bounds is only an error where the
    // code is valid.
    let codes_validity = codes.validity()?.execute_mask(codes.len(), ctx)?;
    match_each_native_ptype!(values.ptype(), |V| {
        match_each_integer_ptype!(codes.dtype().as_ptype(), |C| {
            let mut adapter = GatherSink::<C, V> {
                values: values.as_slice::<V>(),
                codes_validity: &codes_validity,
                scratch: ScratchChunk::new(),
                inner: sink,
                _codes: PhantomData,
            };
            codes.decompress_chunks(ctx, &mut adapter)
        })
    })
}

/// Gathers each chunk of codes from the dictionary into the inner sink's destination, or a scratch
/// chunk when it offers none, and forwards it.
struct GatherSink<'a, C, V> {
    values: &'a [V],
    codes_validity: &'a Mask,
    scratch: ScratchChunk<V>,
    inner: &'a mut dyn ChunkSink,
    _codes: PhantomData<C>,
}

impl<C, V> ChunkSink for GatherSink<'_, C, V>
where
    C: NativePType + Ord + AsPrimitive<usize>,
    V: NativePType,
{
    #[inline]
    fn accept(&mut self, chunk: ChunkMut<'_>, rows: Range<usize>) -> VortexResult<()> {
        let codes = chunk.as_slice::<C>();
        let (values, codes_validity) = (self.values, self.codes_validity);
        emit_with(
            &mut *self.inner,
            V::PTYPE,
            rows.clone(),
            &mut self.scratch,
            |out| gather(codes, values, codes_validity, rows, out),
        )
    }
}

/// Gather `values[codes]` into `out`, rejecting an out-of-bounds code only where it is valid.
#[inline]
fn gather<C, V>(
    codes: &[C],
    values: &[V],
    codes_validity: &Mask,
    rows: Range<usize>,
    out: &mut [V],
) -> VortexResult<()>
where
    C: NativePType + Ord + AsPrimitive<usize>,
    V: NativePType,
{
    let max_code = codes.iter().copied().max().map_or(0, |code| code.as_());
    if max_code < values.len() {
        for (out, code) in out.iter_mut().zip(codes) {
            // SAFETY: every code in the chunk is at most `max_code`, which is in bounds.
            *out = unsafe { *values.get_unchecked(code.as_()) };
        }
        return Ok(());
    }
    for (row, (out, code)) in rows.zip(out.iter_mut().zip(codes)) {
        let code: usize = code.as_();
        *out = match values.get(code) {
            Some(&value) => value,
            None => {
                vortex_ensure!(
                    !codes_validity.value(row),
                    "Dict code {code} is out of bounds for {} values",
                    values.len()
                );
                V::default()
            }
        };
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::ConstantArray;
    use crate::arrays::DictArray;
    use crate::arrays::PrimitiveArray;
    use crate::chunk_iter::ChunkMut;
    use crate::test_harness::assert_streams_like_execute;
    use crate::validity::Validity;

    #[test]
    fn dict_streams_like_execute() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let codes = PrimitiveArray::from_iter((0..5000u32).map(|i| (i * 7 % 5) as u8));
        let values =
            PrimitiveArray::from_option_iter([Some(10i64), None, Some(30), Some(40), Some(50)]);
        let array = DictArray::try_new(codes.into_array(), values.into_array())?.into_array();
        assert_streams_like_execute(&array, &mut ctx)?;
        assert_streams_like_execute(&array.slice(517..4013)?, &mut ctx)?;

        // Codes of any streaming encoding are gathered chunk by chunk.
        let codes = ConstantArray::new(3u16, 2500).into_array();
        let array = DictArray::try_new(codes, buffer![1.5f32, 2.5, 3.5, 4.5].into_array())?;
        assert_streams_like_execute(&array.into_array(), &mut ctx)
    }

    /// Null codes may hold any value, so out-of-bounds codes there must not fail the gather.
    #[test]
    fn dict_null_codes_out_of_bounds() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let codes = PrimitiveArray::new(
            buffer![0u8, 200, 1, 255, 1],
            Validity::from_iter([true, false, true, false, true]),
        );
        // SAFETY: the out-of-bounds codes are null.
        let array = unsafe {
            DictArray::new_unchecked(codes.into_array(), buffer![10i32, 20].into_array())
        }
        .into_array();
        let mut streamed = Vec::new();
        array.decompress_chunks(&mut ctx, &mut |chunk: ChunkMut<'_>, _rows: Range<usize>| {
            streamed.extend_from_slice(chunk.as_slice::<i32>());
            Ok(())
        })?;
        assert_eq!(streamed.len(), 5);
        assert_eq!([streamed[0], streamed[2], streamed[4]], [10, 20, 20]);
        Ok(())
    }
}
