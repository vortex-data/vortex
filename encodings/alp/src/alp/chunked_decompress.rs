// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streaming chunked decompression for ALP arrays: each chunk of encoded integers streamed up from
//! the child is decoded into floats in place and patched while L1-resident.

use std::ops::Range;

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::chunk_iter::ChunkMut;
use vortex_array::chunk_iter::ChunkPatches;
use vortex_array::chunk_iter::ChunkSink;
use vortex_array::dtype::NativePType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ALP;
use crate::ALPArrayExt;
use crate::ALPArraySlotsExt;
use crate::ALPFloat;
use crate::Exponents;
use crate::match_each_alp_float_ptype;

pub(crate) fn supports_decompress_chunks(array: ArrayView<'_, ALP>) -> bool {
    array.encoded().supports_decompress_chunks()
}

pub(crate) fn decompress_chunks(
    array: ArrayView<'_, ALP>,
    ctx: &mut ExecutionCtx,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()> {
    let exponents = array.exponents();
    let patches = array.patches();
    match_each_alp_float_ptype!(array.dtype().as_ptype(), |F| {
        let patches = match &patches {
            Some(patches) => ChunkPatches::try_from_patches(patches, ctx, |_, value: F| value)?,
            None => ChunkPatches::new(Vec::new()),
        };
        let mut adapter = DecodeSink {
            exponents,
            patches,
            inner: sink,
        };
        array.encoded().decompress_child_chunks(ctx, &mut adapter)
    })
}

/// Decodes each chunk of encoded integers into floats in place, then patches it.
struct DecodeSink<'a, F> {
    exponents: Exponents,
    patches: ChunkPatches<F>,
    inner: &'a mut dyn ChunkSink,
}

impl<F> ChunkSink for DecodeSink<'_, F>
where
    F: ALPFloat + NativePType,
    F::ALPInt: NativePType,
{
    #[inline]
    fn accept(&mut self, mut chunk: ChunkMut<'_>, rows: Range<usize>) -> VortexResult<()> {
        F::decode_slice_inplace(chunk.as_slice_mut::<F::ALPInt>(), self.exponents);
        let mut chunk = chunk.retype::<F>();
        self.patches.apply(chunk.as_slice_mut::<F>(), rows.start);
        self.inner.accept(chunk, rows)
    }

    /// The child writes encoded integers into the floats' destination, decoded in place after.
    fn destination(&mut self, rows: Range<usize>) -> Option<ChunkMut<'_>> {
        self.inner
            .destination(rows)
            .map(|chunk| chunk.retype::<F::ALPInt>())
    }

    fn accept_written(&mut self, rows: Range<usize>) -> VortexResult<()> {
        let mut chunk = self
            .inner
            .destination(rows.clone())
            .ok_or_else(|| vortex_err!("ALP's destination for rows {rows:?} is gone"))?
            .retype::<F::ALPInt>();
        F::decode_slice_inplace(chunk.as_slice_mut::<F::ALPInt>(), self.exponents);
        let mut chunk = chunk.retype::<F>();
        self.patches.apply(chunk.as_slice_mut::<F>(), rows.start);
        self.inner.accept_written(rows)
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::test_harness::assert_streams_like_execute;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ALPArrayExt;
    use crate::alp_encode;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    fn alp_streams_like_execute() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Every 301st value cannot be encoded, so it is patched; every 13th is null.
        let values = PrimitiveArray::from_option_iter((0..5000).map(|i| {
            let value = if i % 301 == 0 {
                PI * f64::from(i)
            } else {
                f64::from(i) * 0.25
            };
            (i % 13 != 0).then_some(value)
        }));
        let alp = alp_encode(values.as_view(), None, &mut ctx)?;
        assert!(alp.patches().is_some());
        let array = alp.into_array();
        assert_streams_like_execute(&array, &mut ctx)?;
        assert_streams_like_execute(&array.slice(517..4013)?, &mut ctx)
    }
}
