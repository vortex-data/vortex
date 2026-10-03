// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streaming chunked decompression for FastLanes delta arrays.
//!
//! The deltas stream up from their own encoding (typically FoR over bit-packed), and each
//! 1024-element block is undeltaed against its lane bases and untransposed while L1-resident, so
//! neither the deltas nor the output are materialized.

use fastlanes::Delta as FastLanesDelta;
use fastlanes::FastLanes;
use fastlanes::Transpose;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::chunk_iter::BlockDecodeSink;
use vortex_array::chunk_iter::ChunkSink;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::Delta;
use crate::FL_CHUNK_SIZE;
use crate::delta::array::DeltaArrayExt;
use crate::delta::array::DeltaArraySlotsExt;

pub(crate) fn supports_decompress_chunks(array: ArrayView<'_, Delta>) -> bool {
    array.deltas().supports_decompress_chunks()
}

pub(crate) fn decompress_chunks(
    array: ArrayView<'_, Delta>,
    ctx: &mut ExecutionCtx,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()> {
    let ptype = array.dtype().as_ptype();
    // Signed values decode through their unsigned counterpart: `wrapping_add` on the raw bits
    // inverts the `wrapping_sub` done at compress time regardless of signedness.
    let bases = array
        .bases()
        .clone()
        .execute::<PrimitiveArray>(ctx)?
        .reinterpret_cast(ptype.to_unsigned());
    let deltas = array.deltas();
    vortex_ensure!(
        deltas.len().is_multiple_of(FL_CHUNK_SIZE),
        "Delta deltas must hold whole chunks, got {}",
        deltas.len()
    );
    match_each_unsigned_integer_ptype!(bases.ptype(), |T| {
        const LANES: usize = T::LANES;
        stream_blocks::<T, LANES>(array, bases.as_slice::<T>(), ctx, sink)
    })
}

fn stream_blocks<T, const LANES: usize>(
    array: ArrayView<'_, Delta>,
    bases: &[T],
    ctx: &mut ExecutionCtx,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()>
where
    T: NativePType + FastLanesDelta + Transpose,
{
    let deltas = array.deltas();
    let blocks = deltas.len() / FL_CHUNK_SIZE;
    // Casts between widths may leave more bases than the lanes need, as in `delta_decompress`.
    vortex_ensure!(
        bases.len() >= blocks * LANES,
        "Delta has {} bases for {blocks} chunks of {LANES} lanes",
        bases.len()
    );
    let mut transposed = [T::default(); FL_CHUNK_SIZE];
    let mut adapter = BlockDecodeSink::new(
        array.offset(),
        array.len(),
        array.dtype().as_ptype(),
        |block, deltas: &[T; FL_CHUNK_SIZE], out: &mut [T; FL_CHUNK_SIZE]| {
            let bases = <&[T; LANES]>::try_from(&bases[block * LANES..][..LANES])
                .unwrap_or_else(|_| unreachable!("the slice holds one base per lane"));
            T::undelta::<LANES>(deltas, bases, &mut transposed);
            T::untranspose(&transposed, out);
            Ok(())
        },
        sink,
    );
    deltas.decompress_chunks(ctx, &mut adapter)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::test_harness::assert_streams_like_execute;
    use vortex_error::VortexResult;

    use crate::BitPackedData;
    use crate::Delta;
    use crate::FoR;
    use crate::FoRArrayExt;
    use crate::FoRArraySlotsExt;
    use crate::delta::array::DeltaArraySlotsExt;
    use crate::test::SESSION;

    #[derive(Clone, Copy, Debug)]
    enum Deltas {
        Primitive,
        ForBitPacked,
        /// Chunks that straddle the delta blocks, so they are regrouped into whole blocks.
        Unaligned,
    }

    /// Delta-encode `values`, re-encoding the deltas as `deltas`.
    fn delta_array(values: PrimitiveArray, deltas: Deltas) -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        let delta = Delta::try_from_primitive_array(&values, &mut ctx)?;
        let encoded = delta.deltas().clone();
        let encoded = match deltas {
            Deltas::Primitive => encoded,
            Deltas::ForBitPacked => {
                let for_ = FoR::encode(encoded.execute::<PrimitiveArray>(&mut ctx)?, &mut ctx)?;
                let bp = BitPackedData::encode(for_.encoded(), 9, &mut ctx)?;
                FoR::try_new(bp.into_array(), for_.constant_reference().unwrap())?.into_array()
            }
            Deltas::Unaligned => {
                let pieces = [0, 700, 1500, 2600, 4000, encoded.len()]
                    .windows(2)
                    .map(|w| encoded.slice(w[0]..w[1]))
                    .collect::<VortexResult<Vec<_>>>()?;
                ChunkedArray::try_new(pieces, encoded.dtype().clone())?.into_array()
            }
        };
        Ok(Delta::try_new(delta.bases().clone(), encoded, 0, values.len())?.into_array())
    }

    #[rstest]
    fn delta_streams_like_execute(
        #[values(Deltas::Primitive, Deltas::ForBitPacked, Deltas::Unaligned)] deltas: Deltas,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let unsigned = PrimitiveArray::from_option_iter(
            (0..5000u32).map(|i| (i % 37 != 0).then_some(1_000_000 + i * 3 + i % 7)),
        );
        let signed = PrimitiveArray::from_iter((0..5000i64).map(|i| (i % 300) * 5 - 700));
        for array in [delta_array(unsigned, deltas)?, delta_array(signed, deltas)?] {
            assert_streams_like_execute(&array, &mut ctx)?;
            assert_streams_like_execute(&array.slice(517..4013)?, &mut ctx)?;
        }
        Ok(())
    }
}
