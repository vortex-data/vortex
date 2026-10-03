// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streaming chunked decompression for run-end encoded primitive arrays: runs are expanded one
//! chunk at a time, so the full-length decoded buffer is never written.

use num_traits::AsPrimitive;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::chunk_iter::ChunkSink;
use vortex_array::chunk_iter::stream_from_fn;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_native_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::RunEnd;
use crate::RunEndArrayExt;
use crate::RunEndArraySlotsExt;

pub(crate) fn supports_decompress_chunks(_array: ArrayView<'_, RunEnd>) -> bool {
    true
}

pub(crate) fn decompress_chunks(
    array: ArrayView<'_, RunEnd>,
    ctx: &mut ExecutionCtx,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()> {
    let ends = array.ends().clone().execute::<PrimitiveArray>(ctx)?;
    let values = array.values().clone().execute::<PrimitiveArray>(ctx)?;
    vortex_ensure!(
        ends.len() == values.len(),
        "RunEnd has {} ends but {} values",
        ends.len(),
        values.len()
    );
    let (offset, len) = (array.offset(), array.len());
    match_each_unsigned_integer_ptype!(ends.ptype(), |E| {
        match_each_native_ptype!(values.ptype(), |T| {
            stream_runs(
                ends.as_slice::<E>(),
                offset,
                values.as_slice::<T>(),
                len,
                sink,
            )
        })
    })
}

/// Expand the runs ending at `ends` with `values` into `len` rows starting at row `offset` of
/// the runs.
///
/// Each run's end is resolved as the stream reaches it, so the ends are read once, in their
/// stored type, with no intermediate buffer of them.
fn stream_runs<T: NativePType, E: NativePType + AsPrimitive<usize>>(
    ends: &[E],
    offset: usize,
    values: &[T],
    len: usize,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()> {
    let mut run = 0;
    stream_from_fn(len, sink, |chunk: &mut [T], rows| {
        let mut row = rows.start;
        while row < rows.end {
            let end: usize = ends
                .get(run)
                .ok_or_else(|| vortex_err!("RunEnd runs end before row {row}"))?
                .as_();
            let end = end
                .checked_sub(offset)
                .ok_or_else(|| vortex_err!("run end {end} must be >= offset {offset}"))?;
            let fill_end = end.min(rows.end);
            if fill_end > row {
                chunk[row - rows.start..fill_end - rows.start].fill(values[run]);
                row = fill_end;
            }
            if end <= row {
                run += 1;
            }
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::SliceArray;
    use vortex_array::test_harness::assert_streams_like_execute;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::RunEnd;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    /// Runs of 1 to 2000 rows, so some cross chunk boundaries and some span whole chunks.
    #[test]
    fn runend_streams_like_execute() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_option_iter((0..5000u32).map(|i| {
            let run = [0, 1, 2, 700, 3000][(i as usize * 7 / 5000) % 5] + i / 997;
            (run % 4 != 3).then_some(run)
        }));
        let array = RunEnd::encode(values.into_array(), &mut ctx)?.into_array();
        assert_streams_like_execute(&array, &mut ctx)?;
        // Slicing leaves an offset into the first run.
        assert_streams_like_execute(&array.slice(517..4013)?, &mut ctx)
    }

    /// The executor never streams past a node it is executing toward: exporters execute until a
    /// run-end array to hand it over as is, even behind lazy slices.
    #[test]
    fn executor_stops_at_run_end_behind_slices() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter([7i64, 7, 8, 8]).into_array();
        let runend = RunEnd::encode(values, &mut ctx)?.into_array();
        let wrapped =
            SliceArray::new(SliceArray::new(runend, 0..4).into_array(), 0..4).into_array();
        // Executing to canonical streams this chain of three.
        assert!(wrapped.should_execute_via_chunks());

        assert!(wrapped.execute_until::<RunEnd>(&mut ctx)?.is::<RunEnd>());
        Ok(())
    }
}
