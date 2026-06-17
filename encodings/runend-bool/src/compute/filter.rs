// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::min;

use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::RunEndBool;
use crate::array::RunEndBoolArrayExt;
use crate::compress::value_at_index;

/// Ratio of kept elements to runs below which we decode via per-index lookup instead of a linear
/// run-preserving scan. Mirrors the threshold used by `vortex-runend`.
const FILTER_TAKE_THRESHOLD: f64 = 0.1;

impl FilterKernel for RunEndBool {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let mask_values = mask
            .values()
            .vortex_expect("FilterKernel precondition: mask is Mask::Values");

        let runs_ratio = mask_values.true_count() as f64 / array.ends().len() as f64;

        // Sparse masks: decode the kept elements directly. Each kept index needs a single binary
        // search, which is cheaper than a full linear scan when few elements survive.
        if runs_ratio < FILTER_TAKE_THRESHOLD || mask_values.true_count() < 25 {
            let start = array.start();
            let mut bits = BitBufferMut::with_capacity(mask_values.true_count());
            for idx in mask_values.indices().iter().copied() {
                let run_index = array.find_physical_index(idx)?;
                bits.append(value_at_index(run_index, start));
            }
            let validity = filter_validity(&array.bool_validity(), mask)?;
            return Ok(Some(BoolArray::new(bits.freeze(), validity).into_array()));
        }

        // Dense masks: scan the run ends once, accumulating the kept length of each run. This avoids
        // a per-element binary search and preserves the run-end encoding in the output.
        let ends = array.ends().clone().execute::<PrimitiveArray>(ctx)?;
        let start = array.start();
        let (new_ends, new_start, kept) = match_each_unsigned_integer_ptype!(ends.ptype(), |E| {
            filter_run_end_bool(
                ends.as_slice::<E>(),
                array.offset(),
                array.as_ref().len(),
                start,
                mask_values.bit_buffer(),
            )
        });

        let validity = filter_validity(&array.bool_validity(), mask)?;
        let new_ends = PrimitiveArray::new(Buffer::from(new_ends), Validity::NonNullable)
            .narrow(ctx)
            .vortex_expect("ends must succeed downcasting");

        // SAFETY: filter_run_end_bool produces strictly-increasing ends whose last value is `kept`,
        // the length of the filtered array.
        Ok(Some(
            unsafe {
                RunEndBool::new_unchecked(new_ends.into_array(), new_start, 0, kept, validity)
            }
            .into_array(),
        ))
    }
}

/// Linear run-preserving filter over boolean run ends.
///
/// Scans each run once, counting the elements kept by `mask`, and emits run ends for the filtered
/// array. Because boolean runs strictly alternate, dropping an entire run can leave two kept runs
/// with the same value adjacent; these are merged so the output still alternates and can be encoded
/// by a single `start` flag. Returns `(ends, start, length)` of the filtered array.
fn filter_run_end_bool<E: NativePType + AsPrimitive<usize>>(
    run_ends: &[E],
    offset: usize,
    length: usize,
    start: bool,
    mask: &BitBuffer,
) -> (Vec<u64>, bool, usize) {
    let mut new_ends: Vec<u64> = Vec::new();
    let mut prev = 0usize;
    let mut count = 0usize;
    let mut cur_value = false;
    let mut new_start = false;

    for (run_idx, &end) in run_ends.iter().enumerate() {
        let raw: usize = end.as_();
        let end = min(raw.saturating_sub(offset), length);

        let mut kept = 0usize;
        for i in prev..end {
            // SAFETY: i < end <= length == mask.len()
            kept += usize::from(unsafe { mask.value_unchecked(i) });
        }

        if kept > 0 {
            let value = value_at_index(run_idx, start);
            count += kept;
            if new_ends.is_empty() {
                new_start = value;
                cur_value = value;
                new_ends.push(count as u64);
            } else if value == cur_value {
                // Same value as the previous kept run: merge by extending its end.
                *new_ends.last_mut().vortex_expect("new_ends is non-empty") = count as u64;
            } else {
                cur_value = value;
                new_ends.push(count as u64);
            }
        }

        prev = end;
    }

    (new_ends, new_start, count)
}

fn filter_validity(validity: &Validity, mask: &Mask) -> VortexResult<Validity> {
    Ok(match validity {
        Validity::NonNullable => Validity::NonNullable,
        Validity::AllValid => Validity::AllValid,
        Validity::AllInvalid => Validity::AllInvalid,
        Validity::Array(a) => Validity::Array(a.filter(mask.clone())?),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::RunEndBool;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[test]
    fn filter_runend_bool() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // [T,T,F,F,F,T,T,T,T,T]
        let arr = RunEndBool::try_new(
            buffer![2u32, 5, 10].into_array(),
            true,
            Validity::NonNullable,
            &mut ctx,
        )?;
        let filtered = arr.filter(Mask::from_iter([
            true, false, true, false, true, false, true, false, true, false,
        ]))?;
        // keep indices 0,2,4,6,8 => [T,F,F,T,T]
        let expected = BoolArray::from(BitBuffer::from(vec![true, false, false, true, true]));
        assert_arrays_eq!(filtered, expected);
        Ok(())
    }

    /// 4 runs of 32, dense mask: exercises the linear run-preserving path and asserts the output is
    /// still run-end encoded.
    #[test]
    fn filter_dense_preserves_runend() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // ends = [32, 64, 96, 128], start = true => runs T,F,T,F of length 32.
        let arr = RunEndBool::try_new(
            buffer![32u32, 64, 96, 128].into_array(),
            true,
            Validity::NonNullable,
            &mut ctx,
        )?;
        // Keep every other element (64 kept, ratio 16 => dense path).
        let mask = Mask::from_iter((0..128).map(|i| i % 2 == 0));
        let filtered = arr.into_array().filter(mask)?;

        let executed = filtered.execute_until::<RunEndBool>(&mut ctx)?;
        assert_eq!(
            executed.encoding_id().as_ref(),
            "vortex.runend_bool",
            "dense filter should preserve run-end encoding"
        );

        // Within each run every kept element shares the run value: 16 T, 16 F, 16 T, 16 F.
        let mut expected_bits = Vec::new();
        for run in 0..4 {
            expected_bits.extend(std::iter::repeat_n(run % 2 == 0, 16));
        }
        assert_arrays_eq!(executed, BoolArray::from(BitBuffer::from(expected_bits)));
        Ok(())
    }

    /// Dropping an entire run leaves two same-valued runs adjacent; they must merge.
    #[rstest]
    #[case::merge_true(true)]
    #[case::merge_false(false)]
    fn filter_dense_merges_runs(#[case] start: bool) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // 4 runs of 32. Keep runs 0 and 2 fully, drop runs 1 and 3.
        let arr = RunEndBool::try_new(
            buffer![32u32, 64, 96, 128].into_array(),
            start,
            Validity::NonNullable,
            &mut ctx,
        )?;
        let mask = Mask::from_iter((0..128).map(|i| (0..32).contains(&i) || (64..96).contains(&i)));
        let filtered = arr.into_array().filter(mask)?;

        // Runs 0 and 2 share the same value `start`, so the result is a single run of 64.
        let expected = BoolArray::from(BitBuffer::from(vec![start; 64]));
        assert_arrays_eq!(filtered, expected);
        Ok(())
    }
}
