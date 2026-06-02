// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_error::VortexResult;

use crate::Delta;
use crate::delta::compute::sorted::bool_range;
use crate::delta::compute::sorted::is_known_sorted;
use crate::delta::compute::sorted::lower_bound;
use crate::delta::compute::sorted::upper_bound;

impl CompareKernel for Delta {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only a known-sorted, non-nullable array reduces to a contiguous range.
        if !is_known_sorted(lhs) || lhs.dtype().is_nullable() {
            return Ok(None);
        }

        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };
        if constant.is_null() {
            return Ok(None);
        }

        let nullability = lhs.dtype().nullability() | rhs.dtype().nullability();
        let arr = lhs.array().clone();
        let len = arr.len();

        match_each_integer_ptype!(lhs.dtype().as_ptype(), |T| {
            let Some(c) = constant.as_primitive().pvalue() else {
                return Ok(None);
            };
            let Ok(c) = c.cast::<T>() else {
                // The constant does not fit in the array's domain; defer to the generic path.
                return Ok(None);
            };

            // `[lower_bound(c), upper_bound(c))` is the run of values equal to `c` in the
            // non-decreasing array; every operator is one or two-sided slice of it.
            let (start, end, invert) = match operator {
                CompareOperator::Eq => (
                    lower_bound::<T>(&arr, len, c, ctx)?,
                    upper_bound::<T>(&arr, len, c, ctx)?,
                    false,
                ),
                CompareOperator::NotEq => (
                    lower_bound::<T>(&arr, len, c, ctx)?,
                    upper_bound::<T>(&arr, len, c, ctx)?,
                    true,
                ),
                CompareOperator::Lt => (0, lower_bound::<T>(&arr, len, c, ctx)?, false),
                CompareOperator::Lte => (0, upper_bound::<T>(&arr, len, c, ctx)?, false),
                CompareOperator::Gt => (upper_bound::<T>(&arr, len, c, ctx)?, len, false),
                CompareOperator::Gte => (lower_bound::<T>(&arr, len, c, ctx)?, len, false),
            };

            Ok(Some(bool_range(len, start, end, invert, nullability)))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;
    use std::time::Instant;

    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Delta;
    use crate::delta::compute::sorted::bool_range;
    use crate::delta::compute::sorted::lower_bound;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    fn sorted_delta(n: usize) -> ArrayRef {
        let prim = PrimitiveArray::from_iter(0..n as u64);
        Delta::try_from_primitive_array(&prim, &mut SESSION.create_execution_ctx())
            .unwrap()
            .into_array()
    }

    /// The compare-kernel core for `Lt c` on a non-decreasing array: binary search for the
    /// crossover, then a `[0, end)` true-run. Stat gating is intentionally bypassed so we
    /// measure only the kernel work.
    fn pushdown_lt(arr: &ArrayRef, c: u64) -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        let end = lower_bound::<u64>(arr, arr.len(), c, &mut ctx)?;
        Ok(bool_range(arr.len(), 0, end, false, Nullability::NonNullable))
    }

    /// Today's path: full decode to primitive, then a generic compare.
    fn decode_lt(arr: &ArrayRef, c: u64) -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        let decoded = arr.clone().execute::<PrimitiveArray>(&mut ctx)?.into_array();
        decoded.binary(ConstantArray::new(c, arr.len()).into_array(), Operator::Lt)
    }

    #[test]
    fn test_pushdown_lt_matches_decode() -> VortexResult<()> {
        let n = 5_000usize;
        let arr = sorted_delta(n);
        assert_arrays_eq!(pushdown_lt(&arr, 2_500)?, decode_lt(&arr, 2_500)?);
        Ok(())
    }

    // Run with: cargo test -p vortex-fastlanes --lib delta_pushdown_bench -- --ignored --nocapture
    #[test]
    #[ignore = "timing benchmark; run explicitly with --nocapture"]
    fn delta_pushdown_bench() -> VortexResult<()> {
        const ITERS: u32 = 50;
        println!(
            "\n{:>12}  {:>16}  {:>16}  {:>8}",
            "n", "pushdown", "decode", "speedup"
        );
        for &n in &[1_024usize, 100_000, 2_000_000] {
            let arr = sorted_delta(n);
            let c = (n / 2) as u64;
            assert_arrays_eq!(pushdown_lt(&arr, c)?, decode_lt(&arr, c)?);

            let t0 = Instant::now();
            for _ in 0..ITERS {
                std::hint::black_box(pushdown_lt(&arr, c)?);
            }
            let push = t0.elapsed() / ITERS;

            let t1 = Instant::now();
            for _ in 0..ITERS {
                std::hint::black_box(decode_lt(&arr, c)?);
            }
            let dec = t1.elapsed() / ITERS;

            println!(
                "{n:>12}  {push:>16?}  {dec:>16?}  {:>7.2}x",
                dec.as_secs_f64() / push.as_secs_f64()
            );
        }
        Ok(())
    }
}
