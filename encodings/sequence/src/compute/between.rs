// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::between::BetweenKernel;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::Sequence;

impl BetweenKernel for Sequence {
    fn between(
        array: ArrayView<'_, Self>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let (Some(lower), Some(upper)) = (lower.as_constant(), upper.as_constant()) else {
            return Ok(None);
        };
        let nullability =
            array.dtype().nullability() | lower.dtype().nullability() | upper.dtype().nullability();
        let lower = integer(
            lower
                .as_primitive()
                .pvalue()
                .vortex_expect("non-null bound"),
        )? + i128::from(options.lower_strict.is_strict());
        let upper = integer(
            upper
                .as_primitive()
                .pvalue()
                .vortex_expect("non-null bound"),
        )? - i128::from(options.upper_strict.is_strict());
        let base = integer(array.base())?;
        let step = integer(array.multiplier())?;
        let len = array.len();
        if lower > upper || step == 0 {
            let value = lower <= base && base <= upper;
            return Ok(Some(
                ConstantArray::new(Scalar::bool(value, nullability), len).into_array(),
            ));
        }
        // Construction guarantees a monotone sequence whose values fit the output ptype.
        // i128 preserves unsigned values above i64::MAX and signed descending steps.
        let (start, end) = if step > 0 {
            (
                ceil_div(lower - base, step),
                (upper - base).div_euclid(step) + 1,
            )
        } else {
            (
                ceil_div(base - upper, -step),
                (base - lower).div_euclid(-step) + 1,
            )
        };
        let length = i128::try_from(len)?;
        let start = usize::try_from(start.clamp(0, length))?;
        let end = usize::try_from(end.clamp(0, length))?;
        let mut bits = BitBufferMut::new_unset(len);
        if start < end {
            bits.fill_range(start, end, true);
        }
        let validity = match nullability {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => Validity::AllValid,
        };
        Ok(Some(BoolArray::new(bits.freeze(), validity).into_array()))
    }
}

fn integer(value: PValue) -> VortexResult<i128> {
    if value.ptype().is_signed_int() {
        value.cast::<i64>().map(i128::from)
    } else {
        value.cast::<u64>().map(i128::from)
    }
}

fn ceil_div(value: i128, divisor: i128) -> i128 {
    value.div_euclid(divisor) + i128::from(value.rem_euclid(divisor) != 0)
}

#[cfg(test)]
mod tests;
