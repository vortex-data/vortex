// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::MinMaxPartial;
use super::MinMaxResult;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::chunked::FloatKey;
use crate::aggregate_fn::chunked::Keyed;
use crate::aggregate_fn::chunked::MinMax;
use crate::aggregate_fn::chunked::accumulate;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::dtype::Nullability::NonNullable;
use crate::match_each_float_ptype;
use crate::match_each_integer_ptype;
use crate::scalar::PValue;
use crate::scalar::Scalar;

pub(super) fn accumulate_primitive(
    args: AggregateArgs<'_, NumericalAggregateOpts>,
    partial: &mut MinMaxPartial,
    p: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let validity = p.as_ref().validity()?.execute_mask(p.as_ref().len(), ctx)?;
    let local = if p.ptype().is_int() {
        match_each_integer_ptype!(p.ptype(), |T| {
            let mut acc = MinMax::<T>::new();
            accumulate(p.as_slice::<T>(), &validity, &mut acc);
            acc.finish().map(min_max_result)
        })
    } else {
        match_each_float_ptype!(p.ptype(), |F| {
            // Floats are ordered totally. NaNs are skipped, or are extremes that the partial's
            // merge turns into NaN.
            let values = p.as_slice::<F>();
            let keys = if args.options.skip_nans {
                let mut acc = Keyed::<F, _, true>::new(MinMax::<<F as FloatKey>::Key>::new());
                accumulate(values, &validity, &mut acc);
                acc.inner().finish()
            } else {
                let mut acc = Keyed::<F, _, false>::new(MinMax::<<F as FloatKey>::Key>::new());
                accumulate(values, &validity, &mut acc);
                acc.inner().finish()
            };
            keys.map(|(min, max)| min_max_result((F::from_key(min), F::from_key(max))))
        })
    };
    partial.merge(args, local);
    Ok(())
}

fn min_max_result<T>((min, max): (T, T)) -> MinMaxResult
where
    T: NativePType,
    PValue: From<T>,
{
    MinMaxResult {
        min: Scalar::primitive(min, NonNullable),
        max: Scalar::primitive(max, NonNullable),
    }
}
