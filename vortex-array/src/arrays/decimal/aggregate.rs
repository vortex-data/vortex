// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Forward decimal aggregates that preserve unscaled integer ordering.
//!
//! Child aggregates keep their stored width. Boundary values in partial states are converted back
//! to decimal scalars without rescaling; decimal sum and arithmetic keep their own kernels.

use std::sync::LazyLock;

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::Decimal;
use crate::ArrayRef;
use crate::ArrayVTable;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::fns::all_non_null::AllNonNull;
use crate::aggregate_fn::fns::all_null::AllNull;
use crate::aggregate_fn::fns::count::Count;
use crate::aggregate_fn::fns::first::First;
use crate::aggregate_fn::fns::is_constant::IsConstant;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::last::Last;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::null_count::NullCount;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::aggregate_fn::session::AggregateFnSession;
use crate::arrays::decimal::DecimalArraySlotsExt;
use crate::dtype::DType;
use crate::integer;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

pub(crate) fn register_aggregate_kernels(session: &AggregateFnSession) {
    // One fallback avoids cloning the registry for each supported aggregate.
    session.register_aggregate_kernel(Decimal.id(), None::<AggregateFnId>, &DecimalAggregateKernel);
}

static SUPPORTED_AGGREGATES: LazyLock<[AggregateFnId; 11]> = LazyLock::new(|| {
    [
        MinMax.id(),
        Min.id(),
        Max.id(),
        IsConstant.id(),
        IsSorted.id(),
        First.id(),
        Last.id(),
        Count.id(),
        NullCount.id(),
        AllNull.id(),
        AllNonNull.id(),
    ]
});

#[derive(Debug)]
struct DecimalAggregateKernel;

impl DynAggregateKernel for DecimalAggregateKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !SUPPORTED_AGGREGATES.contains(&aggregate_fn.id()) {
            return Ok(None);
        }

        let Some(array) = batch.as_opt::<Decimal>() else {
            return Ok(None);
        };
        let Some(dtype) = aggregate_fn.state_dtype(batch.dtype()) else {
            return Ok(None);
        };
        let mut accumulator = aggregate_fn.accumulator(array.values().dtype())?;
        accumulator.accumulate(array.values(), ctx)?;
        let partial = accumulator.partial_scalar()?;
        Ok(Some(decimal_partial(&partial, &dtype)?))
    }
}

/// Replace integer boundary values with decimal scalars without applying the decimal scale.
fn decimal_partial(scalar: &Scalar, dtype: &DType) -> VortexResult<Scalar> {
    if scalar.is_null() {
        return Ok(Scalar::null(dtype.clone()));
    }
    if matches!(dtype, DType::Decimal(..)) {
        return Scalar::try_new(
            dtype.clone(),
            Some(ScalarValue::Decimal(integer::scalar_value(scalar)?)),
        );
    }
    if let DType::Struct(fields, _) = dtype {
        let values = scalar
            .as_struct()
            .fields_iter()
            .vortex_expect("Non-null aggregate state")
            .zip_eq(fields.fields())
            .map(|(value, dtype)| decimal_partial(&value, &dtype))
            .collect::<VortexResult<Vec<_>>>()?;
        return Ok(Scalar::struct_(dtype.clone(), values));
    }
    scalar.cast(dtype)
}
