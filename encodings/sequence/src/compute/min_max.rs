// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::aggregate_fn::fns::min_max::make_minmax_dtype;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_pvalue;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexResult;

use crate::Sequence;

/// Sequence-specific min/max kernel.
///
/// A sequence array represents `A[i] = base + i * multiplier`, so min/max can be computed
/// algebraically from `base` and `last` based on the sign of the multiplier.
#[derive(Debug)]
pub(crate) struct SequenceMinMaxKernel;

impl DynAggregateKernel for SequenceMinMaxKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<MinMax>() {
            return Ok(None);
        }

        let Some(seq) = batch.as_opt::<Sequence>() else {
            return Ok(None);
        };

        let struct_dtype = make_minmax_dtype(batch.dtype());

        // Empty sequences shouldn't exist (try_new validates length), but handle gracefully.
        if seq.is_empty() {
            return Ok(Some(Scalar::null(struct_dtype)));
        }

        let base = seq.base();
        let last = seq.last();

        // Determine min and max based on multiplier direction.
        // For unsigned types, multiplier is always >= 0.
        let (min_pvalue, max_pvalue) = match_each_pvalue!(
            seq.multiplier(),
            uint: |_v| { (base, last) },
            int: |v| {
                if v >= 0 {
                    (base, last)
                } else {
                    (last, base)
                }
            },
            float: |_v| { unreachable!("float multiplier not supported for SequenceArray") }
        );

        let non_nullable_dtype = DType::Primitive(seq.ptype(), Nullability::NonNullable);
        let min_scalar = Scalar::try_new(
            non_nullable_dtype.clone(),
            Some(ScalarValue::Primitive(min_pvalue)),
        )?;
        let max_scalar =
            Scalar::try_new(non_nullable_dtype, Some(ScalarValue::Primitive(max_pvalue)))?;

        Ok(Some(Scalar::struct_(
            struct_dtype,
            vec![min_scalar, max_scalar],
        )))
    }
}
