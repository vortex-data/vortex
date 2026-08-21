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
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::Sequence;
use crate::eval;

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

        let output_ptype = seq.dtype().as_ptype();

        // A sequence runs monotonically from its first to its last value, both of which fit the
        // output ptype that `base` is held in.
        let last = seq.index_value(seq.len() - 1);
        let (ascending, _) = eval::step_parts(seq.multiplier())
            .ok_or_else(|| vortex_err!("step {} must be an integer", seq.multiplier()))?;

        let (min_pvalue, max_pvalue) = if ascending {
            (seq.base(), last)
        } else {
            (last, seq.base())
        };

        let non_nullable_dtype = DType::Primitive(output_ptype, Nullability::NonNullable);
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

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::NumericalAggregateOpts;
    use vortex_array::aggregate_fn::fns::min_max::MinMaxResult;
    use vortex_array::aggregate_fn::fns::min_max::min_max;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use crate::Sequence;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    fn min_max_uses_output_dtype() -> VortexResult<()> {
        let array = Sequence::try_new_typed(100i32, -10i32, Nullability::NonNullable, 5)?
            .into_array()
            .cast(DType::Primitive(PType::U8, Nullability::NonNullable))?;

        let MinMaxResult { min, max } = min_max(
            &array,
            &mut SESSION.create_execution_ctx(),
            NumericalAggregateOpts::default(),
        )?
        .ok_or_else(|| vortex_err!("min_max of a non-empty sequence should not be null"))?;

        assert_eq!(min, Scalar::from(60u8));
        assert_eq!(max, Scalar::from(100u8));

        Ok(())
    }
}
