// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::Delta;
use crate::delta::array::DeltaArraySlotsExt;

/// Delta-specific is_constant kernel.
///
/// An array with nulls is constant only if every value is null, which validity alone decides.
/// Otherwise every value equals the base of its lane plus the earlier deltas of the lane, so zero
/// deltas and equal bases make the array constant, without decoding it.
///
/// The converse does not hold for a slice: its first and last chunks can hold values outside the
/// slice, whose deltas and bases need not match. So when the children are not constant, the kernel
/// leaves the answer to the generic path, which decodes the array.
#[derive(Debug)]
pub(crate) struct DeltaIsConstantKernel;

impl DynAggregateKernel for DeltaIsConstantKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<IsConstant>() || batch.is_empty() {
            return Ok(None);
        }
        let Some(array) = batch.as_opt::<Delta>() else {
            return Ok(None);
        };
        let result = if batch.all_invalid(ctx)? {
            true
        } else if !batch.all_valid(ctx)? {
            false
        } else if deltas_are_zero(array.deltas(), ctx)? && is_constant(array.bases(), ctx)? {
            true
        } else {
            return Ok(None);
        };
        Ok(Some(IsConstant::make_partial(batch, result, ctx)?))
    }
}

/// Whether every delta is zero.
fn deltas_are_zero(deltas: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    Ok(is_constant(deltas, ctx)?
        && deltas.execute_scalar(0, ctx)?.as_primitive().as_::<u64>() == Some(0))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::EmptyOptions;
    use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
    use vortex_array::aggregate_fn::fns::is_constant::is_constant;
    use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_error::VortexResult;

    use super::DeltaIsConstantKernel;
    use crate::BitPacked;
    use crate::Delta;
    use crate::DeltaArraySlotsExt;
    use crate::test::SESSION;

    /// The kernel's answer for `array`, or `None` when it leaves the answer to the generic path.
    fn kernel(array: &ArrayRef) -> VortexResult<Option<bool>> {
        let mut ctx = SESSION.create_execution_ctx();
        let aggregate = IsConstant.bind(EmptyOptions);
        let Some(partial) = DeltaIsConstantKernel.aggregate(&aggregate, array, &mut ctx)? else {
            return Ok(None);
        };
        let mut accumulator = aggregate.accumulator(array.dtype())?;
        accumulator.combine_partials(&partial)?;
        Ok(accumulator.finish()?.as_bool().value())
    }

    fn delta(source: &PrimitiveArray) -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        Ok(Delta::try_from_primitive_array(source, &mut ctx)?.into_array())
    }

    #[rstest]
    #[case::constant_u32(PrimitiveArray::from_iter([7u32; 3000]), Some(true))]
    #[case::constant_i64(PrimitiveArray::from_iter([-5i64; 1024]), Some(true))]
    #[case::constant_nullable(PrimitiveArray::from_option_iter([Some(9u16); 2000]), Some(true))]
    #[case::all_null(PrimitiveArray::from_option_iter([None::<u32>; 2000]), Some(true))]
    #[case::some_null(
        PrimitiveArray::from_option_iter((0..2000).map(|i| (i != 1500).then_some(4u32))),
        Some(false)
    )]
    // Zero deltas, but each chunk has its own base.
    #[case::constant_per_chunk(
        PrimitiveArray::from_iter((0..3072u32).map(|i| i / 1024)),
        None
    )]
    #[case::increasing(PrimitiveArray::from_iter(0u32..3000), None)]
    fn kernel_answers(
        #[case] source: PrimitiveArray,
        #[case] expected: Option<bool>,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = delta(&source)?;
        assert_eq!(kernel(&array)?, expected);
        assert_eq!(
            is_constant(&array, &mut ctx)?,
            is_constant(&source.into_array(), &mut ctx)?
        );
        Ok(())
    }

    #[test]
    fn bitpacked_zero_deltas() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let source = PrimitiveArray::from_iter([3u64; 5000]);
        let delta = Delta::try_from_primitive_array(&source, &mut ctx)?;
        let packed = BitPacked::encode(delta.deltas(), 1, &mut ctx)?;
        let array = Delta::try_new(
            delta.bases().clone(),
            packed.into_array(),
            delta.validity()?,
            0,
            source.len(),
        )?
        .into_array();
        assert_eq!(kernel(&array)?, Some(true));
        Ok(())
    }

    /// A slice that is constant while values before it in its first chunk are not: the
    /// children are not constant, so the kernel must not answer false.
    #[test]
    fn constant_slice_of_non_constant_chunk() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let source = PrimitiveArray::from_iter((0..2048u32).map(|i| if i < 10 { i } else { 7 }));
        let array = delta(&source)?.slice(10..2048)?.optimize_ctx(&ctx)?;
        assert!(array.is::<Delta>());
        assert_eq!(kernel(&array)?, None);
        assert!(is_constant(&array, &mut ctx)?);
        Ok(())
    }
}
