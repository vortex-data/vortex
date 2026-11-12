// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::Delta;
use crate::FL_CHUNK_SIZE;
use crate::delta::array::DeltaArrayExt;
use crate::delta::array::DeltaArraySlotsExt;
use crate::delta::array::lane_count;

/// Delta-specific is_constant kernel.
///
/// An array with nulls is constant only if every value is null, which validity alone decides.
/// Otherwise every value is the base of its lane plus the earlier deltas of the lane. In a chunk
/// that lies wholly within the array, the values are all equal exactly when its deltas are zero
/// and its bases are equal, so those chunks are checked on the children without decoding. Only
/// the first and last chunks of a slice can hold values outside it; their visible values are
/// decoded and compared with the rest.
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
        } else {
            all_valid_is_constant(array, ctx)?
        };
        Ok(Some(IsConstant::make_partial(batch, result, ctx)?))
    }
}

/// Whether the values of `array`, which has no nulls, are all equal.
fn all_valid_is_constant(
    array: ArrayView<'_, Delta>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let offset = array.offset();
    let len = array.len();
    // The chunks wholly within the array, counted from the first chunk of the children.
    let full_chunks = offset.div_ceil(FL_CHUNK_SIZE)..(offset + len) / FL_CHUNK_SIZE;
    if full_chunks.is_empty() {
        // At most two partial chunks: decode the array.
        let decoded = array.array().clone().execute::<PrimitiveArray>(ctx)?;
        return Ok(match_each_integer_ptype!(decoded.ptype(), |P| {
            let values = decoded.as_slice::<P>();
            values.iter().all(|&value| value == values[0])
        }));
    }

    let lanes = lane_count(array.dtype().as_ptype());
    let deltas = array
        .deltas()
        .slice(full_chunks.start * FL_CHUNK_SIZE..full_chunks.end * FL_CHUNK_SIZE)?;
    let bases = array
        .bases()
        .slice(full_chunks.start * lanes..full_chunks.end * lanes)?;
    if !deltas_are_zero(&deltas, ctx)? || !is_constant(&bases, ctx)? {
        return Ok(false);
    }
    let constant = bases.execute_scalar(0, ctx)?;

    // The visible values of the partial first and last chunks, in logical rows.
    let head = 0..full_chunks.start * FL_CHUNK_SIZE - offset;
    let tail = full_chunks.end * FL_CHUNK_SIZE - offset..len;
    for rows in [head, tail] {
        if rows.is_empty() {
            continue;
        }
        let decoded = array.array().slice(rows)?.execute::<PrimitiveArray>(ctx)?;
        let equal = match_each_integer_ptype!(decoded.ptype(), |P| {
            let constant = constant
                .as_primitive()
                .typed_value::<P>()
                .vortex_expect("bases are non-nullable");
            decoded
                .as_slice::<P>()
                .iter()
                .all(|&value| value == constant)
        });
        if !equal {
            return Ok(false);
        }
    }
    Ok(true)
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
        Some(false)
    )]
    #[case::increasing(PrimitiveArray::from_iter(0u32..3000), Some(false))]
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

    /// 7 from row 10 to row 4000, and other values around that. A slice decides its partial first
    /// and last chunks by their visible rows alone.
    fn steps() -> PrimitiveArray {
        PrimitiveArray::from_iter((0..5000u32).map(|i| if (10..4000).contains(&i) { 7 } else { i }))
    }

    #[rstest]
    // Hidden values on both sides, in the partial first and last chunks.
    #[case::constant_inside(10..4000, true)]
    #[case::starts_in_hidden(9..4000, false)]
    #[case::ends_in_hidden(10..4001, false)]
    // Within one chunk, and across two partial chunks with no full chunk between them.
    #[case::one_chunk(100..900, true)]
    #[case::two_partial_chunks(1000..1100, true)]
    #[case::two_partial_chunks_differ(3990..4010, false)]
    // Chunk-aligned on one side.
    #[case::aligned_start(1024..3999, true)]
    #[case::aligned_end(10..3072, true)]
    fn sliced_kernel_answers(
        #[case] range: std::ops::Range<usize>,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let source = steps();
        let array = delta(&source)?.slice(range.clone())?.optimize_ctx(&ctx)?;
        assert!(array.is::<Delta>());
        assert_eq!(kernel(&array)?, Some(expected));
        assert_eq!(
            is_constant(&array, &mut ctx)?,
            is_constant(&source.into_array().slice(range)?, &mut ctx)?
        );
        Ok(())
    }
}
