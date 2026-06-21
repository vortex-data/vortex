// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::Chunked;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::scalar::Scalar;

#[derive(Debug)]
pub struct ChunkedArrayAggregate;

impl DynAggregateKernel for ChunkedArrayAggregate {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let Some(chunked) = batch.as_opt::<Chunked>() else {
            return Ok(None);
        };

        let mut acc = aggregate_fn.accumulator(chunked.dtype())?;
        // Skip empty chunks: they contribute no elements.
        for chunk in chunked.non_empty_chunks() {
            acc.accumulate(chunk, ctx)?;
        }
        // Return the partial (not finalized) result, since the outer accumulator
        // will call combine_partials() on this value.
        Ok(Some(acc.flush()?))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::sum::Sum;
    use crate::arrays::BoolArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::scalar::Scalar;

    fn run_sum(batch: &crate::ArrayRef) -> VortexResult<Scalar> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let mut acc = Accumulator::try_new(
            Sum,
            NumericalAggregateOpts::default(),
            batch.dtype().clone(),
        )?;
        acc.accumulate(batch, &mut ctx)?;
        acc.finish()
    }

    #[test]
    fn sum_chunked_i32() -> VortexResult<()> {
        let chunked = ChunkedArray::try_new(
            vec![
                buffer![1i32, 2, 3].into_array(),
                buffer![4i32, 5, 6].into_array(),
            ],
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )?;
        let result = run_sum(&chunked.into_array())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(21));
        Ok(())
    }

    #[test]
    fn sum_chunked_f64() -> VortexResult<()> {
        let chunked = ChunkedArray::try_new(
            vec![
                buffer![1.5f64, 2.5].into_array(),
                buffer![3.0f64].into_array(),
            ],
            DType::Primitive(PType::F64, Nullability::NonNullable),
        )?;
        let result = run_sum(&chunked.into_array())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(7.0));
        Ok(())
    }

    #[test]
    fn sum_chunked_with_nulls() -> VortexResult<()> {
        let chunked = ChunkedArray::try_new(
            vec![
                PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array(),
                PrimitiveArray::from_option_iter([None, Some(5)]).into_array(),
            ],
            DType::Primitive(PType::I32, Nullability::Nullable),
        )?;
        let result = run_sum(&chunked.into_array())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(9));
        Ok(())
    }

    #[test]
    fn sum_chunked_all_null() -> VortexResult<()> {
        let chunked = ChunkedArray::try_new(
            vec![
                PrimitiveArray::from_option_iter([None::<i32>, None]).into_array(),
                PrimitiveArray::from_option_iter([None::<i32>]).into_array(),
            ],
            DType::Primitive(PType::I32, Nullability::Nullable),
        )?;
        let result = run_sum(&chunked.into_array())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(0));
        Ok(())
    }

    #[test]
    fn sum_chunked_single_chunk() -> VortexResult<()> {
        let chunked = ChunkedArray::try_new(
            vec![buffer![10i32, 20, 30].into_array()],
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )?;
        let result = run_sum(&chunked.into_array())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(60));
        Ok(())
    }

    #[test]
    fn sum_chunked_empty_chunks() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let chunked = ChunkedArray::try_new(
            vec![
                Buffer::<i32>::empty().into_array(),
                buffer![1i32, 2, 3].into_array(),
                Buffer::<i32>::empty().into_array(),
                buffer![4i32, 5].into_array(),
                Buffer::<i32>::empty().into_array(),
            ],
            dtype,
        )?;
        let result = run_sum(&chunked.into_array())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(15));
        Ok(())
    }

    #[test]
    fn sum_chunked_all_empty() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let chunked = ChunkedArray::try_new(vec![], dtype)?;
        let result = run_sum(&chunked.into_array())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(0));
        Ok(())
    }

    #[test]
    fn sum_chunked_many_small_chunks() -> VortexResult<()> {
        let chunked = ChunkedArray::try_new(
            vec![
                buffer![1i32].into_array(),
                buffer![2i32].into_array(),
                buffer![3i32].into_array(),
                buffer![4i32].into_array(),
                buffer![5i32].into_array(),
            ],
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )?;
        let result = run_sum(&chunked.into_array())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(15));
        Ok(())
    }

    #[test]
    fn sum_chunked_u64() -> VortexResult<()> {
        let chunked = ChunkedArray::try_new(
            vec![
                buffer![100u64, 200].into_array(),
                buffer![300u64].into_array(),
            ],
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )?;
        let result = run_sum(&chunked.into_array())?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(600));
        Ok(())
    }

    #[test]
    fn sum_chunked_bool() -> VortexResult<()> {
        let b1: BoolArray = [true, false, true].into_iter().collect();
        let b2: BoolArray = [true, true].into_iter().collect();
        let chunked = ChunkedArray::try_new(
            vec![b1.into_array(), b2.into_array()],
            DType::Bool(Nullability::NonNullable),
        )?;
        let result = run_sum(&chunked.into_array())?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(4));
        Ok(())
    }

    #[test]
    fn sum_chunked_bool_with_nulls() -> VortexResult<()> {
        let b1 = BoolArray::from_iter([Some(true), None, Some(true)]);
        let b2 = BoolArray::from_iter([Some(false), None]);
        let chunked = ChunkedArray::try_new(
            vec![b1.into_array(), b2.into_array()],
            DType::Bool(Nullability::Nullable),
        )?;
        let result = run_sum(&chunked.into_array())?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(2));
        Ok(())
    }

    #[test]
    fn sum_chunked_checked_overflow() -> VortexResult<()> {
        let chunked = ChunkedArray::try_new(
            vec![buffer![i64::MAX].into_array(), buffer![1i64].into_array()],
            DType::Primitive(PType::I64, Nullability::NonNullable),
        )?;
        let result = run_sum(&chunked.into_array())?;
        assert!(result.is_null());
        Ok(())
    }

    #[test]
    fn sum_chunked_nested() -> VortexResult<()> {
        let inner = ChunkedArray::try_new(
            vec![buffer![1i32, 2].into_array(), buffer![3i32].into_array()],
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )?;
        let outer = ChunkedArray::try_new(
            vec![inner.into_array(), buffer![4i32, 5, 6].into_array()],
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )?;
        let result = run_sum(&outer.into_array())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(21));
        Ok(())
    }
}
