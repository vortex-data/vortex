// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::runend::RunEnd;
use crate::arrays::runend::array::RunEndArrayExt;
use crate::arrays::runend::array::RunEndArraySlotsExt;
use crate::arrays::runend::array::has_all_runs_referenced;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::scalar_fn::fns::cast::CastKernel;
use crate::scalar_fn::fns::cast::CastReduce;

impl CastReduce for RunEnd {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let safe_for_unused_values = match (array.dtype(), dtype) {
            (DType::Primitive(from, _), DType::Primitive(to, _)) => from.casts_losslessly_to(*to),
            (from, to) => from.as_nullable() == to.as_nullable(),
        };
        let may_reject_unused_values = !safe_for_unused_values
            || (!dtype.is_nullable() && !array.values().validity()?.definitely_no_nulls());
        if may_reject_unused_values && !has_all_runs_referenced(array) {
            return Ok(None);
        }

        cast_runs(array, dtype, None).map(Some)
    }
}

impl CastKernel for RunEnd {
    fn cast(
        array: ArrayView<'_, Self>,
        dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if array.is_empty() {
            return array.values().slice(0..0)?.cast(dtype.clone()).map(Some);
        }

        cast_runs(array, dtype, Some(ctx)).map(Some)
    }
}

/// Cast the run values. With an execution context, first trim runs outside the logical range.
fn cast_runs(
    array: ArrayView<'_, RunEnd>,
    dtype: &DType,
    ctx: Option<&mut ExecutionCtx>,
) -> VortexResult<ArrayRef> {
    let (ends, values) = if let Some(ctx) = ctx {
        let runs = array.find_physical_range(0..array.len(), ctx)?;
        (
            array.ends().slice(runs.clone())?,
            array.values().slice(runs)?,
        )
    } else {
        (array.ends().clone(), array.values().clone())
    };
    let values = values.cast(dtype.clone())?;

    // SAFETY: Both children retain the same runs, which cover the entire logical range.
    Ok(unsafe { RunEnd::new_unchecked(ends, values, array.offset(), array.len()) }.into_array())
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::ExtensionArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::runend::RunEnd;
    use crate::arrays::runend::RunEndArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::test_harness::ExpectedOutcome;
    use crate::test_harness::assert_execution_paths;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn test_cast_runend_ignores_unreferenced_null_timestamp() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let timestamp = Timestamp::new(TimeUnit::Milliseconds, Nullability::Nullable).erased();
        let values = ExtensionArray::try_new(
            timestamp,
            PrimitiveArray::from_option_iter([None, Some(5i64)]).into_array(),
        )?;
        let runs = RunEnd::try_new_offset_length(
            buffer![1u32, 2].into_array(),
            values.into_array(),
            1,
            1,
            &mut ctx,
        )?;
        let target = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
        let expected = ExtensionArray::try_new(target.clone(), buffer![5i64].into_array())?;
        assert_execution_paths(
            &[runs.into_array()],
            |inputs| inputs[0].cast(DType::Extension(target.clone())),
            ExpectedOutcome::Array(expected.into_array()),
            &mut ctx,
        );
        Ok(())
    }

    #[test]
    fn test_cast_runend_ignores_unreferenced_overflow() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let runs = RunEnd::try_new_offset_length(
            buffer![1u32, 2].into_array(),
            buffer![5i64, i64::MAX].into_array(),
            0,
            1,
            &mut ctx,
        )?;
        assert_execution_paths(
            &[runs.into_array()],
            |inputs| inputs[0].cast(DType::Primitive(PType::I32, Nullability::NonNullable)),
            ExpectedOutcome::Array(buffer![5i32].into_array()),
            &mut ctx,
        );
        Ok(())
    }

    #[test]
    fn test_cast_empty_runend_rejects_unsupported_dtype() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let runs = RunEnd::try_new_offset_length(
            buffer![1u32].into_array(),
            buffer![5i64].into_array(),
            0,
            0,
            &mut ctx,
        )?;
        assert_execution_paths(
            &[runs.into_array()],
            |inputs| inputs[0].cast(DType::Utf8(Nullability::NonNullable)),
            ExpectedOutcome::Error,
            &mut ctx,
        );
        Ok(())
    }

    #[test]
    fn test_cast_runend_i32_to_i64() {
        let mut ctx = SESSION.create_execution_ctx();
        let runend = RunEnd::try_new(
            buffer![3u64, 5, 8, 10].into_array(),
            buffer![100i32, 200, 100, 300].into_array(),
            &mut ctx,
        )
        .unwrap();

        let casted = runend
            .into_array()
            .cast(DType::Primitive(PType::I64, Nullability::NonNullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );

        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_iter([100i64, 100, 100, 200, 200, 100, 100, 100, 300, 300]),
            &mut ctx
        );
    }

    #[test]
    fn test_cast_runend_nullable() {
        let mut ctx = SESSION.create_execution_ctx();
        let runend = RunEnd::try_new(
            buffer![2u64, 4, 7].into_array(),
            PrimitiveArray::from_option_iter([Some(10i32), None, Some(20)]).into_array(),
            &mut ctx,
        )
        .unwrap();

        let casted = runend
            .into_array()
            .cast(DType::Primitive(PType::I64, Nullability::Nullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::I64, Nullability::Nullable)
        );
        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_option_iter([
                Some(10i64),
                Some(10),
                None,
                None,
                Some(20),
                Some(20),
                Some(20)
            ]),
            &mut ctx
        );
    }

    #[test]
    fn test_cast_runend_with_offset() {
        let mut ctx = SESSION.create_execution_ctx();
        // Create a RunEndArray: [100, 100, 100, 200, 200, 300, 300, 300, 300, 300]
        let runend = RunEnd::try_new(
            buffer![3u64, 5, 10].into_array(),
            buffer![100i32, 200, 300].into_array(),
            &mut ctx,
        )
        .unwrap();

        // Slice it to get offset 3, length 5: [200, 200, 300, 300, 300]
        let sliced = runend.slice(3..8).unwrap();

        // Verify the slice is correct before casting
        assert_arrays_eq!(
            sliced,
            PrimitiveArray::from_iter([200, 200, 300, 300, 300]),
            &mut ctx
        );

        // Cast the sliced array
        let casted = sliced
            .cast(DType::Primitive(PType::I64, Nullability::NonNullable))
            .unwrap();

        // Verify the cast preserved the offset
        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_iter([200i64, 200, 300, 300, 300]),
            &mut ctx
        );
    }

    type RunEndBuilder = fn(&mut crate::ExecutionCtx) -> RunEndArray;

    #[rstest]
    #[case(|ctx: &mut crate::ExecutionCtx| RunEnd::try_new(
        buffer![3u64, 5, 8].into_array(),
        buffer![100i32, 200, 300].into_array(),
        ctx,
    ).unwrap())]
    #[case(|ctx: &mut crate::ExecutionCtx| RunEnd::try_new(
        buffer![1u64, 4, 10].into_array(),
        buffer![1.5f32, 2.5, 3.5].into_array(),
        ctx,
    ).unwrap())]
    #[case(|ctx: &mut crate::ExecutionCtx| RunEnd::try_new(
        buffer![2u64, 3, 5].into_array(),
        PrimitiveArray::from_option_iter([Some(42i32), None, Some(84)]).into_array(),
        ctx,
    ).unwrap())]
    #[case(|ctx: &mut crate::ExecutionCtx| RunEnd::try_new(
        buffer![10u64].into_array(),
        buffer![255u8].into_array(),
        ctx,
    ).unwrap())]
    #[case(|ctx: &mut crate::ExecutionCtx| RunEnd::try_new(
        buffer![2u64, 4, 6, 8, 10].into_array(),
        BoolArray::from_iter(vec![true, false, true, false, true]).into_array(),
        ctx,
    ).unwrap())]
    fn test_cast_runend_conformance(#[case] build: RunEndBuilder) {
        let mut ctx = SESSION.create_execution_ctx();
        let array = build(&mut ctx);
        test_cast_conformance(&array.into_array(), &mut ctx);
    }
}
