// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_error::VortexResult;

use crate::RunEnd;
use crate::array::RunEndArrayExt;
impl CastReduce for RunEnd {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Cast the values array to the target type
        let casted_values = array.values().cast(dtype.clone())?;

        // TODO(ctx): trait fixes - CastReduce::cast has a fixed signature.
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // SAFETY: casting does not affect the ends being valid
        unsafe {
            Ok(Some(
                RunEnd::new_unchecked(
                    array.ends().clone(),
                    casted_values,
                    array.offset(),
                    array.len(),
                    &mut ctx,
                )
                .into_array(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_buffer::buffer;

    use crate::RunEnd;
    use crate::RunEndArray;

    #[test]
    fn test_cast_runend_i32_to_i64() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
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

        // Verify by decoding to canonical form
        let decoded = casted.execute::<PrimitiveArray>(&mut ctx).unwrap();
        // RunEnd encoding should expand to [100, 100, 100, 200, 200, 100, 100, 100, 300, 300]
        assert_eq!(decoded.len(), 10);
        assert_eq!(
            TryInto::<i64>::try_into(&decoded.execute_scalar(0, &mut ctx).unwrap()).unwrap(),
            100i64
        );
        assert_eq!(
            TryInto::<i64>::try_into(&decoded.execute_scalar(3, &mut ctx).unwrap()).unwrap(),
            200i64
        );
        assert_eq!(
            TryInto::<i64>::try_into(&decoded.execute_scalar(5, &mut ctx).unwrap()).unwrap(),
            100i64
        );
        assert_eq!(
            TryInto::<i64>::try_into(&decoded.execute_scalar(8, &mut ctx).unwrap()).unwrap(),
            300i64
        );
    }

    #[test]
    fn test_cast_runend_nullable() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
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
    }

    #[test]
    fn test_cast_runend_with_offset() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
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
        assert_arrays_eq!(sliced, PrimitiveArray::from_iter([200, 200, 300, 300, 300]));

        // Cast the sliced array
        let casted = sliced
            .cast(DType::Primitive(PType::I64, Nullability::NonNullable))
            .unwrap();

        // Verify the cast preserved the offset
        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_iter([200i64, 200, 300, 300, 300])
        );
    }

    type RunEndBuilder = fn(&mut vortex_array::ExecutionCtx) -> RunEndArray;

    #[rstest]
    #[case(|ctx: &mut vortex_array::ExecutionCtx| RunEnd::try_new(
        buffer![3u64, 5, 8].into_array(),
        buffer![100i32, 200, 300].into_array(),
        ctx,
    ).unwrap())]
    #[case(|ctx: &mut vortex_array::ExecutionCtx| RunEnd::try_new(
        buffer![1u64, 4, 10].into_array(),
        buffer![1.5f32, 2.5, 3.5].into_array(),
        ctx,
    ).unwrap())]
    #[case(|ctx: &mut vortex_array::ExecutionCtx| RunEnd::try_new(
        buffer![2u64, 3, 5].into_array(),
        PrimitiveArray::from_option_iter([Some(42i32), None, Some(84)]).into_array(),
        ctx,
    ).unwrap())]
    #[case(|ctx: &mut vortex_array::ExecutionCtx| RunEnd::try_new(
        buffer![10u64].into_array(),
        buffer![255u8].into_array(),
        ctx,
    ).unwrap())]
    #[case(|ctx: &mut vortex_array::ExecutionCtx| RunEnd::try_new(
        buffer![2u64, 4, 6, 8, 10].into_array(),
        BoolArray::from_iter(vec![true, false, true, false, true]).into_array(),
        ctx,
    ).unwrap())]
    fn test_cast_runend_conformance(#[case] build: RunEndBuilder) {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let array = build(&mut ctx);
        test_cast_conformance(&array.into_array());
    }
}
