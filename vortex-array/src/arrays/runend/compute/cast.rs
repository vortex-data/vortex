// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ArrayView;
use crate::IntoArray;
use crate::arrays::runend::RunEnd;
use crate::arrays::runend::array::RunEndArrayExt;
use crate::arrays::runend::array::RunEndArraySlotsExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::scalar_fn::fns::cast::CastReduce;
impl CastReduce for RunEnd {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Cast the values array to the target type
        let casted_values = array.values().cast(dtype.clone())?;

        // SAFETY: casting does not affect the ends being valid
        unsafe {
            Ok(Some(
                RunEnd::new_unchecked(
                    array.ends().clone(),
                    casted_values,
                    array.offset(),
                    array.len(),
                )
                .into_array(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::runend::RunEnd;
    use crate::arrays::runend::RunEndArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

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
