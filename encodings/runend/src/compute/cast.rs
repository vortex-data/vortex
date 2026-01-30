// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::compute::CastKernel;
use vortex_array::compute::CastKernelAdapter;
use vortex_array::compute::cast;
use vortex_array::register_kernel;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::RunEndArray;
use crate::RunEndVTable;

impl CastKernel for RunEndVTable {
    fn cast(&self, array: &RunEndArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Cast the values array to the target type
        let casted_values = cast(array.values(), dtype)?;

        // SAFETY: casting does not affect the ends being valid
        unsafe {
            Ok(Some(
                RunEndArray::new_unchecked(
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

register_kernel!(CastKernelAdapter(RunEndVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::Array;
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::cast;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;

    use crate::RunEndArray;

    #[test]
    fn test_cast_runend_i32_to_i64() {
        let runend = RunEndArray::try_new(
            buffer![3u64, 5, 8, 10].into_array(),
            buffer![100i32, 200, 100, 300].into_array(),
        )
        .unwrap();

        let casted = cast(
            runend.as_ref(),
            &DType::Primitive(PType::I64, Nullability::NonNullable),
        )
        .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );

        // Verify by decoding to canonical form
        let decoded = casted.to_primitive();
        // RunEnd encoding should expand to [100, 100, 100, 200, 200, 100, 100, 100, 300, 300]
        assert_eq!(decoded.len(), 10);
        assert_eq!(
            TryInto::<i64>::try_into(decoded.scalar_at(0).unwrap().as_ref())
                .ok()
                .unwrap(),
            100i64
        );
        assert_eq!(
            TryInto::<i64>::try_into(decoded.scalar_at(3).unwrap().as_ref())
                .ok()
                .unwrap(),
            200i64
        );
        assert_eq!(
            TryInto::<i64>::try_into(decoded.scalar_at(5).unwrap().as_ref())
                .ok()
                .unwrap(),
            100i64
        );
        assert_eq!(
            TryInto::<i64>::try_into(decoded.scalar_at(8).unwrap().as_ref())
                .ok()
                .unwrap(),
            300i64
        );
    }

    #[test]
    fn test_cast_runend_nullable() {
        let runend = RunEndArray::try_new(
            buffer![2u64, 4, 7].into_array(),
            PrimitiveArray::from_option_iter([Some(10i32), None, Some(20)]).into_array(),
        )
        .unwrap();

        let casted = cast(
            runend.as_ref(),
            &DType::Primitive(PType::I64, Nullability::Nullable),
        )
        .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::I64, Nullability::Nullable)
        );
    }

    #[test]
    fn test_cast_runend_with_offset() {
        // Create a RunEndArray: [100, 100, 100, 200, 200, 300, 300, 300, 300, 300]
        let runend = RunEndArray::try_new(
            buffer![3u64, 5, 10].into_array(),
            buffer![100i32, 200, 300].into_array(),
        )
        .unwrap();

        // Slice it to get offset 3, length 5: [200, 200, 300, 300, 300]
        let sliced = runend.slice(3..8).unwrap();

        // Verify the slice is correct before casting
        assert_arrays_eq!(sliced, PrimitiveArray::from_iter([200, 200, 300, 300, 300]));

        // Cast the sliced array
        let casted = cast(
            sliced.as_ref(),
            &DType::Primitive(PType::I64, Nullability::NonNullable),
        )
        .unwrap();

        // Verify the cast preserved the offset
        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_iter([200i64, 200, 300, 300, 300])
        );
    }

    #[rstest]
    #[case(RunEndArray::try_new(
        buffer![3u64, 5, 8].into_array(),
        buffer![100i32, 200, 300].into_array()
    ).unwrap())]
    #[case(RunEndArray::try_new(
        buffer![1u64, 4, 10].into_array(),
        buffer![1.5f32, 2.5, 3.5].into_array()
    ).unwrap())]
    #[case(RunEndArray::try_new(
        buffer![2u64, 3, 5].into_array(),
        PrimitiveArray::from_option_iter([Some(42i32), None, Some(84)]).into_array()
    ).unwrap())]
    #[case(RunEndArray::try_new(
        buffer![10u64].into_array(),
        buffer![255u8].into_array()
    ).unwrap())]
    #[case(RunEndArray::try_new(
        buffer![2u64, 4, 6, 8, 10].into_array(),
        BoolArray::from_iter(vec![true, false, true, false, true]).into_array()
    ).unwrap())]
    fn test_cast_runend_conformance(#[case] array: RunEndArray) {
        test_cast_conformance(array.as_ref());
    }
}
