// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::RunArray;
use arrow_array::types::RunEndIndexType;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrow::FromArrowArray;
use vortex_array::search_sorted::{SearchSorted, SearchSortedSide};
use vortex_array::validity::Validity;
use vortex_array::{ArrayRef, IntoArray};
use vortex_buffer::Buffer;
use vortex_dtype::NativePType;
use vortex_scalar::PValue;

use crate::RunEndArray;
use crate::ops::find_slice_end_index;

impl<R: RunEndIndexType> FromArrowArray<&RunArray<R>> for RunEndArray
where
    R::Native: NativePType,
{
    fn from_arrow(array: &RunArray<R>, nullable: bool) -> Self {
        let offset = array.run_ends().offset();
        let len = array.run_ends().len();
        let ends_buf =
            Buffer::<R::Native>::from_arrow_scalar_buffer(array.run_ends().inner().clone());
        let ends = PrimitiveArray::new(ends_buf, Validity::NonNullable)
            .reinterpret_cast(R::Native::PTYPE.to_unsigned());
        let values = ArrayRef::from_arrow(array.values().as_ref(), nullable);

        let (ends_slice, values_slice) = if offset == 0 && len == array.run_ends().max_value() {
            (ends.into_array(), values)
        } else {
            let slice_begin = ends
                .as_primitive_typed()
                .search_sorted(&PValue::from(offset), SearchSortedSide::Right)
                .to_ends_index(ends.len());
            let slice_end = find_slice_end_index(ends.as_ref(), offset + len);

            (
                ends.slice(slice_begin..slice_end),
                values.slice(slice_begin..slice_end),
            )
        };

        // SAFETY: arrow-rs enforces the RunEndArray invariants, we inherit their guarantees
        unsafe { RunEndArray::new_unchecked(ends_slice, values_slice, offset, len) }
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::types::{Int32Type, Int64Type};
    use arrow_array::{Float64Array, Int32Array, Int64Array, RunArray};
    use vortex_array::arrow::FromArrowArray;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::RunEndArray;

    #[test]
    fn test_arrow_run_array_to_vortex() {
        // Create an Arrow RunArray with UInt32 run ends and Int32 values
        // Run ends: [3, 5, 8] means runs of length 3, 2, 3
        // Values: [10, 20, 30] means values 10, 10, 10, 20, 20, 30, 30, 30
        let run_ends = Int32Array::from(vec![3i32, 5, 8]);
        let values = Int32Array::from(vec![10, 20, 30]);
        let arrow_run_array = RunArray::<Int32Type>::try_new(&run_ends, &values).unwrap();

        // Convert to Vortex
        let vortex_array = RunEndArray::from_arrow(&arrow_run_array, false);

        // Verify basic properties
        assert_eq!(vortex_array.len(), 8);
        assert_eq!(
            vortex_array.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );

        // Verify the values at different positions
        assert_eq!(vortex_array.scalar_at(0), 10.into()); // First run
        assert_eq!(vortex_array.scalar_at(2), 10.into()); // Still first run
        assert_eq!(vortex_array.scalar_at(3), 20.into()); // Second run
        assert_eq!(vortex_array.scalar_at(4), 20.into()); // Still second run
        assert_eq!(vortex_array.scalar_at(5), 30.into()); // Third run
        assert_eq!(vortex_array.scalar_at(7), 30.into()); // Still third run
    }

    #[test]
    fn test_arrow_run_array_with_nulls_to_vortex() {
        // Create an Arrow RunArray with nullable values
        let run_ends = Int32Array::from(vec![2i32, 4, 6]);
        let values = Int32Array::from(vec![Some(100), None, Some(300)]);
        let arrow_run_array = RunArray::<Int32Type>::try_new(&run_ends, &values).unwrap();

        // Convert to Vortex with nullable=true
        let vortex_array = RunEndArray::from_arrow(&arrow_run_array, true);

        // Verify basic properties
        assert_eq!(vortex_array.len(), 6);
        assert_eq!(
            vortex_array.dtype(),
            &DType::Primitive(PType::I32, Nullability::Nullable)
        );

        // Verify the values
        assert_eq!(vortex_array.scalar_at(0), 100.into());
        assert_eq!(vortex_array.scalar_at(1), 100.into());
        assert!(vortex_array.scalar_at(2).is_null()); // Null value
        assert!(vortex_array.scalar_at(3).is_null()); // Null value
        assert_eq!(vortex_array.scalar_at(4), 300.into());
        assert_eq!(vortex_array.scalar_at(5), 300.into());
    }

    #[test]
    fn test_arrow_run_array_with_different_types() {
        // Test with UInt64 run ends and Float64 values
        let run_ends = Int64Array::from(vec![1i64, 3, 4]);
        let values = Float64Array::from(vec![1.5, 2.5, 3.5]);
        let arrow_run_array = RunArray::<Int64Type>::try_new(&run_ends, &values).unwrap();

        // Convert to Vortex
        let vortex_array = RunEndArray::from_arrow(&arrow_run_array, false);

        // Verify properties
        assert_eq!(vortex_array.len(), 4);
        assert_eq!(
            vortex_array.dtype(),
            &DType::Primitive(PType::F64, Nullability::NonNullable)
        );

        // Verify values
        assert_eq!(vortex_array.scalar_at(0), 1.5.into());
        assert_eq!(vortex_array.scalar_at(1), 2.5.into());
        assert_eq!(vortex_array.scalar_at(2), 2.5.into());
        assert_eq!(vortex_array.scalar_at(3), 3.5.into());
    }

    #[test]
    fn test_sliced_arrow_run_array_to_vortex() {
        // Create an Arrow RunArray with run ends and values
        // Run ends: [2, 5, 8, 10] means runs of length 2, 3, 3, 2
        // Values: [100, 200, 300, 400] means: 100, 100, 200, 200, 200, 300, 300, 300, 400, 400
        let run_ends = Int32Array::from(vec![2i32, 5, 8, 10]);
        let values = Int32Array::from(vec![100, 200, 300, 400]);
        let arrow_run_array = RunArray::<Int32Type>::try_new(&run_ends, &values).unwrap();

        // Slice the array from index 1 to 7 (length 6)
        // This should give us: 100, 200, 200, 200, 300, 300
        let sliced_array = arrow_run_array.slice(1, 6);

        // Convert the sliced array to Vortex
        let vortex_array = RunEndArray::from_arrow(&sliced_array, false);

        // Verify the sliced array properties
        assert_eq!(vortex_array.len(), 6);
        assert_eq!(
            vortex_array.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );

        // Verify the values in the sliced array
        assert_eq!(vortex_array.scalar_at(0), 100.into()); // Index 1 of original
        assert_eq!(vortex_array.scalar_at(1), 200.into()); // Index 2 of original
        assert_eq!(vortex_array.scalar_at(2), 200.into()); // Index 3 of original
        assert_eq!(vortex_array.scalar_at(3), 200.into()); // Index 4 of original
        assert_eq!(vortex_array.scalar_at(4), 300.into()); // Index 5 of original
        assert_eq!(vortex_array.scalar_at(5), 300.into()); // Index 6 of original
    }

    #[test]
    fn test_sliced_arrow_run_array_with_nulls_to_vortex() {
        // Create an Arrow RunArray with nullable values
        // Run ends: [3, 6, 9, 12] means runs of length 3, 3, 3, 3
        // Values: [Some(10), None, Some(30), Some(40)]
        let run_ends = Int64Array::from(vec![3i64, 6, 9, 12]);
        let values = Int64Array::from(vec![Some(10), None, Some(30), Some(40)]);
        let arrow_run_array = RunArray::<Int64Type>::try_new(&run_ends, &values).unwrap();

        // Slice from index 4 to 10 (length 6)
        // Original: 10, 10, 10, null, null, null, 30, 30, 30, 40, 40, 40
        // Sliced:   null, null, 30, 30, 30, 40
        let sliced_array = arrow_run_array.slice(4, 6);

        // Convert to Vortex with nullable=true
        let vortex_array = RunEndArray::from_arrow(&sliced_array, true);

        // Verify properties
        assert_eq!(vortex_array.len(), 6);
        assert_eq!(
            vortex_array.dtype(),
            &DType::Primitive(PType::I64, Nullability::Nullable)
        );

        // Verify the values in the sliced array
        assert!(vortex_array.scalar_at(0).is_null());
        assert!(vortex_array.scalar_at(1).is_null());
        assert_eq!(vortex_array.scalar_at(2), 30i64.into());
        assert_eq!(vortex_array.scalar_at(3), 30i64.into());
        assert_eq!(vortex_array.scalar_at(4), 30i64.into());
        assert_eq!(vortex_array.scalar_at(5), 40i64.into());
    }

    #[test]
    fn test_sliced_to_0_arrow_run_array_with_nulls_to_vortex() {
        // Create an Arrow RunArray with nullable values
        // Run ends: [3, 6, 9, 12] means runs of length 3, 3, 3, 3
        // Values: [Some(10), None, Some(30), Some(40)]
        let run_ends = Int64Array::from(vec![3i64, 6, 9, 12]);
        let values = Int64Array::from(vec![Some(10), None, Some(30), Some(40)]);
        let arrow_run_array = RunArray::<Int64Type>::try_new(&run_ends, &values).unwrap();

        // Slice from index 4 to 4 (length 0)
        // Original: 10, 10, 10, null, null, null, 30, 30, 30, 40, 40, 40
        // Sliced:   [ ]
        let sliced_array = arrow_run_array.slice(4, 0);

        // Convert to Vortex with nullable=true
        let vortex_array = RunEndArray::from_arrow(&sliced_array, true);

        // Verify properties
        assert_eq!(vortex_array.len(), 0);
        assert_eq!(
            vortex_array.dtype(),
            &DType::Primitive(PType::I64, Nullability::Nullable)
        );
    }
}
