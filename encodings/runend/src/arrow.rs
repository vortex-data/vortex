// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::RunArray;
use arrow_array::types::RunEndIndexType;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrow::FromArrowArray;
use vortex_array::dtype::NativePType;
use vortex_array::scalar::PValue;
use vortex_array::search_sorted::SearchSorted;
use vortex_array::search_sorted::SearchSortedSide;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::RunEndData;
use crate::ops::find_slice_end_index;
impl<R: RunEndIndexType> FromArrowArray<&RunArray<R>> for RunEndData
where
    R::Native: NativePType,
{
    fn from_arrow(array: &RunArray<R>, nullable: bool) -> VortexResult<Self> {
        let offset = array.run_ends().offset();
        let len = array.run_ends().len();
        let ends_buf =
            Buffer::<R::Native>::from_arrow_scalar_buffer(array.run_ends().inner().clone());
        let ends = PrimitiveArray::new(ends_buf, Validity::NonNullable)
            .reinterpret_cast(R::Native::PTYPE.to_unsigned());
        let values = ArrayRef::from_arrow(array.values().as_ref(), nullable)?;

        let ends_array = ends.into_array();
        let (ends_slice, values_slice) = if offset == 0 && len == array.run_ends().max_value() {
            (ends_array, values)
        } else {
            let slice_begin = ends_array
                .as_primitive_typed()
                .search_sorted(&PValue::from(offset), SearchSortedSide::Right)?
                .to_ends_index(ends_array.len());
            let slice_end = find_slice_end_index(&ends_array, offset + len)?;

            (
                ends_array.slice(slice_begin..slice_end)?,
                values.slice(slice_begin..slice_end)?,
            )
        };

        // SAFETY: arrow-rs enforces the RunEndArray invariants, we inherit their guarantees
        Ok(unsafe { RunEndData::new_unchecked(ends_slice, values_slice, offset, len) })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use arrow_array::Float64Array;
    use arrow_array::Int32Array;
    use arrow_array::Int64Array;
    use arrow_array::RunArray;
    use arrow_array::types::Int32Type;
    use arrow_array::types::Int64Type;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use rstest::rstest;
    use vortex_array::IntoArray as _;
    use vortex_array::VortexSessionExecute as _;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrow::ArrowArrayExecutor;
    use vortex_array::arrow::FromArrowArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::session::ArraySession;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::RunEndData;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[test]
    fn test_arrow_run_array_to_vortex() -> VortexResult<()> {
        // Create an Arrow RunArray with UInt32 run ends and Int32 values
        // Run ends: [3, 5, 8] means runs of length 3, 2, 3
        // Values: [10, 20, 30] means values 10, 10, 10, 20, 20, 30, 30, 30
        let run_ends = Int32Array::from(vec![3i32, 5, 8]);
        let values = Int32Array::from(vec![10, 20, 30]);
        let arrow_run_array = RunArray::<Int32Type>::try_new(&run_ends, &values).unwrap();

        // Convert to Vortex
        let vortex_array = RunEndData::from_arrow(&arrow_run_array, false)?;

        assert_arrays_eq!(
            vortex_array.into_array(),
            buffer![10i32, 10, 10, 20, 20, 30, 30, 30].into_array()
        );
        Ok(())
    }

    #[test]
    fn test_arrow_run_array_with_nulls_to_vortex() -> VortexResult<()> {
        // Create an Arrow RunArray with nullable values
        let run_ends = Int32Array::from(vec![2i32, 4, 6]);
        let values = Int32Array::from(vec![Some(100), None, Some(300)]);
        let arrow_run_array = RunArray::<Int32Type>::try_new(&run_ends, &values).unwrap();

        // Convert to Vortex with nullable=true
        let vortex_array = RunEndData::from_arrow(&arrow_run_array, true)?;

        assert_arrays_eq!(
            vortex_array.into_array(),
            PrimitiveArray::from_option_iter([
                Some(100i32),
                Some(100i32),
                None,
                None,
                Some(300i32),
                Some(300i32)
            ])
        );
        Ok(())
    }

    #[test]
    fn test_arrow_run_array_with_different_types() -> VortexResult<()> {
        // Test with UInt64 run ends and Float64 values
        let run_ends = Int64Array::from(vec![1i64, 3, 4]);
        let values = Float64Array::from(vec![1.5, 2.5, 3.5]);
        let arrow_run_array = RunArray::<Int64Type>::try_new(&run_ends, &values).unwrap();

        // Convert to Vortex
        let vortex_array = RunEndData::from_arrow(&arrow_run_array, false)?;

        assert_arrays_eq!(vortex_array, buffer![1.5f64, 2.5, 2.5, 3.5].into_array());
        Ok(())
    }

    #[test]
    fn test_sliced_arrow_run_array_to_vortex() -> VortexResult<()> {
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
        let vortex_array = RunEndData::from_arrow(&sliced_array, false)?;
        assert_arrays_eq!(
            vortex_array,
            buffer![100, 200, 200, 200, 300, 300].into_array()
        );
        Ok(())
    }

    #[test]
    fn test_sliced_arrow_run_array_with_nulls_to_vortex() -> VortexResult<()> {
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
        let vortex_array = RunEndData::from_arrow(&sliced_array, true)?;

        assert_arrays_eq!(
            vortex_array,
            PrimitiveArray::from_option_iter([
                None,
                None,
                Some(30i64),
                Some(30),
                Some(30),
                Some(40),
            ])
        );
        Ok(())
    }

    #[test]
    fn test_sliced_to_0_arrow_run_array_with_nulls_to_vortex() -> VortexResult<()> {
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
        let vortex_array = RunEndData::from_arrow(&sliced_array, true)?;

        // Verify properties
        assert_eq!(vortex_array.len(), 0);
        assert_eq!(
            vortex_array.dtype(),
            &DType::Primitive(PType::I64, Nullability::Nullable)
        );
        Ok(())
    }

    fn ree_type(ends: DataType, values_dtype: DataType) -> DataType {
        DataType::RunEndEncoded(
            Arc::new(Field::new("run_ends", ends, false)),
            Arc::new(Field::new("values", values_dtype, true)),
        )
    }

    fn execute(
        array: vortex_array::ArrayRef,
        dt: &DataType,
    ) -> VortexResult<arrow_array::ArrayRef> {
        array.execute_arrow(Some(dt), &mut SESSION.create_execution_ctx())
    }

    #[test]
    fn test_roundtrip_arrow_to_vortex_to_arrow() -> VortexResult<()> {
        let original = RunArray::<Int32Type>::try_new(
            &Int32Array::from(vec![3i32, 5, 8]),
            &Int32Array::from(vec![10, 20, 30]),
        )?;
        let vortex_array = RunEndData::from_arrow(&original, false)?;
        let target = ree_type(DataType::Int32, DataType::Int32);
        let result = execute(vortex_array.into_array(), &target)?;

        let expected = RunArray::<Int32Type>::try_new(
            &Int32Array::from(vec![3, 5, 8]),
            &Int32Array::from(vec![10, 20, 30]),
        )?;
        assert_eq!(result.as_ref(), &expected);
        Ok(())
    }

    /// Slicing a RunEndArray and converting to Arrow REE must produce
    /// correctly trimmed and adjusted run ends for both zero and non-zero offsets.
    #[rstest]
    #[case::nonzero_offset(
        &[10i32, 10, 20, 20, 20, 30, 30],
        1..5usize,
        &[1i32, 4],
        &[10i32, 20],
    )]
    #[case::zero_offset_excess_runs(
        &[10i32, 10, 10, 20, 20, 30, 30, 30, 30, 30],
        0..4usize,
        &[3i32, 4],
        &[10i32, 20],
    )]
    fn sliced_runend_to_arrow_ree(
        #[case] input: &[i32],
        #[case] slice_range: std::ops::Range<usize>,
        #[case] expected_ends: &[i32],
        #[case] expected_values: &[i32],
    ) -> VortexResult<()> {
        let array =
            RunEndData::encode(PrimitiveArray::from_iter(input.iter().copied()).into_array())?;
        let sliced = array.into_array().slice(slice_range.clone())?;
        let target = ree_type(DataType::Int32, DataType::Int32);
        let result = execute(sliced, &target)?;

        assert_eq!(result.len(), slice_range.len());
        let expected = RunArray::<Int32Type>::try_new(
            &Int32Array::from(expected_ends.to_vec()),
            &Int32Array::from(expected_values.to_vec()),
        )?;
        assert_eq!(result.as_ref(), &expected);
        Ok(())
    }
}
