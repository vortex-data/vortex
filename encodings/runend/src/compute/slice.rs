use vortex_array::compute::{slice, SliceFn};
use vortex_array::{ArrayData, IntoArrayData};
use vortex_error::VortexResult;

use crate::{RunEndArray, RunEndEncoding};

impl SliceFn<RunEndArray> for RunEndEncoding {
    fn slice(&self, array: &RunEndArray, start: usize, stop: usize) -> VortexResult<ArrayData> {
        let new_length = stop - start;

        let (slice_begin, slice_end) = if new_length == 0 {
            let values_len = array.values().len();
            (values_len, values_len)
        } else {
            let physical_start = array.find_physical_index(start)?;
            let physical_stop = array.find_physical_index(stop)?;

            (physical_start, physical_stop + 1)
        };

        Ok(RunEndArray::with_offset_and_length(
            slice(array.ends(), slice_begin, slice_end)?,
            slice(array.values(), slice_begin, slice_end)?,
            if new_length == 0 {
                0
            } else {
                start + array.offset()
            },
            new_length,
        )?
        .into_array())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::compute::slice;
    use vortex_array::{IntoArrayData, IntoArrayVariant};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::RunEndArray;

    #[test]
    fn slice_array() {
        let arr = slice(
            RunEndArray::try_new(
                buffer![2u32, 5, 10].into_array(),
                buffer![1i32, 2, 3].into_array(),
            )
            .unwrap()
            .as_ref(),
            3,
            8,
        )
        .unwrap();
        assert_eq!(
            arr.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_eq!(arr.len(), 5);

        assert_eq!(
            arr.into_primitive().unwrap().as_slice::<i32>(),
            vec![2, 2, 3, 3, 3]
        );
    }

    #[test]
    fn double_slice() {
        let arr = slice(
            RunEndArray::try_new(
                buffer![2u32, 5, 10].into_array(),
                buffer![1i32, 2, 3].into_array(),
            )
            .unwrap()
            .as_ref(),
            3,
            8,
        )
        .unwrap();
        assert_eq!(arr.len(), 5);

        let doubly_sliced = slice(&arr, 0, 3).unwrap();

        assert_eq!(
            doubly_sliced.into_primitive().unwrap().as_slice::<i32>(),
            vec![2, 2, 3]
        );
    }

    #[test]
    fn slice_end_inclusive() {
        let arr = slice(
            RunEndArray::try_new(
                buffer![2u32, 5, 10].into_array(),
                buffer![1i32, 2, 3].into_array(),
            )
            .unwrap()
            .as_ref(),
            4,
            10,
        )
        .unwrap();
        assert_eq!(
            arr.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_eq!(arr.len(), 6);

        assert_eq!(
            arr.into_primitive().unwrap().as_slice::<i32>(),
            vec![2, 3, 3, 3, 3, 3]
        );
    }

    #[test]
    fn slice_at_end() {
        let re_array = RunEndArray::try_new(
            buffer![7_u64, 10].into_array(),
            buffer![2_u64, 3].into_array(),
        )
        .unwrap();

        assert_eq!(re_array.len(), 10);

        let sliced_array = slice(&re_array, re_array.len(), re_array.len()).unwrap();
        assert!(sliced_array.is_empty());

        let re_slice = RunEndArray::try_from(sliced_array).unwrap();
        assert!(re_slice.ends().is_empty());
        assert!(re_slice.values().is_empty())
    }
}
