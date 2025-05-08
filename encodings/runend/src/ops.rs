use vortex_array::arrays::ConstantArray;
use vortex_array::{Array, ArrayOperationsImpl, ArrayRef};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::RunEndArray;

impl ArrayOperationsImpl for RunEndArray {
    fn _slice(&self, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        let new_length = stop - start;

        let (slice_begin, slice_end) = if new_length == 0 {
            let values_len = self.values().len();
            (values_len, values_len)
        } else {
            let physical_start = self.find_physical_index(start)?;
            let physical_stop = self.find_physical_index(stop)?;

            (physical_start, physical_stop + 1)
        };

        if slice_begin + 1 == slice_end {
            let value = self.values().scalar_at(slice_begin)?;
            return Ok(ConstantArray::new(value, new_length).into_array());
        }

        Ok(RunEndArray::with_offset_and_length(
            self.ends().slice(slice_begin, slice_end)?,
            self.values().slice(slice_begin, slice_end)?,
            if new_length == 0 {
                0
            } else {
                start + self.offset()
            },
            new_length,
        )?
        .into_array())
    }

    fn _scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        self.values().scalar_at(self.find_physical_index(index)?)
    }
}

#[cfg(test)]
mod tests {

    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::{Array, ArrayStatistics, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::RunEndArray;

    #[test]
    fn slice_array() {
        let arr = RunEndArray::try_new(
            buffer![2u32, 5, 10].into_array(),
            buffer![1i32, 2, 3].into_array(),
        )
        .unwrap()
        .slice(3, 8)
        .unwrap();
        assert_eq!(
            arr.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_eq!(arr.len(), 5);

        assert_eq!(
            arr.to_primitive().unwrap().as_slice::<i32>(),
            vec![2, 2, 3, 3, 3]
        );
    }

    #[test]
    fn double_slice() {
        let arr = RunEndArray::try_new(
            buffer![2u32, 5, 10].into_array(),
            buffer![1i32, 2, 3].into_array(),
        )
        .unwrap()
        .slice(3, 8)
        .unwrap();
        assert_eq!(arr.len(), 5);

        let doubly_sliced = arr.slice(0, 3).unwrap();

        assert_eq!(
            doubly_sliced.to_primitive().unwrap().as_slice::<i32>(),
            vec![2, 2, 3]
        );
    }

    #[test]
    fn slice_end_inclusive() {
        let arr = RunEndArray::try_new(
            buffer![2u32, 5, 10].into_array(),
            buffer![1i32, 2, 3].into_array(),
        )
        .unwrap()
        .slice(4, 10)
        .unwrap();
        assert_eq!(
            arr.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_eq!(arr.len(), 6);

        assert_eq!(
            arr.to_primitive().unwrap().as_slice::<i32>(),
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

        let sliced_array = re_array.slice(re_array.len(), re_array.len()).unwrap();
        assert!(sliced_array.is_empty());
    }

    #[test]
    fn slice_single_end() {
        let re_array = RunEndArray::try_new(
            buffer![7_u64, 10].into_array(),
            buffer![2_u64, 3].into_array(),
        )
        .unwrap();

        assert_eq!(re_array.len(), 10);

        let sliced_array = re_array.slice(2, 5).unwrap();

        assert!(sliced_array.is_constant())
    }

    #[test]
    fn ree_scalar_at_end() {
        let scalar = RunEndArray::encode(
            PrimitiveArray::from_iter([1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5]).into_array(),
        )
        .unwrap()
        .scalar_at(11)
        .unwrap();
        assert_eq!(scalar, 5.into());
    }
}
