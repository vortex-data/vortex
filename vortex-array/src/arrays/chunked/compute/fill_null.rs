use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::{ChunkedArray, ChunkedEncoding};
use crate::compute::{fill_null, FillNullFn};
use crate::{Array, ArrayRef, IntoArray};

impl FillNullFn<&ChunkedArray> for ChunkedEncoding {
    fn fill_null(&self, array: &ChunkedArray, fill_value: Scalar) -> VortexResult<ArrayRef> {
        Ok(ChunkedArray::new_unchecked(
            array
                .chunks()
                .iter()
                .map(|c| fill_null(c, fill_value.clone()))
                .collect::<VortexResult<Vec<_>>>()?,
            array.dtype().as_nonnullable(),
        )
        .into_array())
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::BooleanBuffer;
    use vortex_dtype::{DType, Nullability};

    use crate::array::Array;
    use crate::arrays::{BoolArray, ChunkedArray};
    use crate::compute::fill_null;
    use crate::validity::Validity;
    use crate::IntoArray;

    #[test]
    fn fill_null_chunks() {
        let chunked = ChunkedArray::try_new(
            vec![
                BoolArray::new(BooleanBuffer::new_set(5), Validity::AllInvalid).to_array(),
                BoolArray::new(BooleanBuffer::new_set(5), Validity::AllValid).to_array(),
            ],
            DType::Bool(Nullability::Nullable),
        )
        .unwrap();

        let filled = fill_null(&chunked, false.into()).unwrap();
        assert_eq!(*filled.dtype(), DType::Bool(Nullability::NonNullable));
    }
}
