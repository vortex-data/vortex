use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::arrays::{ChunkedArray, ChunkedVTable};
use crate::compute::{CastKernel, CastKernelAdapter, cast};
use crate::{ArrayRef, IntoArray, register_kernel};

impl CastKernel for ChunkedVTable {
    fn cast(&self, array: &ChunkedArray, dtype: &DType) -> VortexResult<ArrayRef> {
        let mut cast_chunks = Vec::new();
        for chunk in array.chunks() {
            cast_chunks.push(cast(chunk, dtype)?);
        }

        Ok(ChunkedArray::new_unchecked(cast_chunks, dtype.clone()).into_array())
    }
}

register_kernel!(CastKernelAdapter(ChunkedVTable).lift());

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::IntoArray;
    use crate::arrays::chunked::ChunkedArray;
    use crate::canonical::ToCanonical;
    use crate::compute::cast;

    #[test]
    fn test_cast_chunked() {
        let arr0 = buffer![0u32, 1].into_array();
        let arr1 = buffer![2u32, 3].into_array();

        let chunked = ChunkedArray::try_new(
            vec![arr0, arr1],
            DType::Primitive(PType::U32, Nullability::NonNullable),
        )
        .unwrap()
        .into_array();

        // Two levels of chunking, just to be fancy.
        let root = ChunkedArray::try_new(
            vec![chunked],
            DType::Primitive(PType::U32, Nullability::NonNullable),
        )
        .unwrap()
        .into_array();

        assert_eq!(
            cast(
                &root,
                &DType::Primitive(PType::U64, Nullability::NonNullable)
            )
            .unwrap()
            .to_primitive()
            .unwrap()
            .as_slice::<u64>(),
            &[0u64, 1, 2, 3],
        );
    }
}
