use crate::Array;
use crate::arrays::ChunkedEncoding;
use crate::compute::TakeFn;
use crate::vtable::ComputeVTable;

mod cast;
mod compare;
mod elementwise;
mod fill_null;
mod filter;
mod invert;
mod is_constant;
mod is_sorted;
mod mask;
mod min_max;
mod sum;
mod take;

impl ComputeVTable for ChunkedEncoding {
    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::IntoArray;
    use crate::array::Array;
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
