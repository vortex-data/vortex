use vortex_array::compute::unary::{scalar_at, ScalarAtFn};
use vortex_array::ArrayDType;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{unpack_single, BitPackedArray, BitPackedEncoding};

impl ScalarAtFn<BitPackedArray> for BitPackedEncoding {
    fn scalar_at(&self, array: &BitPackedArray, index: usize) -> VortexResult<Scalar> {
        if let Some(patches) = array.patches() {
            // NB: All non-null values are considered patches
            if patches.with_dyn(|a| a.is_valid(index)) {
                return scalar_at(&patches, index)?.cast(array.dtype());
            }
        }

        unpack_single(array, index)?.cast(array.dtype())
    }
}

#[cfg(test)]
mod test {
    use vortex_array::array::{PrimitiveArray, SparseArray};
    use vortex_array::compute::unary::scalar_at;
    use vortex_array::validity::Validity;
    use vortex_array::IntoArrayData;
    use vortex_buffer::Buffer;
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_scalar::Scalar;

    use crate::BitPackedArray;

    #[test]
    fn invalid_patches() {
        let packed_array = BitPackedArray::try_new(
            Buffer::from(vec![0u8; 128]),
            PType::U32,
            Validity::AllInvalid,
            Some(
                SparseArray::try_new(
                    PrimitiveArray::from(vec![1u64]).into_array(),
                    PrimitiveArray::from_vec(vec![999u32], Validity::AllValid).into_array(),
                    8,
                    Scalar::null_typed::<u32>(),
                )
                .unwrap()
                .into_array(),
            ),
            1,
            8,
        )
        .unwrap()
        .into_array();
        assert_eq!(
            scalar_at(&packed_array, 1).unwrap(),
            Scalar::null(DType::Primitive(PType::U32, Nullability::Nullable))
        );
    }
}
