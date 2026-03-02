// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::patches::Patches;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_array::vtable::ValidityHelper;
use vortex_error::VortexResult;

use crate::bitpacking::BitPackedArray;
use crate::bitpacking::BitPackedVTable;

impl CastReduce for BitPackedVTable {
    fn cast(array: &BitPackedArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if array.dtype().eq_ignore_nullability(dtype) {
            let new_validity = array
                .validity()
                .clone()
                .cast_nullability(dtype.nullability(), array.len())?;
            return Ok(Some(
                BitPackedArray::try_new(
                    array.packed().clone(),
                    dtype.as_ptype(),
                    new_validity,
                    array
                        .patches()
                        .map(|patches| {
                            let new_values = patches.values().cast(dtype.clone())?;
                            Patches::new(
                                patches.array_len(),
                                patches.offset(),
                                patches.indices().clone(),
                                new_values,
                                patches.chunk_offsets().clone(),
                            )
                        })
                        .transpose()?,
                    array.bit_width(),
                    array.len(),
                    array.offset(),
                )?
                .into_array(),
            ));
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_buffer::buffer;

    use crate::BitPackedArray;

    #[test]
    fn test_cast_bitpacked_u8_to_u32() {
        let packed =
            BitPackedArray::encode(&buffer![10u8, 20, 30, 40, 50, 60].into_array(), 6).unwrap();

        let casted = packed
            .to_array()
            .cast(DType::Primitive(PType::U32, Nullability::NonNullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::NonNullable)
        );

        assert_arrays_eq!(
            casted.as_ref(),
            PrimitiveArray::from_iter([10u32, 20, 30, 40, 50, 60])
        );
    }

    #[test]
    fn test_cast_bitpacked_nullable() {
        let values = PrimitiveArray::from_option_iter([Some(5u16), None, Some(10), Some(15), None]);
        let packed = BitPackedArray::encode(&values.to_array(), 4).unwrap();

        let casted = packed
            .to_array()
            .cast(DType::Primitive(PType::U32, Nullability::Nullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::Nullable)
        );
    }

    #[rstest]
    #[case(BitPackedArray::encode(&buffer![0u8, 10, 20, 30, 40, 50, 60, 63].into_array(), 6).unwrap())]
    #[case(BitPackedArray::encode(&buffer![0u16, 100, 200, 300, 400, 500].into_array(), 9).unwrap())]
    #[case(BitPackedArray::encode(&buffer![0u32, 1000, 2000, 3000, 4000].into_array(), 12).unwrap())]
    #[case(BitPackedArray::encode(&PrimitiveArray::from_option_iter([Some(1u32), None, Some(7), Some(15), None]).to_array(), 4).unwrap())]
    fn test_cast_bitpacked_conformance(#[case] array: BitPackedArray) {
        test_cast_conformance(&array.to_array());
    }
}
