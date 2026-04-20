// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::patches::Patches;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_error::VortexResult;

use crate::bitpacking::BitPacked;
use crate::bitpacking::array::BitPackedArrayExt;
impl CastReduce for BitPacked {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if array.dtype().eq_ignore_nullability(dtype) {
            let new_validity = array
                .validity()?
                .cast_nullability(dtype.nullability(), array.len())?;
            return Ok(Some(
                BitPacked::try_new(
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
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_buffer::buffer;

    use crate::BitPackedArray;
    use crate::BitPackedData;

    fn bp(array: &ArrayRef, bit_width: u8) -> BitPackedArray {
        BitPackedData::encode(array, bit_width, &mut LEGACY_SESSION.create_execution_ctx()).unwrap()
    }

    #[test]
    fn test_cast_bitpacked_u8_to_u32() {
        let packed = bp(&buffer![10u8, 20, 30, 40, 50, 60].into_array(), 6);

        let casted = packed
            .into_array()
            .cast(DType::Primitive(PType::U32, Nullability::NonNullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::NonNullable)
        );

        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_iter([10u32, 20, 30, 40, 50, 60])
        );
    }

    #[test]
    fn test_cast_bitpacked_nullable() {
        let values = PrimitiveArray::from_option_iter([Some(5u16), None, Some(10), Some(15), None]);
        let packed = bp(&values.into_array(), 4);

        let casted = packed
            .into_array()
            .cast(DType::Primitive(PType::U32, Nullability::Nullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::Nullable)
        );
    }

    #[rstest]
    #[case(bp(&buffer![0u8, 10, 20, 30, 40, 50, 60, 63].into_array(), 6))]
    #[case(bp(&buffer![0u16, 100, 200, 300, 400, 500].into_array(), 9))]
    #[case(bp(&buffer![0u32, 1000, 2000, 3000, 4000].into_array(), 12))]
    #[case(bp(&PrimitiveArray::from_option_iter([Some(1u32), None, Some(7), Some(15), None]).into_array(), 4))]
    fn test_cast_bitpacked_conformance(#[case] array: BitPackedArray) {
        test_cast_conformance(&array.into_array());
    }
}
