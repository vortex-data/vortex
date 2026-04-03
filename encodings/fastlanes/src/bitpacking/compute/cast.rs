// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_error::VortexResult;

use crate::bitpacking::BitPacked;
impl CastReduce for BitPacked {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if array.dtype().eq_ignore_nullability(dtype) {
            let new_validity = array
                .validity(array.dtype().nullability())
                .cast_nullability(dtype.nullability(), array.len())?;
            return Ok(Some(
                BitPacked::try_new(
                    array.packed().clone(),
                    dtype.as_ptype(),
                    new_validity,
<<<<<<< HEAD
                    array
                        .patches(array.len())
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
=======
>>>>>>> c2fc4fd43 (add a LazyPatchedArray)
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

    use crate::bitpack_compress::BitPackedEncoder;

    #[test]
    fn test_cast_bitpacked_u8_to_u32() {
        let parray = PrimitiveArray::from_iter([10u8, 20, 30, 40, 50, 60]);

        let packed = BitPackedEncoder::new(&parray)
            .with_bit_width(6)
            .pack()
            .unwrap()
            .unwrap_unpatched();

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
        let packed = BitPackedEncoder::new(&values)
            .with_bit_width(4)
            .pack()
            .unwrap()
            .unwrap_unpatched();

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
    #[case(PrimitiveArray::from_iter([0u8, 10, 20, 30, 40, 50, 60, 63]), 6)]
    #[case(PrimitiveArray::from_iter([0u16, 100, 200, 300, 400, 500]), 9)]
    #[case(PrimitiveArray::from_iter([0u32, 1000, 2000, 3000, 4000]), 12)]
    #[case(PrimitiveArray::from_option_iter([Some(1u32), None, Some(7), Some(15), None]), 4)]
    fn test_cast_bitpacked_conformance(#[case] parray: PrimitiveArray, #[case] bw: u8) {
        let array = BitPackedEncoder::new(&parray)
            .with_bit_width(bw)
            .pack()
            .unwrap()
            .into_array()
            .unwrap();
        test_cast_conformance(&array);
    }
}
