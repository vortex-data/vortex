// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_error::VortexResult;

use crate::rle::RLE;
use crate::rle::RLEArrayExt;
impl CastReduce for RLE {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Cast RLE values.
        let casted_values = array
            .values()
            .cast(DType::Primitive(dtype.as_ptype(), Nullability::NonNullable))?;

        // Cast RLE indices such that validity matches the target dtype.
        let casted_indices = if array.indices().dtype().nullability() != dtype.nullability() {
            array.indices().cast(DType::Primitive(
                array.indices().dtype().as_ptype(),
                dtype.nullability(),
            ))?
        } else {
            array.indices().clone()
        };

        Ok(Some(
            RLE::try_new(
                casted_values,
                casted_indices,
                array.values_idx_offsets().clone(),
                array.offset(),
                array.len(),
            )?
            .into_array(),
        ))
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
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;

    use crate::RLEData;
    use crate::rle::RLEArray;

    fn rle(primitive: &PrimitiveArray) -> RLEArray {
        RLEData::encode(primitive.as_view()).unwrap()
    }

    #[test]
    fn try_cast_rle_success() {
        let primitive = PrimitiveArray::new(
            Buffer::from_iter([10u8, 20, 30, 40, 50]),
            Validity::from_iter([true, true, true, true, true]),
        );
        let encoded = rle(&primitive);

        let casted = encoded
            .into_array()
            .cast(DType::Primitive(PType::U16, Nullability::NonNullable))
            .unwrap();
        assert_arrays_eq!(casted, PrimitiveArray::from_iter([10u16, 20, 30, 40, 50]));
    }

    #[test]
    #[should_panic]
    fn try_cast_rle_fail() {
        let primitive = PrimitiveArray::new(
            Buffer::from_iter([10u8, 20, 30, 40, 50]),
            Validity::from_iter([true, false, true, true, false]),
        );
        let encoded = rle(&primitive);
        encoded
            .into_array()
            .cast(DType::Primitive(PType::U8, Nullability::NonNullable))
            .and_then(|a| a.to_canonical().map(|c| c.into_array()))
            .unwrap();
    }

    #[rstest]
    #[case::u8(
        PrimitiveArray::new(
            Buffer::from_iter([0u8, 10, 20, 30, 40, 50]),
            Validity::NonNullable,
        )
    )]
    #[case::u8_nullable(
        PrimitiveArray::new(
            Buffer::from_iter([0u8, 10, 20, 30, 40]),
            Validity::from_iter([true, false, true, false, true]),
        )
    )]
    #[case::u16(
        PrimitiveArray::new(
            Buffer::from_iter([0u16, 100, 200, 300, 400, 500]),
            Validity::NonNullable,
        )
    )]
    #[case::u16_nullable(
        PrimitiveArray::new(
            Buffer::from_iter([0u16, 100, 200, 300, 400]),
            Validity::from_iter([false, true, false, true, true]),
        )
    )]
    #[case::u32(
        PrimitiveArray::new(
            Buffer::from_iter([0u32, 1000, 2000, 3000, 4000]),
            Validity::NonNullable,
        )
    )]
    #[case::u32_nullable(
        PrimitiveArray::new(
            Buffer::from_iter([0u32, 1000, 2000, 3000, 4000]),
            Validity::from_iter([true, true, false, false, true]),
        )
    )]
    #[case::u64(
        PrimitiveArray::new(
            Buffer::from_iter([0u64, 10000, 20000, 30000]),
            Validity::NonNullable,
        )
    )]
    #[case::u64_nullable(
        PrimitiveArray::new(
            Buffer::from_iter([0u64, 10000, 20000, 30000]),
            Validity::from_iter([false, false, true, true]),
        )
    )]
    fn test_cast_rle_conformance(#[case] primitive: PrimitiveArray) {
        let rle_array = rle(&primitive);
        test_cast_conformance(&rle_array.into_array());
    }
}
