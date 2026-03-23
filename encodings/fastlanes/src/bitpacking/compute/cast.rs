// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_array::vtable::ValidityHelper;
use vortex_error::VortexResult;

use crate::bitpacking::BitPacked;
use crate::bitpacking::BitPackedArray;

impl CastReduce for BitPacked {
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
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_buffer::buffer;

    use crate::bitpack_compress::BitPackEncoder;

    #[test]
    fn test_cast_bitpacked_u8_to_u32() {
        let packed = BitPackEncoder::new(
            &buffer![10u8, 20, 30, 40, 50, 60]
                .into_array()
                .to_primitive(),
        )
        .with_bit_width(6)
        .pack()
        .unwrap()
        .into_packed();

        let casted = packed
            .into_array()
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
        let packed = BitPackEncoder::new(&values)
            .with_bit_width(4)
            .pack()
            .unwrap()
            .into_packed();

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
    #[case(BitPackEncoder::new(&buffer![0u8, 10, 20, 30, 40, 50, 60, 63].into_array().to_primitive()).with_bit_width(6).pack().unwrap().into_packed().into_array())]
    #[case(BitPackEncoder::new(&buffer![0u16, 100, 200, 300, 400, 500].into_array().to_primitive()).with_bit_width(9).pack().unwrap().into_packed().into_array())]
    #[case(BitPackEncoder::new(&buffer![0u32, 1000, 2000, 3000, 4000].into_array().to_primitive()).with_bit_width(12).pack().unwrap().into_packed().into_array())]
    #[case(BitPackEncoder::new(&PrimitiveArray::from_option_iter([Some(1u32), None, Some(7), Some(15), None])).with_bit_width(4).pack().unwrap().into_packed().into_array())]
    fn test_cast_bitpacked_conformance(#[case] array: ArrayRef) {
        test_cast_conformance(&array.into_array());
    }
}
