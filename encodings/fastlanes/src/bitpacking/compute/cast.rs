// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{CastKernel, CastKernelAdapter, cast};
use vortex_array::patches::Patches;
use vortex_array::vtable::ValidityHelper;
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::bitpacking::{BitPackedArray, BitPackedVTable};

impl CastKernel for BitPackedVTable {
    fn cast(&self, array: &BitPackedArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if array.dtype().eq_ignore_nullability(dtype) {
            let new_validity = array
                .validity()
                .clone()
                .cast_nullability(dtype.nullability())?;
            return Ok(Some(
                BitPackedArray::try_new(
                    array.packed().clone(),
                    dtype.as_ptype(),
                    new_validity,
                    array
                        .patches()
                        .map(|patches| {
                            let new_values = cast(patches.values(), dtype)?;
                            VortexResult::Ok(Patches::new(
                                patches.array_len(),
                                patches.offset(),
                                patches.indices().clone(),
                                new_values,
                            ))
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

register_kernel!(CastKernelAdapter(BitPackedVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::cast;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::BitPackedArray;

    #[test]
    fn test_cast_bitpacked_u8_to_u32() {
        let packed =
            BitPackedArray::encode(buffer![10u8, 20, 30, 40, 50, 60].into_array().as_ref(), 6)
                .unwrap();

        let casted = cast(
            packed.as_ref(),
            &DType::Primitive(PType::U32, Nullability::NonNullable),
        )
        .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::NonNullable)
        );

        let decoded = casted.to_primitive();
        assert_eq!(decoded.as_slice::<u32>(), &[10u32, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn test_cast_bitpacked_nullable() {
        let values = PrimitiveArray::from_option_iter([Some(5u16), None, Some(10), Some(15), None]);
        let packed = BitPackedArray::encode(values.as_ref(), 4).unwrap();

        let casted = cast(
            packed.as_ref(),
            &DType::Primitive(PType::U32, Nullability::Nullable),
        )
        .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::Nullable)
        );
    }

    #[rstest]
    #[case(BitPackedArray::encode(buffer![0u8, 10, 20, 30, 40, 50, 60, 63].into_array().as_ref(), 6).unwrap())]
    #[case(BitPackedArray::encode(buffer![0u16, 100, 200, 300, 400, 500].into_array().as_ref(), 9).unwrap())]
    #[case(BitPackedArray::encode(buffer![0u32, 1000, 2000, 3000, 4000].into_array().as_ref(), 12).unwrap())]
    #[case(BitPackedArray::encode(PrimitiveArray::from_option_iter([Some(1u32), None, Some(7), Some(15), None]).as_ref(), 4).unwrap())]
    fn test_cast_bitpacked_conformance(#[case] array: BitPackedArray) {
        test_cast_conformance(array.as_ref());
    }
}
