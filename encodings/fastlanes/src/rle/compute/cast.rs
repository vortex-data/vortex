// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{CastKernel, CastKernelAdapter, cast};
use vortex_array::{ArrayRef, register_kernel};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::rle::{RLEArray, RLEVTable};

impl CastKernel for RLEVTable {
    fn cast(&self, array: &RLEArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Cast RLE values.
        let casted_values = cast(array.values(), dtype)?;

        // Cast RLE indices such that validity matches the target dtype.
        let casted_indices = if array.indices().dtype().nullability() != dtype.nullability() {
            cast(
                array.indices(),
                &DType::Primitive(array.indices().dtype().as_ptype(), dtype.nullability()),
            )?
        } else {
            array.indices().clone()
        };

        Ok(Some(unsafe {
            RLEArray::new_unchecked(
                casted_values,
                casted_indices,
                array.values_idx_offsets().clone(),
                dtype.clone(),
                array.offset(),
                array.len(),
            )
            .into()
        }))
    }
}

register_kernel!(CastKernelAdapter(RLEVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::cast;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::rle::RLEArray;

    #[test]
    fn try_cast_rle_success() {
        let primitive = PrimitiveArray::new(
            Buffer::from_iter([10u8, 20, 30, 40, 50]),
            Validity::from_iter([true, true, true, true, true]),
        );
        let rle = RLEArray::encode(&primitive).unwrap();

        let res = cast(
            rle.as_ref(),
            &DType::Primitive(PType::U16, Nullability::NonNullable),
        );
        assert!(res.is_ok());
        assert_eq!(
            res.unwrap().dtype(),
            &DType::Primitive(PType::U16, Nullability::NonNullable)
        );
    }

    #[test]
    #[should_panic]
    fn try_cast_rle_fail() {
        let primitive = PrimitiveArray::new(
            Buffer::from_iter([10u8, 20, 30, 40, 50]),
            Validity::from_iter([true, false, true, true, false]),
        );
        let rle = RLEArray::encode(&primitive).unwrap();
        cast(
            rle.as_ref(),
            &DType::Primitive(PType::U8, Nullability::NonNullable),
        )
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
        let rle_array = RLEArray::encode(&primitive).unwrap();
        test_cast_conformance(rle_array.as_ref());
    }
}
