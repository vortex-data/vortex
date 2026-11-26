// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::scalar_fn::builtin::mask::Mask as MaskExpr;
use vortex_array::scalar_fn::builtin::mask::MaskReduce;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::DecimalBytePartsArrayExt;
use crate::decimal_byte_parts::DecimalBytePartsArraySlotsExt;

impl MaskReduce for DecimalByteParts {
    fn mask(array: ArrayView<'_, Self>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // Validity lives in the MSP, so only that part needs masking: the lower parts hold
        // undefined bits in null slots, which is exactly what a masked-out row is.
        let masked_msp = MaskExpr::try_new(array.msp().clone(), mask.clone())?.into_array();
        array.with_msp(masked_msp).map(|a| Some(a.into_array()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::mask::test_mask_conformance;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::i256;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use crate::DecimalByteParts;
    use crate::DecimalBytePartsArray;
    use crate::decimal_byte_parts::testing::i128_parts;
    use crate::decimal_byte_parts::testing::i256_parts;
    use crate::decimal_byte_parts::testing::widen_msp;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    fn wide_i128() -> Vec<i128> {
        vec![
            1 << 70,
            -(1 << 70),
            (1 << 64) - 1,
            0,
            99_999_999_999_999_999_999_999_999_999_999_999_999,
        ]
    }

    fn wide_i256() -> Vec<i256> {
        vec![
            i256::from_parts(0, 1),
            i256::from_parts(0, -1),
            i256::from_parts(u128::MAX, 0),
            i256::from_parts(7, 1 << 64),
            i256::from_parts(0, 0),
        ]
    }

    #[rstest]
    #[case::decimal_i32(DecimalByteParts::try_new(
        widen_msp(
            buffer![100i32, 200, 300, 400, 500].into_array(),
            DecimalDType::new(10, 2),
        ).unwrap(),
        DecimalDType::new(10, 2)
    ).unwrap())]
    #[case::decimal_i64(DecimalByteParts::try_new(
        buffer![1000i64, 2000, 3000, 4000, 5000].into_array(),
        DecimalDType::new(19, 4)
    ).unwrap())]
    #[case::decimal_nullable_i32(DecimalByteParts::try_new(
        widen_msp(
            PrimitiveArray::from_option_iter([Some(100i32), None, Some(300), Some(400), None]).into_array(),
            DecimalDType::new(10, 2),
        ).unwrap(),
        DecimalDType::new(10, 2)
    ).unwrap())]
    #[case::decimal_nullable_i64(DecimalByteParts::try_new(
        PrimitiveArray::from_option_iter([Some(1000i64), None, Some(3000), Some(4000), None]).into_array(),
        DecimalDType::new(19, 4)
    ).unwrap())]
    #[case::decimal_single(DecimalByteParts::try_new(
        buffer![42i32].into_array(),
        DecimalDType::new(5, 1)
    ).unwrap())]
    #[case::decimal_large(DecimalByteParts::try_new(
        widen_msp(
            PrimitiveArray::from_iter((0..1500).map(|i| i * 100)).into_array(),
            DecimalDType::new(10, 2),
        ).unwrap(),
        DecimalDType::new(10, 2)
    ).unwrap())]
    #[case::decimal_i128_one_lower_part(i128_parts(wide_i128(), Validity::NonNullable))]
    #[case::decimal_i128_nullable(i128_parts(wide_i128(), Validity::from_iter([true, false, true, true, false])))]
    #[case::decimal_i256_three_lower_parts(i256_parts(wide_i256(), Validity::NonNullable))]
    #[case::decimal_i256_nullable(i256_parts(wide_i256(), Validity::from_iter([false, true, true, false, true])))]
    fn test_mask_decimal_byte_parts_conformance(#[case] array: DecimalBytePartsArray) {
        test_mask_conformance(&array.into_array(), &mut SESSION.create_execution_ctx());
    }
}
