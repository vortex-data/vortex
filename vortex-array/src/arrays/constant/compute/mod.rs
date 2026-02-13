// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod binary_numeric;
mod cast;
mod compare;
mod fill_null;
mod filter;
mod mask;
mod min_max;
mod not;
pub(crate) mod rules;
mod slice;
mod sum;
mod take;

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_dtype::half::f16;
    use vortex_scalar::Scalar;

    use crate::IntoArray;
    use crate::arrays::ConstantArray;
    use crate::compute::conformance::consistency::test_array_consistency;
    use crate::compute::conformance::filter::test_filter_conformance;
    use crate::compute::conformance::mask::test_mask_conformance;

    #[test]
    fn test_mask_constant() {
        test_mask_conformance(&ConstantArray::new(Scalar::null_native::<i32>(), 5).into_array());
        test_mask_conformance(&ConstantArray::new(Scalar::from(3u16), 5).into_array());
        test_mask_conformance(&ConstantArray::new(Scalar::from(1.0f32 / 0.0f32), 5).into_array());
        test_mask_conformance(
            &ConstantArray::new(Scalar::from(f16::from_f32(3.0f32)), 5).into_array(),
        );
    }

    #[test]
    fn test_filter_constant() {
        test_filter_conformance(&ConstantArray::new(Scalar::null_native::<i32>(), 5).into_array());
        test_filter_conformance(&ConstantArray::new(Scalar::from(3u16), 5).into_array());
        test_filter_conformance(&ConstantArray::new(Scalar::from(1.0f32 / 0.0f32), 5).into_array());
        test_filter_conformance(
            &ConstantArray::new(Scalar::from(f16::from_f32(3.0f32)), 5).into_array(),
        );
    }

    #[rstest]
    // From test_all_consistency
    #[case::constant_i32(ConstantArray::new(Scalar::from(42i32), 5))]
    #[case::constant_str(ConstantArray::new(Scalar::from("constant"), 5))]
    #[case::constant_null(ConstantArray::new(
        Scalar::null(vortex_dtype::DType::Primitive(
            vortex_dtype::PType::I32,
            vortex_dtype::Nullability::Nullable
        )),
        5
    ))]
    // Additional test cases
    #[case::constant_f64(ConstantArray::new(Scalar::from(std::f64::consts::PI), 10))]
    #[case::constant_bool(ConstantArray::new(Scalar::from(true), 7))]
    #[case::constant_single(ConstantArray::new(Scalar::from(99u64), 1))]
    #[case::constant_large(ConstantArray::new(Scalar::from("hello"), 1000))]
    fn test_constant_consistency(#[case] array: ConstantArray) {
        test_array_consistency(array.as_ref());
    }
}
