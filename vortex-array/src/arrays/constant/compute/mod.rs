// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod fill_null;
mod filter;
mod not;
pub(crate) mod rules;
mod slice;
mod take;
pub(crate) mod uncompressed_size;

#[cfg(test)]
mod test {
    use std::f64;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::compute::conformance::consistency::test_array_consistency;
    use crate::compute::conformance::filter::test_filter_conformance;
    use crate::compute::conformance::mask::test_mask_conformance;
    use crate::dtype::half::f16;
    use crate::scalar::Scalar;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn test_mask_constant() {
        test_mask_conformance(
            &ConstantArray::new(Scalar::null_native::<i32>(), 5).into_array(),
            &mut SESSION.create_execution_ctx(),
        );
        test_mask_conformance(
            &ConstantArray::new(Scalar::from(3u16), 5).into_array(),
            &mut SESSION.create_execution_ctx(),
        );
        test_mask_conformance(
            &ConstantArray::new(Scalar::from(1.0f32 / 0.0f32), 5).into_array(),
            &mut SESSION.create_execution_ctx(),
        );
        test_mask_conformance(
            &ConstantArray::new(Scalar::from(f16::from_f32(3.0f32)), 5).into_array(),
            &mut SESSION.create_execution_ctx(),
        );
    }

    #[test]
    fn test_filter_constant() {
        test_filter_conformance(
            &ConstantArray::new(Scalar::null_native::<i32>(), 5).into_array(),
            &mut SESSION.create_execution_ctx(),
        );
        test_filter_conformance(
            &ConstantArray::new(Scalar::from(3u16), 5).into_array(),
            &mut SESSION.create_execution_ctx(),
        );
        test_filter_conformance(
            &ConstantArray::new(Scalar::from(1.0f32 / 0.0f32), 5).into_array(),
            &mut SESSION.create_execution_ctx(),
        );
        test_filter_conformance(
            &ConstantArray::new(Scalar::from(f16::from_f32(3.0f32)), 5).into_array(),
            &mut SESSION.create_execution_ctx(),
        );
    }

    #[rstest]
    // From test_all_consistency
    #[case::constant_i32(ConstantArray::new(Scalar::from(42i32), 5))]
    #[case::constant_str(ConstantArray::new(Scalar::from("constant"), 5))]
    #[case::constant_null(ConstantArray::new(
        Scalar::null(crate::dtype::DType::Primitive(
            crate::dtype::PType::I32,
            crate::dtype::Nullability::Nullable
        )),
        5
    ))]
    // Additional test cases
    #[case::constant_f64(ConstantArray::new(Scalar::from(f64::consts::PI), 10))]
    #[case::constant_bool(ConstantArray::new(Scalar::from(true), 7))]
    #[case::constant_single(ConstantArray::new(Scalar::from(99u64), 1))]
    #[case::constant_large(ConstantArray::new(Scalar::from("hello"), 1000))]
    fn test_constant_consistency(#[case] array: ConstantArray) {
        test_array_consistency(&array.into_array(), &mut SESSION.create_execution_ctx());
    }
}
