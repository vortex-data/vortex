// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_buffer::Buffer;

    use crate::PcoArray;

    fn pco_f32() -> PcoArray {
        let values = PrimitiveArray::new(
            Buffer::copy_from(vec![1.23f32, 4.56, 7.89, 10.11, 12.13]),
            vortex_array::validity::Validity::NonNullable,
        );
        PcoArray::from_primitive(&values, 0, 128).unwrap()
    }

    fn pco_f64() -> PcoArray {
        let values = PrimitiveArray::new(
            Buffer::copy_from(vec![100.1f64, 200.2, 300.3, 400.4, 500.5]),
            vortex_array::validity::Validity::NonNullable,
        );
        PcoArray::from_primitive(&values, 0, 128).unwrap()
    }

    fn pco_i32() -> PcoArray {
        let values = PrimitiveArray::new(
            Buffer::copy_from(vec![100i32, 200, 300, 400, 500]),
            vortex_array::validity::Validity::NonNullable,
        );
        PcoArray::from_primitive(&values, 0, 128).unwrap()
    }

    fn pco_u64() -> PcoArray {
        let values = PrimitiveArray::new(
            Buffer::copy_from(vec![1000u64, 2000, 3000, 4000]),
            vortex_array::validity::Validity::NonNullable,
        );
        PcoArray::from_primitive(&values, 0, 128).unwrap()
    }

    fn pco_i16() -> PcoArray {
        let values = PrimitiveArray::new(
            Buffer::copy_from(vec![10i16, 20, 30, 40, 50]),
            vortex_array::validity::Validity::NonNullable,
        );
        PcoArray::from_primitive(&values, 0, 128).unwrap()
    }

    fn pco_i32_alt() -> PcoArray {
        let values = PrimitiveArray::new(
            Buffer::copy_from(vec![1i32, 2, 3, 4, 5]),
            vortex_array::validity::Validity::NonNullable,
        );
        PcoArray::from_primitive(&values, 0, 128).unwrap()
    }

    fn pco_single() -> PcoArray {
        let values = PrimitiveArray::new(
            Buffer::copy_from(vec![42.42f64]),
            vortex_array::validity::Validity::NonNullable,
        );
        PcoArray::from_primitive(&values, 0, 128).unwrap()
    }

    fn pco_large() -> PcoArray {
        let values = PrimitiveArray::new(
            Buffer::copy_from((0..1000).map(|i| i as u32).collect::<Vec<_>>()),
            vortex_array::validity::Validity::NonNullable,
        );
        PcoArray::from_primitive(&values, 3, 128).unwrap()
    }

    #[rstest]
    #[case::f32(pco_f32())]
    #[case::f64(pco_f64())]
    #[case::i32(pco_i32())]
    #[case::u64(pco_u64())]
    #[case::i16(pco_i16())]
    #[case::i32_alt(pco_i32_alt())]
    #[case::single(pco_single())]
    #[case::large(pco_large())]
    fn test_pco_consistency(#[case] array: PcoArray) {
        test_array_consistency(&array.to_array());
    }
}
