// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_mask::MaskValues;

use crate::arrays::PrimitiveArray;
use crate::arrays::filter::execute::buffer;
use crate::arrays::filter::execute::byte_compress;
use crate::arrays::filter::execute::filter_validity;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::validity::Validity;

pub fn filter_primitive(array: &PrimitiveArray, mask: &Arc<MaskValues>) -> PrimitiveArray {
    let validity = array
        .validity()
        .vortex_expect("primitive validity should be derivable");
    let filtered_validity = filter_validity(validity, mask);

    match array.ptype() {
        // Byte-compress avoids materializing indices/slices and processes 8 elements per mask byte.
        PType::U8 => filter_byte_compress::<u8>(array, filtered_validity, mask),
        PType::I8 => filter_byte_compress::<i8>(array, filtered_validity, mask),
        PType::U16 => filter_byte_compress::<u16>(array, filtered_validity, mask),
        PType::I16 => filter_byte_compress::<i16>(array, filtered_validity, mask),
        PType::U32 => filter_byte_compress::<u32>(array, filtered_validity, mask),
        PType::I32 => filter_byte_compress::<i32>(array, filtered_validity, mask),
        _ => match_each_native_ptype!(array.ptype(), |T| {
            let filtered_buffer = buffer::filter_buffer(array.to_buffer::<T>(), mask.as_ref());

            // SAFETY: We filter both the validity and the buffer with the same mask, so they must
            // have the same length.
            unsafe { PrimitiveArray::new_unchecked(filtered_buffer, filtered_validity) }
        }),
    }
}

fn filter_byte_compress<T: NativePType>(
    array: &PrimitiveArray,
    filtered_validity: Validity,
    mask: &Arc<MaskValues>,
) -> PrimitiveArray {
    let filtered_buffer = byte_compress::filter_buffer(array.to_buffer::<T>(), mask.as_ref());

    // SAFETY: We filter both the validity and the buffer with the same mask, so they must have the
    // same length.
    unsafe { PrimitiveArray::new_unchecked(filtered_buffer, filtered_validity) }
}

#[cfg(test)]
#[expect(clippy::cast_possible_truncation)]
mod test {
    use itertools::Itertools;
    use rstest::rstest;
    use vortex_mask::Mask;

    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    #[expect(deprecated)]
    use crate::canonical::ToCanonical as _;
    use crate::compute::conformance::filter::LARGE_SIZE;
    use crate::compute::conformance::filter::MEDIUM_SIZE;
    use crate::compute::conformance::filter::test_filter_conformance;

    #[test]
    fn filter_run_variant_mixed_test() {
        let mask = [true, true, false, true, true, true, false, true];
        let arr = PrimitiveArray::from_iter([1u32, 24, 54, 2, 3, 2, 3, 2]);

        #[expect(deprecated)]
        let filtered = arr.filter(Mask::from_iter(mask)).unwrap().to_primitive();
        assert_eq!(
            filtered.len(),
            mask.iter().filter(|x| **x).collect_vec().len()
        );

        let rust_arr = arr.as_slice::<u32>();
        assert_eq!(
            filtered.as_slice::<u32>().to_vec(),
            mask.iter()
                .enumerate()
                .filter(|(_idx, b)| **b)
                .map(|m| rust_arr[m.0])
                .collect_vec()
        )
    }

    #[rstest]
    #[case(PrimitiveArray::from_iter([-2i8, -1, 0, 1, 2]))]
    #[case(PrimitiveArray::from_iter([1u16, 2, 3, 4, 5]))]
    #[case(PrimitiveArray::from_iter([-2i16, -1, 0, 1, 2]))]
    #[case(PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]))]
    #[case(PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4), None]))]
    #[case(PrimitiveArray::from_iter([42u64]))]
    #[case(PrimitiveArray::from_iter(0..MEDIUM_SIZE as i32))]
    #[case(PrimitiveArray::from_option_iter(
        (0..MEDIUM_SIZE).map(|i| if i % 3 == 0 { None } else { Some(i as i64) }))
    )]
    #[case(PrimitiveArray::from_iter(0..LARGE_SIZE as u32))]
    #[case(PrimitiveArray::from_iter([0.1f32, 0.2, 0.3, 0.4, 0.5]))]
    #[case(PrimitiveArray::from_option_iter([Some(1.1f64), None, Some(2.2), Some(3.3), None]))]
    fn test_filter_primitive_conformance(#[case] array: PrimitiveArray) {
        test_filter_conformance(&array.into_array());
    }
}
