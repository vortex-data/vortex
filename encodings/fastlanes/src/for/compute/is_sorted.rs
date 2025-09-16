// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{IsSortedKernel, IsSortedKernelAdapter, is_sorted, is_strict_sorted};
use vortex_array::{ToCanonical, register_kernel};
use vortex_error::VortexResult;

use crate::{FoRArray, FoRVTable};

/// FoR can express sortedness directly on its encoded form.
///
/// If the minimum is greater than or equal to zero, subtracting it from the other values does not
/// wrap (the value always decreases and the smallest value is zero because min - min = 0).
///
/// Subtraction without wrapping is order-preserving, so we only need to consider what happens to
/// wrapped numbers.
///
/// Non-negative minimum values can't wrap. For a negative minimum value, wrapping means that
///
/// ```text
/// a + abs(min) > 127
/// ```
///
/// There's some residue r,
///
/// ```text
/// r < 128
/// ```
///
/// such that
///
/// ```text
/// a + abs(min) mod 128 = r
/// ```
///
/// For example,
///
/// ```text
/// min = -128
/// a = 1
///
/// 1 - -128 = 129
/// ```
///
/// And 129's residue is 1. 129 is represented as
///
/// ```text
/// -128 + 1 = -127
/// ```
///
/// The unsigned representation is
///
/// ```text
/// 2^8 - 127
/// ```
///
/// More directly, for some residue r:
///
/// ```text
/// 2^8 + (-128 + r)
///   = 2^8 - 128 + r
///   = 128 + r
/// ```
///
/// Addition is order-preserving, so all the wrapped values preserve their order and they're all
/// represented as unsigned values larger than 127 so they also preserve their order with the
/// unwrapped values.
impl IsSortedKernel for FoRVTable {
    fn is_sorted(&self, array: &FoRArray) -> VortexResult<Option<bool>> {
        let encoded = array.encoded().to_primitive();
        is_sorted(
            &encoded
                .reinterpret_cast(encoded.ptype().to_unsigned())
                .to_array(),
        )
    }

    fn is_strict_sorted(&self, array: &FoRArray) -> VortexResult<Option<bool>> {
        let encoded = array.encoded().to_primitive();
        is_strict_sorted(
            &encoded
                .reinterpret_cast(encoded.ptype().to_unsigned())
                .to_array(),
        )
    }
}

register_kernel!(IsSortedKernelAdapter(FoRVTable).lift());

#[cfg(test)]
mod test {
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::is_sorted;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use crate::FoRArray;

    #[test]
    fn test_sorted() {
        let a = PrimitiveArray::new(buffer![-1, 0, i8::MAX], Validity::NonNullable);
        let b = FoRArray::encode(a).unwrap();
        assert!(
            is_sorted(&b.clone().into_array()).unwrap().unwrap(),
            "{}",
            b.encoded().display_values()
        );

        let a = PrimitiveArray::new(buffer![i8::MIN, 0, i8::MAX], Validity::NonNullable);
        let b = FoRArray::encode(a).unwrap();
        assert!(
            is_sorted(&b.clone().into_array()).unwrap().unwrap(),
            "{}",
            b.encoded().display_values()
        );

        let a = PrimitiveArray::new(buffer![i8::MIN, 0, 30, 127], Validity::NonNullable);
        let b = FoRArray::encode(a).unwrap();
        assert!(
            is_sorted(&b.clone().into_array()).unwrap().unwrap(),
            "{}",
            b.encoded().display_values()
        );

        let a = PrimitiveArray::new(buffer![i8::MIN, -3, -1], Validity::NonNullable);
        let b = FoRArray::encode(a).unwrap();
        assert!(
            is_sorted(&b.clone().into_array()).unwrap().unwrap(),
            "{}",
            b.encoded().display_values()
        );

        let a = PrimitiveArray::new(buffer![-10, -3, -1], Validity::NonNullable);
        let b = FoRArray::encode(a).unwrap();
        assert!(
            is_sorted(&b.clone().into_array()).unwrap().unwrap(),
            "{}",
            b.encoded().display_values()
        );

        let a = PrimitiveArray::new(buffer![-10, -11, -1], Validity::NonNullable);
        let b = FoRArray::encode(a).unwrap();
        assert!(
            !is_sorted(&b.clone().into_array()).unwrap().unwrap(),
            "{}",
            b.encoded().display_values()
        );

        let a = PrimitiveArray::new(buffer![-10, i8::MIN, -1], Validity::NonNullable);
        let b = FoRArray::encode(a).unwrap();
        assert!(
            !is_sorted(&b.clone().into_array()).unwrap().unwrap(),
            "{}",
            b.encoded().display_values()
        );
    }
}
