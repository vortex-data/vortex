// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare implementations for BinaryViewScalar.

use vortex_vector::binaryview::BinaryViewScalar;
use vortex_vector::binaryview::BinaryViewType;
use vortex_vector::bool::BoolScalar;

use crate::comparison::Compare;
use crate::comparison::Equal;
use crate::comparison::GreaterThan;
use crate::comparison::GreaterThanOrEqual;
use crate::comparison::LessThan;
use crate::comparison::LessThanOrEqual;
use crate::comparison::NotEqual;

/// Compare two BinaryViewScalars using the provided comparison function.
fn compare_binaryview_scalar<T: BinaryViewType, F>(
    lhs: BinaryViewScalar<T>,
    rhs: BinaryViewScalar<T>,
    cmp: F,
) -> BoolScalar
where
    F: Fn(&[u8], &[u8]) -> bool,
{
    match (lhs.value(), rhs.value()) {
        (Some(l), Some(r)) => {
            let l_bytes: &[u8] = AsRef::<T::Slice>::as_ref(l).as_ref();
            let r_bytes: &[u8] = AsRef::<T::Slice>::as_ref(r).as_ref();
            BoolScalar::new(Some(cmp(l_bytes, r_bytes)))
        }
        _ => BoolScalar::new(None),
    }
}

impl<T: BinaryViewType> Compare<Equal> for BinaryViewScalar<T> {
    type Output = BoolScalar;

    fn compare(self, rhs: Self) -> Self::Output {
        compare_binaryview_scalar(self, rhs, |l, r| l == r)
    }
}

impl<T: BinaryViewType> Compare<NotEqual> for BinaryViewScalar<T> {
    type Output = BoolScalar;

    fn compare(self, rhs: Self) -> Self::Output {
        compare_binaryview_scalar(self, rhs, |l, r| l != r)
    }
}

impl<T: BinaryViewType> Compare<LessThan> for BinaryViewScalar<T> {
    type Output = BoolScalar;

    fn compare(self, rhs: Self) -> Self::Output {
        compare_binaryview_scalar(self, rhs, |l, r| l < r)
    }
}

impl<T: BinaryViewType> Compare<LessThanOrEqual> for BinaryViewScalar<T> {
    type Output = BoolScalar;

    fn compare(self, rhs: Self) -> Self::Output {
        compare_binaryview_scalar(self, rhs, |l, r| l <= r)
    }
}

impl<T: BinaryViewType> Compare<GreaterThan> for BinaryViewScalar<T> {
    type Output = BoolScalar;

    fn compare(self, rhs: Self) -> Self::Output {
        compare_binaryview_scalar(self, rhs, |l, r| l > r)
    }
}

impl<T: BinaryViewType> Compare<GreaterThanOrEqual> for BinaryViewScalar<T> {
    type Output = BoolScalar;

    fn compare(self, rhs: Self) -> Self::Output {
        compare_binaryview_scalar(self, rhs, |l, r| l >= r)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BufferString;
    use vortex_vector::binaryview::StringType;

    use super::*;

    #[test]
    fn test_string_scalar_equal() {
        let left = BinaryViewScalar::<StringType>::new(Some(BufferString::from("hello")));
        let right = BinaryViewScalar::<StringType>::new(Some(BufferString::from("hello")));

        assert_eq!(Compare::<Equal>::compare(left, right).value(), Some(true));
    }

    #[test]
    fn test_string_scalar_not_equal() {
        let left = BinaryViewScalar::<StringType>::new(Some(BufferString::from("hello")));
        let right = BinaryViewScalar::<StringType>::new(Some(BufferString::from("world")));

        assert_eq!(
            Compare::<NotEqual>::compare(left, right).value(),
            Some(true)
        );
    }

    #[test]
    fn test_string_scalar_less_than() {
        let left = BinaryViewScalar::<StringType>::new(Some(BufferString::from("apple")));
        let right = BinaryViewScalar::<StringType>::new(Some(BufferString::from("banana")));

        assert_eq!(
            Compare::<LessThan>::compare(left, right).value(),
            Some(true)
        );
    }

    #[test]
    fn test_string_scalar_with_null() {
        let left = BinaryViewScalar::<StringType>::new(Some(BufferString::from("hello")));
        let right = BinaryViewScalar::<StringType>::new(None);

        assert_eq!(Compare::<Equal>::compare(left, right).value(), None);
    }
}
