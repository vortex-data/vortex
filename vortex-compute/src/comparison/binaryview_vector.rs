// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare implementations for BinaryViewVector.

use std::ops::BitAnd;

use vortex_vector::VectorOps;
use vortex_vector::binaryview::BinaryViewType;
use vortex_vector::binaryview::BinaryViewVector;
use vortex_vector::bool::BoolVector;

use crate::comparison::Compare;
use crate::comparison::Equal;
use crate::comparison::GreaterThan;
use crate::comparison::GreaterThanOrEqual;
use crate::comparison::LessThan;
use crate::comparison::LessThanOrEqual;
use crate::comparison::NotEqual;

/// Compare two BinaryViewVectors element-wise using the provided comparison function.
///
/// Only accesses view data for positions that are valid in both vectors.
fn compare_binaryview<T: BinaryViewType, F>(
    lhs: &BinaryViewVector<T>,
    rhs: &BinaryViewVector<T>,
    cmp: F,
) -> BoolVector
where
    F: Fn(&[u8], &[u8]) -> bool,
{
    let validity = lhs.validity().bitand(rhs.validity());
    let validity_bits = validity.to_bit_buffer();

    let bits = validity_bits.map_cmp(|i, valid| {
        if valid {
            // SAFETY: map_cmp provides validity bit, only access data when valid
            let l = unsafe { lhs.get_ref_unchecked(i) };
            let r = unsafe { rhs.get_ref_unchecked(i) };
            cmp(l, r)
        } else {
            false
        }
    });

    BoolVector::new(bits, validity)
}

impl<T: BinaryViewType> Compare<Equal> for &BinaryViewVector<T> {
    type Output = BoolVector;

    fn compare(self, rhs: Self) -> Self::Output {
        compare_binaryview(self, rhs, |l, r| l == r)
    }
}

impl<T: BinaryViewType> Compare<NotEqual> for &BinaryViewVector<T> {
    type Output = BoolVector;

    fn compare(self, rhs: Self) -> Self::Output {
        compare_binaryview(self, rhs, |l, r| l != r)
    }
}

impl<T: BinaryViewType> Compare<LessThan> for &BinaryViewVector<T> {
    type Output = BoolVector;

    fn compare(self, rhs: Self) -> Self::Output {
        compare_binaryview(self, rhs, |l, r| l < r)
    }
}

impl<T: BinaryViewType> Compare<LessThanOrEqual> for &BinaryViewVector<T> {
    type Output = BoolVector;

    fn compare(self, rhs: Self) -> Self::Output {
        compare_binaryview(self, rhs, |l, r| l <= r)
    }
}

impl<T: BinaryViewType> Compare<GreaterThan> for &BinaryViewVector<T> {
    type Output = BoolVector;

    fn compare(self, rhs: Self) -> Self::Output {
        compare_binaryview(self, rhs, |l, r| l > r)
    }
}

impl<T: BinaryViewType> Compare<GreaterThanOrEqual> for &BinaryViewVector<T> {
    type Output = BoolVector;

    fn compare(self, rhs: Self) -> Self::Output {
        compare_binaryview(self, rhs, |l, r| l >= r)
    }
}

impl<T: BinaryViewType> Compare<Equal> for BinaryViewVector<T> {
    type Output = BoolVector;

    fn compare(self, rhs: Self) -> Self::Output {
        Compare::<Equal>::compare(&self, &rhs)
    }
}

impl<T: BinaryViewType> Compare<NotEqual> for BinaryViewVector<T> {
    type Output = BoolVector;

    fn compare(self, rhs: Self) -> Self::Output {
        Compare::<NotEqual>::compare(&self, &rhs)
    }
}

impl<T: BinaryViewType> Compare<LessThan> for BinaryViewVector<T> {
    type Output = BoolVector;

    fn compare(self, rhs: Self) -> Self::Output {
        Compare::<LessThan>::compare(&self, &rhs)
    }
}

impl<T: BinaryViewType> Compare<LessThanOrEqual> for BinaryViewVector<T> {
    type Output = BoolVector;

    fn compare(self, rhs: Self) -> Self::Output {
        Compare::<LessThanOrEqual>::compare(&self, &rhs)
    }
}

impl<T: BinaryViewType> Compare<GreaterThan> for BinaryViewVector<T> {
    type Output = BoolVector;

    fn compare(self, rhs: Self) -> Self::Output {
        Compare::<GreaterThan>::compare(&self, &rhs)
    }
}

impl<T: BinaryViewType> Compare<GreaterThanOrEqual> for BinaryViewVector<T> {
    type Output = BoolVector;

    fn compare(self, rhs: Self) -> Self::Output {
        Compare::<GreaterThanOrEqual>::compare(&self, &rhs)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::bitbuffer;
    use vortex_mask::Mask;
    use vortex_vector::VectorMutOps;
    use vortex_vector::binaryview::BinaryViewVectorMut;
    use vortex_vector::binaryview::StringType;

    use super::*;

    fn make_string_vector(values: &[&str]) -> BinaryViewVector<StringType> {
        let mut builder = BinaryViewVectorMut::<StringType>::with_capacity(values.len());
        for v in values {
            builder.append_values(*v, 1);
        }
        builder.freeze()
    }

    #[test]
    fn test_string_vector_equal() {
        let left = make_string_vector(&["apple", "banana", "cherry"]);
        let right = make_string_vector(&["apple", "orange", "cherry"]);

        let result = Compare::<Equal>::compare(&left, &right);
        let expected = BoolVector::new(bitbuffer![1 0 1], Mask::new_true(3));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_string_vector_less_than() {
        let left = make_string_vector(&["apple", "banana", "cherry"]);
        let right = make_string_vector(&["banana", "banana", "apple"]);

        let result = Compare::<LessThan>::compare(&left, &right);
        // "apple" < "banana" = true, "banana" < "banana" = false, "cherry" < "apple" = false
        let expected = BoolVector::new(bitbuffer![1 0 0], Mask::new_true(3));
        assert_eq!(result, expected);
    }
}
