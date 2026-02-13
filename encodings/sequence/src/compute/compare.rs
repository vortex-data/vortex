// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::NativePType;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexExpect;
use vortex_scalar::PValue;

/// Find the index where `base + idx * multiplier == intercept`, if one exists.
///
/// Returns `None` if:
/// - `len` is 0
/// - `intercept` is outside the range of the sequence
/// - `intercept` doesn't fall exactly on a sequence value
pub(crate) fn find_intersection_scalar(
    base: PValue,
    multiplier: PValue,
    len: usize,
    intercept: PValue,
) -> Option<usize> {
    match_each_integer_ptype!(base.ptype(), |P| {
        let intercept = intercept.cast::<P>();
        let base = base.cast::<P>();
        let multiplier = multiplier.cast::<P>();
        find_intersection(base, multiplier, len, intercept)
    })
}

fn find_intersection<P: NativePType>(
    base: P,
    multiplier: P,
    len: usize,
    intercept: P,
) -> Option<usize> {
    if len == 0 {
        return None;
    }

    let count = P::from_usize(len - 1).vortex_expect("idx must fit into type");
    let end_element = base + (multiplier * count);

    // Handle ascending vs descending sequences
    let (min_val, max_val) = if multiplier.is_ge(P::zero()) {
        (base, end_element)
    } else {
        (end_element, base)
    };

    // Check if intercept is in range
    if !intercept.is_ge(min_val) || !intercept.is_le(max_val) {
        return None;
    }

    // Handle zero multiplier (constant sequence)
    if multiplier == P::zero() {
        return (intercept == base).then_some(0);
    }

    // Check if (intercept - base) is evenly divisible by multiplier
    let diff = intercept - base;
    if diff % multiplier != P::zero() {
        return None;
    }

    let idx = diff / multiplier;
    idx.to_usize()
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::Operator;
    use vortex_array::compute::compare;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::Nullability::Nullable;

    use crate::SequenceArray;

    #[test]
    fn test_compare_match() {
        let lhs = SequenceArray::typed_new(2i64, 1, NonNullable, 4).unwrap();
        let rhs = ConstantArray::new(4i64, lhs.len());
        let result = compare(lhs.as_ref(), rhs.as_ref(), Operator::Eq).unwrap();
        let expected = BoolArray::from_iter([false, false, true, false]);
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn test_compare_match_scale() {
        let lhs = SequenceArray::typed_new(2i64, 3, Nullable, 4).unwrap();
        let rhs = ConstantArray::new(8i64, lhs.len());
        let result = compare(lhs.as_ref(), rhs.as_ref(), Operator::Eq).unwrap();
        let expected = BoolArray::from_iter([Some(false), Some(false), Some(true), Some(false)]);
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn test_compare_no_match() {
        let lhs = SequenceArray::typed_new(2i64, 1, NonNullable, 4).unwrap();
        let rhs = ConstantArray::new(1i64, lhs.len());
        let result = compare(lhs.as_ref(), rhs.as_ref(), Operator::Eq).unwrap();
        let expected = BoolArray::from_iter([false, false, false, false]);
        assert_arrays_eq!(result, expected);
    }
}
