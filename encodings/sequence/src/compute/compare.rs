// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::compute::Operator;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::CompareKernel;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::SequenceArray;
use crate::array::SequenceVTable;

impl CompareKernel for SequenceVTable {
    fn compare(
        lhs: &SequenceArray,
        rhs: &dyn Array,
        operator: Operator,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // TODO(joe): support other operators (NotEq, Lt, Lte, Gt, Gte) in encoded space.
        if operator != Operator::Eq {
            return Ok(None);
        }

        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };

        // Check if there exists an integer solution to const = base + (0..len) * multiplier.
        let set_idx = find_intersection_scalar(
            lhs.base(),
            lhs.multiplier(),
            lhs.len(),
            constant
                .as_primitive()
                .pvalue()
                .vortex_expect("null constant handled in adaptor"),
        );

        let nullability = lhs.dtype().nullability() | rhs.dtype().nullability();
        let validity = match nullability {
            Nullability::NonNullable => vortex_array::validity::Validity::NonNullable,
            Nullability::Nullable => vortex_array::validity::Validity::AllValid,
        };

        if let Ok(set_idx) = set_idx {
            let buffer = BitBuffer::from_iter((0..lhs.len()).map(|idx| idx == set_idx));
            Ok(Some(BoolArray::new(buffer, validity).to_array()))
        } else {
            Ok(Some(
                ConstantArray::new(Scalar::bool(false, nullability), lhs.len()).to_array(),
            ))
        }
    }
}

/// Find the index where `base + idx * multiplier == intercept`, if one exists.
///
/// # Errors
/// Return `VortexError` if:
/// - `len` is 0
/// - `intercept` or `multiplier` can't be cast to `base`'s PType
/// - `intercept` is outside the range of the sequence
/// - `intercept` doesn't fall exactly on a sequence value
pub(crate) fn find_intersection_scalar(
    base: PValue,
    multiplier: PValue,
    len: usize,
    intercept: PValue,
) -> VortexResult<usize> {
    match_each_integer_ptype!(base.ptype(), |P| {
        let intercept = intercept.cast::<P>()?;
        let base = base.cast::<P>()?;
        let multiplier = multiplier.cast::<P>()?;
        find_intersection(base, multiplier, len, intercept)
    })
}

fn find_intersection<P: NativePType>(
    base: P,
    multiplier: P,
    len: usize,
    intercept: P,
) -> VortexResult<usize> {
    if len == 0 {
        vortex_bail!("len == 0")
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
        vortex_bail!("{intercept} is outside of ({min_val}, {max_val}) range")
    }

    // Handle zero multiplier (constant sequence)
    if multiplier == P::zero() {
        if intercept == base {
            return Ok(0);
        } else {
            vortex_bail!("{intercept} != {base} with zero multiplier")
        }
    }

    // Check if (intercept - base) is evenly divisible by multiplier
    let diff = intercept - base;
    if diff % multiplier != P::zero() {
        vortex_bail!("{diff} % {multiplier} != 0")
    }

    let idx = diff / multiplier;
    idx.to_usize()
        .ok_or_else(|| vortex_err!("Cannot represent {idx} as usize"))
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::Operator;
    use vortex_array::compute::compare;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::Nullability::Nullable;

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
