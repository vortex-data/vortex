// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::{BoolArray, BooleanBuffer, ConstantArray};
use vortex_array::compute::{CompareKernel, Operator};
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef};
use vortex_dtype::{DType, NativePType, Nullability, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::{PValue, Scalar};

use crate::SequenceArray;
use crate::array::SequenceVTable;

impl CompareKernel for SequenceVTable {
    fn compare(
        &self,
        lhs: &SequenceArray,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>> {
        if operator != Operator::Eq {
            return Ok(None);
        };

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
                .vortex_expect("non-null constant"),
        );

        let nullability = lhs.dtype().nullability() | rhs.dtype().nullability();
        let validity = match nullability {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => Validity::AllValid,
        };

        if let Some(set_idx) = set_idx {
            let buffer = BooleanBuffer::from_iter((0..lhs.len()).map(|idx| idx == set_idx));
            Ok(Some(BoolArray::new(buffer, validity).to_array()))
        } else {
            Ok(Some(
                ConstantArray::new(
                    Scalar::new(DType::Bool(nullability), false.into()),
                    lhs.len(),
                )
                .to_array(),
            ))
        }
    }
}

pub(crate) fn find_intersection_scalar(
    base: PValue,
    multiplier: PValue,
    len: usize,
    intercept: PValue,
) -> Option<usize> {
    match_each_integer_ptype!(base.ptype(), |P| {
        let intercept = intercept
            .as_primitive()
            .vortex_expect("constant pvalue matching already validated");

        let base = base
            .as_primitive::<P>()
            .vortex_expect("base pvalue matching already validated");
        let multiplier = multiplier
            .as_primitive::<P>()
            .vortex_expect("multiplier pvalue matching already validated");

        find_intersection(base, multiplier, len, intercept)
    })
}

fn find_intersection<P: NativePType>(
    base: P,
    multiplier: P,
    len: usize,
    intercept: P,
) -> Option<usize> {
    // Array is non-empty here.
    let count = <P>::from_usize(len - 1).vortex_expect("idx must fit into type");

    let end_element = base + (multiplier * count);

    (intercept.is_ge(base)
        && intercept.is_le(end_element)
        && (intercept - base) % multiplier == P::zero())
    .then(|| ((intercept - base) / multiplier).to_usize())
    .flatten()
}

#[cfg(test)]
mod tests {
    use vortex_array::ToCanonical;
    use vortex_array::arrays::{BoolArray, ConstantArray};
    use vortex_array::compute::{Operator, compare};

    use crate::SequenceArray;

    #[test]
    fn test_compare_match() {
        let lhs = SequenceArray::typed_new(2i64, 1, 4).unwrap();

        let rhs = ConstantArray::new(4i64, lhs.len());

        let result = compare(lhs.as_ref(), rhs.as_ref(), Operator::Eq).unwrap();

        assert_eq!(
            result.to_bool().unwrap().boolean_buffer(),
            BoolArray::from_iter(vec![false, false, true, false]).boolean_buffer(),
        )
    }

    #[test]
    fn test_compare_match_scale() {
        let lhs = SequenceArray::typed_new(2i64, 3, 4).unwrap();

        let rhs = ConstantArray::new(8i64, lhs.len());

        let result = compare(lhs.as_ref(), rhs.as_ref(), Operator::Eq).unwrap();

        assert_eq!(
            result.to_bool().unwrap().boolean_buffer(),
            BoolArray::from_iter(vec![false, false, true, false]).boolean_buffer(),
        )
    }

    #[test]
    fn test_compare_no_match() {
        let lhs = SequenceArray::typed_new(2i64, 1, 4).unwrap();

        let rhs = ConstantArray::new(1i64, lhs.len());

        let result = compare(lhs.as_ref(), rhs.as_ref(), Operator::Eq).unwrap();

        assert_eq!(
            result.to_bool().unwrap().boolean_buffer(),
            BoolArray::from_iter(vec![false, false, false, false]).boolean_buffer(),
        )
    }
}
