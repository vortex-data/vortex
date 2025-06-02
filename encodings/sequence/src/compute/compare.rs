use num_traits::{AsPrimitive, FromPrimitive};
use vortex_array::arrays::{BoolArray, BooleanBuffer, ConstantArray};
use vortex_array::compute::{CompareKernel, Operator};
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef};
use vortex_dtype::{DType, Nullability, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::Scalar;

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
        let set_idx: Option<usize> = match_each_integer_ptype!(lhs.ptype(), |P| {
            let c = constant
                .as_primitive()
                .as_::<P>()?
                .vortex_expect("null constant already checked in entry");

            let base = lhs.base().as_primitive::<P>()?;
            let multiplier = lhs.multiplier().as_primitive::<P>()?;

            // Array is non-empty here.
            let count = <P>::from_usize(lhs.len() - 1).vortex_expect("idx must fit into type");

            let end_element = base + (multiplier * count);

            (c >= base && c <= end_element && c - base % multiplier == 0)
                .then(|| (c - base / multiplier).as_())
        });

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
