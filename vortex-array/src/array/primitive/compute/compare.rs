use vortex_error::VortexResult;

use crate::array::primitive::PrimitiveArray;
use crate::array::ConstantArray;
use crate::compute::{arrow_compare, MaybeCompareFn, Operator};
use crate::stats::{ArrayStatistics, Stat};
use crate::ArrayData;

impl MaybeCompareFn for PrimitiveArray {
    fn maybe_compare(
        &self,
        other: &ArrayData,
        operator: Operator,
    ) -> Option<VortexResult<ArrayData>> {
        // If the RHS is constant, then delegate to Arrow since.
        if other
            .statistics()
            .get_as::<bool>(Stat::IsConstant)
            .unwrap_or(false)
        {
            return Some(arrow_compare(self.as_ref(), other, operator));
        }

        if let Ok(constant) = ConstantArray::try_from(other) {
            return Some(arrow_compare(self.as_ref(), constant.as_ref(), operator));
        }

        // If the RHS is primitive, then delegate to Arrow.
        if let Ok(primitive) = PrimitiveArray::try_from(other) {
            return Some(arrow_compare(self.as_ref(), primitive.as_ref(), operator));
        }

        None
    }
}

#[cfg(test)]
#[allow(clippy::panic_in_result_fn)]
mod test {
    use itertools::Itertools;

    use super::*;
    use crate::array::BoolArray;
    use crate::compute::compare;
    use crate::IntoArrayVariant;

    fn to_int_indices(indices_bits: BoolArray) -> Vec<u64> {
        let filtered = indices_bits
            .boolean_buffer()
            .iter()
            .enumerate()
            .filter_map(|(idx, v)| {
                let valid_and_true = indices_bits.validity().is_valid(idx) & v;
                valid_and_true.then_some(idx as u64)
            })
            .collect_vec();
        filtered
    }

    #[test]
    fn test_basic_comparisons() -> VortexResult<()> {
        let arr = PrimitiveArray::from_nullable_vec(vec![
            Some(1i32),
            Some(2),
            Some(3),
            Some(4),
            None,
            Some(5),
            Some(6),
            Some(7),
            Some(8),
            None,
            Some(9),
            None,
        ])
        .into_array();

        let matches = compare(&arr, &arr, Operator::Eq)?.into_bool()?;
        assert_eq!(to_int_indices(matches), [0u64, 1, 2, 3, 5, 6, 7, 8, 10]);

        let matches = compare(&arr, &arr, Operator::NotEq)?.into_bool()?;
        let empty: [u64; 0] = [];
        assert_eq!(to_int_indices(matches), empty);

        let other = PrimitiveArray::from_nullable_vec(vec![
            Some(1i32),
            Some(2),
            Some(3),
            Some(4),
            None,
            Some(6),
            Some(7),
            Some(8),
            Some(9),
            None,
            Some(10),
            None,
        ])
        .into_array();

        let matches = compare(&arr, &other, Operator::Lte)?.into_bool()?;
        assert_eq!(to_int_indices(matches), [0u64, 1, 2, 3, 5, 6, 7, 8, 10]);

        let matches = compare(&arr, &other, Operator::Lt)?.into_bool()?;
        assert_eq!(to_int_indices(matches), [5u64, 6, 7, 8, 10]);

        let matches = compare(&other, &arr, Operator::Gte)?.into_bool()?;
        assert_eq!(to_int_indices(matches), [0u64, 1, 2, 3, 5, 6, 7, 8, 10]);

        let matches = compare(&other, &arr, Operator::Gt)?.into_bool()?;
        assert_eq!(to_int_indices(matches), [5u64, 6, 7, 8, 10]);
        Ok(())
    }
}
