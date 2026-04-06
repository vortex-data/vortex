// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Masked;
use crate::arrays::dict::TakeReduce;
use crate::arrays::masked::MaskedData;
use crate::builtins::ArrayBuiltins;
use crate::scalar::Scalar;

impl TakeReduce for Masked {
    fn take(array: ArrayView<'_, Masked>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let taken_child = if !indices.all_valid()? {
            // This is safe because we'll mask out these positions in the validity.
            let fill_scalar = Scalar::zero_value(indices.dtype());
            let filled_take_indices = indices.clone().fill_null(fill_scalar)?;
            array.child().take(filled_take_indices)?
        } else {
            array.child().take(indices.clone())?
        };

        // Compute the new validity by taking from array's validity and merging with indices validity
        let taken_validity = array.validity().take(indices)?;

        // Construct new MaskedArray
        Ok(Some(
            MaskedData::try_new(taken_child, taken_validity)?.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::IntoArray;
    use crate::arrays::MaskedArray;
    use crate::arrays::PrimitiveArray;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::validity::Validity;

    #[rstest]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]).into_array(),
            Validity::from_iter([true, true, false, true, false])
        ).unwrap()
    )]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
            Validity::AllValid
        ).unwrap()
    )]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter(0..100).into_array(),
            Validity::from_iter((0..100).map(|i| i % 3 != 0))
        ).unwrap()
    )]
    fn test_take_masked_conformance(#[case] array: MaskedArray) {
        test_take_conformance(&array.into_array());
    }
}
