// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Masked;
use crate::arrays::MaskedArray;
use crate::arrays::dict::TakeReduce;
use crate::arrays::masked::MaskedArraySlotsExt;
use crate::builtins::ArrayBuiltins;
use crate::scalar::Scalar;
use crate::validity::Validity;

impl TakeReduce for Masked {
    fn take(array: ArrayView<'_, Masked>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let indices_all_valid = matches!(
            indices.validity()?,
            Validity::NonNullable | Validity::AllValid
        );
        let taken_child = if !indices_all_valid {
            // This is safe because we'll mask out these positions in the validity.
            let fill_scalar = Scalar::zero_value(indices.dtype());
            let filled_take_indices = indices.clone().fill_null(fill_scalar)?;
            array.child().take(filled_take_indices)?
        } else {
            array.child().take(indices.clone())?
        };

        // Compute the new validity by taking from array's validity and merging with indices validity
        let taken_validity = array.validity()?.take(indices)?;

        // SAFETY: the child is null-free and is taken with non-null indices, so the taken child is
        // null-free. The child and validity are taken by the same indices, so their lengths match.
        Ok(Some(
            unsafe { MaskedArray::new_unchecked(taken_child, taken_validity) }.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::MaskedArray;
    use crate::arrays::PrimitiveArray;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[rstest]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]).into_array(),
            Validity::from_iter([true, true, false, true, false]),
            &mut SESSION.create_execution_ctx(),
        ).unwrap()
    )]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
            Validity::AllValid,
            &mut SESSION.create_execution_ctx(),
        ).unwrap()
    )]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter(0..100).into_array(),
            Validity::from_iter((0..100).map(|i| i % 3 != 0)),
            &mut SESSION.create_execution_ctx(),
        ).unwrap()
    )]
    fn test_take_masked_conformance(#[case] array: MaskedArray) {
        test_take_conformance(&array.into_array(), &mut SESSION.create_execution_ctx());
    }
}
