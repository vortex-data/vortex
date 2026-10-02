// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::fill_null::FillNullReduce;
use vortex_error::VortexResult;

use crate::RunEnd;
use crate::array::RunEndArrayExt;
use crate::array::RunEndArraySlotsExt;

impl FillNullReduce for RunEnd {
    fn fill_null(
        array: ArrayView<'_, Self>,
        fill_value: &Scalar,
    ) -> VortexResult<Option<ArrayRef>> {
        let new_values = array.values().fill_null(fill_value.clone())?;
        // SAFETY: modifying values only, does not affect ends
        Ok(Some(
            unsafe {
                RunEnd::new_unchecked(
                    array.ends().clone(),
                    new_values,
                    array.offset(),
                    array.len(),
                )
            }
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use crate::RunEnd;
    use crate::tests::SESSION;

    /// `null_as_false` pushes the coercion into the run values; nulls must still become `false`.
    #[test]
    fn null_as_false_over_nullable_run_end() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = BoolArray::from_iter([Some(true), None, Some(false), Some(true)]).into_array();
        let array = RunEnd::try_new(
            PrimitiveArray::from_iter([2u32, 5, 6, 9]).into_array(),
            values,
            &mut ctx,
        )?
        .into_array();

        let mask = array.slice(1..8)?.null_as_false().execute(&mut ctx)?;
        assert_eq!(
            mask,
            Mask::from_iter([true, false, false, false, false, true, true])
        );
        Ok(())
    }
}
