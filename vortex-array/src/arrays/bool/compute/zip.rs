// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::bool::BoolArrayExt;
use crate::scalar_fn::fns::zip::ZipKernel;
use crate::scalar_fn::fns::zip::zip_validity;

/// A branchless boolean zip kernel that blends the two value bitmaps with the mask in one pass.
///
/// Booleans are bit-packed, so selecting `if_true` where the mask is set and `if_false` where it is
/// not is a single bitwise blend over the packed words — `(true & mask) | (false & !mask)` — instead
/// of the generic per-run builder. Validity is combined with [`zip_validity`], which itself reuses
/// this kernel (terminating immediately, since validity bitmaps are non-nullable).
impl ZipKernel for Bool {
    fn zip(
        if_true: ArrayView<'_, Bool>,
        if_false: &ArrayRef,
        mask: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(if_false) = if_false.as_opt::<Bool>() else {
            return Ok(None);
        };

        // Null mask entries select `if_false`, matching `Zip`'s SQL ELSE semantics.
        let mask = mask.try_to_mask_fill_null_false(ctx)?;
        let mask_values = match &mask {
            // Defer trivial masks to the generic zip, which just casts the surviving side.
            Mask::AllTrue(_) | Mask::AllFalse(_) => return Ok(None),
            Mask::Values(values) => values,
        };
        let mask_bits = mask_values.bit_buffer();

        // Branchless blend of the packed value bits: `(true & mask) | (false & !mask)`.
        let true_bits = if_true.to_bit_buffer();
        let false_bits = if_false.to_bit_buffer();
        let values = (&true_bits & mask_bits) | false_bits.bitand_not(mask_bits);

        let validity = zip_validity(if_true.validity()?, if_false.validity()?, &mask)?;

        Ok(Some(BoolArray::new(values, validity).into_array()))
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::Bool;
    use crate::arrays::BoolArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;

    /// Blend two non-nullable bool arrays across the 64-bit mask chunk boundary + remainder.
    #[test]
    fn zip_nonnull_spans_mask_chunks() -> VortexResult<()> {
        let len = 150usize;
        let if_true = BoolArray::from_iter((0..len).map(|i| i.is_multiple_of(2))).into_array();
        let if_false = BoolArray::from_iter((0..len).map(|i| i.is_multiple_of(3))).into_array();

        let bits: Vec<bool> = (0..len).map(|i| i.is_multiple_of(5) || i == 64).collect();
        let mask = Mask::from_iter(bits.iter().copied());

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = mask
            .into_array()
            .zip(if_true, if_false)?
            .execute::<ArrayRef>(&mut ctx)?;
        assert!(result.is::<Bool>());

        let expected = BoolArray::from_iter((0..len).map(|i| {
            if bits[i] {
                i.is_multiple_of(2)
            } else {
                i.is_multiple_of(3)
            }
        }))
        .into_array();
        assert_arrays_eq!(result, expected);
        Ok(())
    }

    /// With `Validity::Array` on both sides, select values and validity from the chosen side.
    #[test]
    fn zip_nullable_selects_values_and_validity() -> VortexResult<()> {
        let len = 130usize;
        let if_true = BoolArray::from_iter(
            (0..len).map(|i| (!i.is_multiple_of(4)).then_some(i.is_multiple_of(2))),
        )
        .into_array();
        let if_false = BoolArray::from_iter(
            (0..len).map(|i| (!i.is_multiple_of(5)).then_some(i.is_multiple_of(3))),
        )
        .into_array();

        let bits: Vec<bool> = (0..len).map(|i| i.is_multiple_of(2)).collect();
        let mask = Mask::from_iter(bits.iter().copied());

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = mask
            .into_array()
            .zip(if_true, if_false)?
            .execute::<ArrayRef>(&mut ctx)?;
        assert!(result.is::<Bool>());

        let expected = BoolArray::from_iter((0..len).map(|i| {
            if bits[i] {
                (!i.is_multiple_of(4)).then_some(i.is_multiple_of(2))
            } else {
                (!i.is_multiple_of(5)).then_some(i.is_multiple_of(3))
            }
        }))
        .into_array();
        assert_arrays_eq!(result, expected);
        Ok(())
    }
}
