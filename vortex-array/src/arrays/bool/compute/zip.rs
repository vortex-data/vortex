// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
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
/// of the generic per-run builder. Validity is combined with the shared `zip_validity`, which itself
/// reuses this kernel (terminating immediately, since validity bitmaps are non-nullable).
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

        let values = zip_value_bits(
            &if_true.to_bit_buffer(),
            &if_false.to_bit_buffer(),
            mask_bits,
        );

        let validity = zip_validity(if_true.validity()?, if_false.validity()?, &mask)?;

        Ok(Some(BoolArray::new(values, validity).into_array()))
    }
}

fn zip_value_bits(if_true: &BitBuffer, if_false: &BitBuffer, mask: &BitBuffer) -> BitBuffer {
    assert_eq!(if_true.len(), if_false.len());
    assert_eq!(if_true.len(), mask.len());

    let true_chunks = if_true.chunks();
    let false_chunks = if_false.chunks();
    let mask_chunks = mask.chunks();

    let mut values = BufferMut::<u64>::with_capacity(true_chunks.num_u64s());
    for ((true_bits, false_bits), mask_bits) in true_chunks
        .iter()
        .zip(false_chunks.iter())
        .zip(mask_chunks.iter())
    {
        values.push((true_bits & mask_bits) | (false_bits & !mask_bits));
    }

    if true_chunks.remainder_len() != 0 {
        let true_bits = true_chunks.remainder_bits();
        let false_bits = false_chunks.remainder_bits();
        let mask_bits = mask_chunks.remainder_bits();
        values.push((true_bits & mask_bits) | (false_bits & !mask_bits));
    }

    BitBuffer::new(values.freeze().into_byte_buffer(), if_true.len())
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use super::zip_value_bits;
    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::Bool;
    use crate::arrays::BoolArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;

    #[test]
    fn blend_value_bits_boundaries() {
        for len in [0usize, 1, 2, 7, 8, 9, 63, 64, 65, 127, 128] {
            let if_true = (0..len).map(|i| i.is_multiple_of(2)).collect();
            let if_false = (0..len).map(|i| i.is_multiple_of(3)).collect();
            let mask = (0..len).map(|i| i % 3 != 1).collect();

            let values = zip_value_bits(&if_true, &if_false, &mask);

            assert_eq!(values.len(), len);
            assert_eq!(
                values.iter().collect::<Vec<_>>(),
                (0..len)
                    .map(|i| {
                        if i % 3 != 1 {
                            i.is_multiple_of(2)
                        } else {
                            i.is_multiple_of(3)
                        }
                    })
                    .collect::<Vec<_>>(),
                "failed for len {len}",
            );
        }
    }

    /// Blend two non-nullable bool arrays across the 64-bit mask chunk boundary + remainder.
    #[test]
    fn zip_nonnull_spans_mask_chunks() -> VortexResult<()> {
        let len = 150usize;
        let if_true = BoolArray::from_iter((0..len).map(|i| i.is_multiple_of(2))).into_array();
        let if_false = BoolArray::from_iter((0..len).map(|i| i.is_multiple_of(3))).into_array();

        let bits: Vec<bool> = (0..len).map(|i| i.is_multiple_of(5) || i == 64).collect();
        let mask = Mask::from_iter(bits.iter().copied());

        let mut ctx = array_session().create_execution_ctx();
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
        assert_arrays_eq!(result, expected, &mut ctx);
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

        let mut ctx = array_session().create_execution_ctx();
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
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }
}
