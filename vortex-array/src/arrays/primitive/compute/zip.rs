// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;

use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;
use crate::scalar_fn::fns::zip::ZipKernel;
use crate::scalar_fn::fns::zip::zip_validity;

/// A dedicated primitive zip kernel that selects values branchlessly per row.
///
/// The generic zip path copies runs of `if_true`/`if_false` between mask boundaries, which is fast
/// for clustered masks but degrades to per-element work on fragmented masks. This kernel instead
/// walks the mask as 64-bit chunks and blends both sides per row without a data-dependent branch,
/// so the inner loop stays branch-free and auto-vectorizable regardless of mask shape.
impl ZipKernel for Primitive {
    fn zip(
        if_true: ArrayView<'_, Primitive>,
        if_false: &ArrayRef,
        mask: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(if_false) = if_false.as_opt::<Primitive>() else {
            return Ok(None);
        };

        if if_true.ptype() != if_false.ptype() {
            vortex_bail!(
                "zip requires if_true and if_false to share a primitive type, got {} and {}",
                if_true.ptype(),
                if_false.ptype()
            );
        }

        // Null mask entries select `if_false`, matching `Zip`'s SQL ELSE semantics.
        let mask = mask.try_to_mask_fill_null_false(ctx)?;
        match &mask {
            // Defer trivial masks to the generic zip, which just casts the surviving side.
            Mask::AllTrue(_) | Mask::AllFalse(_) => return Ok(None),
            Mask::Values(_) => {}
        }

        let validity = zip_validity(if_true.validity()?, if_false.validity()?, &mask)?;

        // TODO(perf): inspect the mask's true_count (and validity) to special-case heavily-skewed
        // masks. When one side dominates (true_count near 0 or near len), it is cheaper to bulk
        // copy — or mutate in place, if the dominant side is uniquely owned — that side's values
        // and validity, then conditionally pull in only the minority rows from the other side,
        // rather than blending every row.
        let array = match_each_native_ptype!(if_true.ptype(), |T| {
            let values =
                select_values::<T>(if_true.as_slice::<T>(), if_false.as_slice::<T>(), &mask);
            PrimitiveArray::new(values.freeze(), validity).into_array()
        });
        Ok(Some(array))
    }
}

/// Branchlessly blend `if_true` and `if_false` per row into a fresh value buffer.
fn select_values<T: NativePType>(
    true_values: &[T],
    false_values: &[T],
    mask: &Mask,
) -> BufferMut<T> {
    let len = true_values.len();
    let mut out = BufferMut::<T>::with_capacity(len);
    {
        let out_slice = out.spare_capacity_mut();

        let mask_bits = mask
            .values()
            .vortex_expect("mask is Mask::Values")
            .bit_buffer();
        // TODO(perf): `unaligned_chunks` is a faster single-buffer iterator than `chunks`; switch to
        // it here, handling its lead/trailing padding.
        let chunks = mask_bits.chunks();

        let mut base = 0;
        for word in chunks.iter() {
            let end = base + 64;
            select_block(
                word,
                &true_values[base..end],
                &false_values[base..end],
                &mut out_slice[base..end],
            );
            base = end;
        }

        let remainder = chunks.remainder_len();
        if remainder > 0 {
            let end = base + remainder;
            select_block(
                chunks.remainder_bits(),
                &true_values[base..end],
                &false_values[base..end],
                &mut out_slice[base..end],
            );
        }
    }

    // SAFETY: `select_block` initialized every slot covered by the chunks plus remainder, i.e. `len`.
    unsafe { out.set_len(len) };
    out
}

/// Blend one 64-bit mask chunk's worth of rows: bit `j` (LSB-first) keeps `true_values[j]`, an unset
/// bit keeps `false_values[j]`. Slices are trimmed to the output length up front so the compiler can
/// elide bounds checks and lower the body to a vector blend / conditional move.
#[inline]
fn select_block<T: NativePType>(
    word: u64,
    true_values: &[T],
    false_values: &[T],
    out: &mut [MaybeUninit<T>],
) {
    let n = out.len();
    let true_values = &true_values[..n];
    let false_values = &false_values[..n];
    for j in 0..n {
        let pick = (word >> j) & 1 == 1;
        out[j].write(if pick {
            true_values[j]
        } else {
            false_values[j]
        });
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures use small indices that fit the target widths"
    )]

    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::Primitive;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;

    /// The branchless kernel must agree with the scalar reference across the chunk boundary (index
    /// 63/64) and the trailing remainder, for non-nullable inputs.
    #[test]
    fn zip_nonnull_spans_mask_chunks() -> VortexResult<()> {
        let len = 150usize;
        let if_true = PrimitiveArray::from_iter(0..len as i64).into_array();
        let if_false = PrimitiveArray::from_iter((0..len as i64).map(|i| 1_000 + i)).into_array();

        let bits: Vec<bool> = (0..len).map(|i| i.is_multiple_of(3) || i == 64).collect();
        let mask = Mask::from_iter(bits.iter().copied());

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = mask
            .into_array()
            .zip(if_true, if_false)?
            .execute::<ArrayRef>(&mut ctx)?;
        assert!(result.is::<Primitive>());

        let expected = PrimitiveArray::from_iter(
            (0..len).map(|i| if bits[i] { i as i64 } else { 1_000 + i as i64 }),
        )
        .into_array();
        assert_arrays_eq!(result, expected);
        Ok(())
    }

    /// With `Validity::Array` on both sides the kernel must select values and validity from the
    /// chosen side across the chunk boundary.
    #[test]
    fn zip_nullable_selects_values_and_validity() -> VortexResult<()> {
        let len = 130usize;
        let if_true =
            PrimitiveArray::from_option_iter((0..len as i64).map(|i| (i % 4 != 0).then_some(i)))
                .into_array();
        let if_false = PrimitiveArray::from_option_iter(
            (0..len as i64).map(|i| (i % 5 != 0).then_some(1_000 + i)),
        )
        .into_array();

        let bits: Vec<bool> = (0..len).map(|i| i.is_multiple_of(2)).collect();
        let mask = Mask::from_iter(bits.iter().copied());

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = mask
            .into_array()
            .zip(if_true, if_false)?
            .execute::<ArrayRef>(&mut ctx)?;
        assert!(result.is::<Primitive>());

        let expected = PrimitiveArray::from_option_iter((0..len).map(|i| {
            let v = i as i64;
            if bits[i] {
                (!i.is_multiple_of(4)).then_some(v)
            } else {
                (!i.is_multiple_of(5)).then_some(1_000 + v)
            }
        }))
        .into_array();
        assert_arrays_eq!(result, expected);
        Ok(())
    }
}
