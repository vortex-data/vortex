// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Primitive;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::zip::ZipKernel;
use crate::zip::fixed_width::zip_fixed_width;

/// A dedicated primitive zip kernel that selects values branchlessly per row.
///
/// The row selection is shared with [`Decimal`](vortex_array::arrays::Decimal) through
/// `zip_fixed_width`.
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

        zip_fixed_width(if_true, if_false, mask, ctx)
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures use small indices that fit the target widths"
    )]

    use std::sync::LazyLock;

    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Primitive;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::builtins::ArrayFunctions;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();

        vortex_array::initialize(&session);

        crate::initialize(&session);

        session
    });

    /// The branchless kernel must agree with the scalar reference across the chunk boundary (index
    /// 63/64) and the trailing remainder, for non-nullable inputs.
    #[test]
    fn zip_nonnull_spans_mask_chunks() -> VortexResult<()> {
        let len = 150usize;
        let if_true = PrimitiveArray::from_iter(0..len as i64).into_array();
        let if_false = PrimitiveArray::from_iter((0..len as i64).map(|i| 1_000 + i)).into_array();

        let bits: Vec<bool> = (0..len).map(|i| i.is_multiple_of(3) || i == 64).collect();
        let mask = Mask::from_iter(bits.iter().copied());

        let mut ctx = SESSION.create_execution_ctx();
        let result = mask
            .into_array()
            .zip(if_true, if_false)?
            .execute::<ArrayRef>(&mut ctx)?;
        assert!(result.is::<Primitive>());

        let expected = PrimitiveArray::from_iter(
            (0..len).map(|i| if bits[i] { i as i64 } else { 1_000 + i as i64 }),
        )
        .into_array();
        assert_arrays_eq!(result, expected, &mut ctx);
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

        let mut ctx = SESSION.create_execution_ctx();
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
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }
}
