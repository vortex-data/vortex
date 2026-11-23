// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Decimal;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::zip::ZipKernel;
use crate::zip::fixed_width::zip_fixed_width;

/// A decimal zip kernel that selects values branchlessly per row.
///
/// The row selection is shared with [`Primitive`](vortex_array::arrays::Primitive) through
/// `zip_fixed_width`.
impl ZipKernel for Decimal {
    fn zip(
        if_true: ArrayView<'_, Decimal>,
        if_false: &ArrayRef,
        mask: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(if_false) = if_false.as_opt::<Decimal>() else {
            return Ok(None);
        };

        vortex_ensure!(
            if_true.dtype().eq_ignore_nullability(if_false.dtype()),
            "zip requires if_true and if_false to share a decimal type, got {} and {}",
            if_true.dtype(),
            if_false.dtype()
        );

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

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Decimal;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::NativeDecimalType;
    use vortex_array::dtype::i256;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
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

    /// Builds a decimal array whose precision makes `T` its canonical storage width.
    fn decimal<T: NativeDecimalType>(values: Vec<T>, validity: Validity) -> ArrayRef {
        let mut ctx = SESSION.create_execution_ctx();
        DecimalArray::new(
            values.into_iter().collect::<Buffer<T>>(),
            DecimalDType::new(T::MAX_PRECISION, 2),
            validity,
            &mut ctx,
        )
        .into_array()
    }

    /// The kernel selects values and validity from the chosen side across the 64-bit chunk
    /// boundary and the trailing remainder, for every storage width.
    #[test]
    fn zip_selects_values_and_validity() -> VortexResult<()> {
        check_selects_values_and_validity(|i| i as i32)?;
        check_selects_values_and_validity(|i| i as i64)?;
        check_selects_values_and_validity(|i| i as i128)?;
        check_selects_values_and_validity(|i| i256::from_i128(i as i128))
    }

    fn check_selects_values_and_validity<T: NativeDecimalType>(
        value: impl Fn(usize) -> T,
    ) -> VortexResult<()> {
        let len = 150usize;
        let true_valid = |i: usize| !i.is_multiple_of(4);
        let false_valid = |i: usize| !i.is_multiple_of(5);
        let bits: Vec<bool> = (0..len).map(|i| i.is_multiple_of(3) || i == 64).collect();

        let if_true = decimal(
            (0..len).map(&value).collect(),
            Validity::from_iter((0..len).map(true_valid)),
        );
        let if_false = decimal(
            (0..len).map(|i| value(1_000 + i)).collect(),
            Validity::from_iter((0..len).map(false_valid)),
        );

        let mut ctx = SESSION.create_execution_ctx();
        let result = Mask::from_iter(bits.iter().copied())
            .into_array()
            .zip(if_true, if_false)?
            .execute::<ArrayRef>(&mut ctx)?;
        assert!(result.is::<Decimal>());

        let expected = decimal(
            (0..len)
                .map(|i| if bits[i] { value(i) } else { value(1_000 + i) })
                .collect(),
            Validity::from_iter((0..len).map(|i| {
                if bits[i] {
                    true_valid(i)
                } else {
                    false_valid(i)
                }
            })),
        );
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    /// Non-nullable branches stay non-nullable, and one nullable branch widens the result.
    #[rstest]
    #[case::non_nullable(Validity::NonNullable, Validity::NonNullable)]
    #[case::nullable_if_true(Validity::AllValid, Validity::NonNullable)]
    #[case::nullable_if_false(Validity::NonNullable, Validity::AllValid)]
    fn zip_unions_nullability(
        #[case] true_validity: Validity,
        #[case] false_validity: Validity,
    ) -> VortexResult<()> {
        let nullability = true_validity.nullability() | false_validity.nullability();
        let if_true = decimal(vec![1i64, 2, 3, 4], true_validity);
        let if_false = decimal(vec![10i64, 20, 30, 40], false_validity);

        let mut ctx = SESSION.create_execution_ctx();
        let result = Mask::from_iter([true, false, false, true])
            .into_array()
            .zip(if_true, if_false)?;

        let expected = decimal(vec![1i64, 20, 30, 4], Validity::from(nullability));
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }
}
