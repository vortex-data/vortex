// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::Decimal;
use crate::arrays::fixed_width::zip::zip_fixed_width;
use crate::scalar_fn::fns::zip::ZipKernel;

/// A decimal zip kernel that selects values branchlessly per row.
///
/// The row selection is shared with [`Primitive`](crate::arrays::Primitive) through
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

        // One decimal type can be stored at different widths. The generic zip widens them.
        if if_true.values_type() != if_false.values_type() {
            return Ok(None);
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

    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::Decimal;
    use crate::arrays::DecimalArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DecimalDType;
    use crate::dtype::NativeDecimalType;
    use crate::dtype::i256;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn decimal<T: NativeDecimalType>(values: Vec<T>, validity: Validity) -> ArrayRef {
        DecimalArray::new(
            values.into_iter().collect::<Buffer<T>>(),
            DecimalDType::new(38, 2),
            validity,
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

    /// Equal decimal types stored at different widths take the generic path.
    #[test]
    fn zip_mixed_storage_widths() -> VortexResult<()> {
        let if_true = decimal(vec![1i32, 2, 3, 4], Validity::NonNullable);
        let if_false = decimal(vec![10i64, 20, 30, 40], Validity::NonNullable);

        let mut ctx = SESSION.create_execution_ctx();
        let result = Mask::from_iter([true, false, true, false])
            .into_array()
            .zip(if_true, if_false)?;

        let expected = decimal(vec![1i64, 20, 3, 40], Validity::NonNullable);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }
}
