// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution for decimal [`Interleave`] values.
//!
//! Sources can use different stored widths for the same logical dtype. Decimal execution
//! normalizes them to the logical canonical width before gathering rows.

use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::super::Interleave;
use super::super::InterleaveArrayExt;
use super::super::InterleaveSlots;
use super::fixed_width::Source;
use super::fixed_width::gather;
use crate::AnyColumnar;
use crate::ArrayRef;
use crate::array::Array;
use crate::arrays::Constant;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::Primitive;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::decimal::copy_to_canonical_width;
use crate::dtype::DecimalDType;
use crate::dtype::NativeDecimalType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::match_each_decimal_value_type;
use crate::require_child;

pub(super) fn execute(
    mut array: Array<Interleave>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionResult> {
    let num_values = array.num_values();
    array =
        require_child!(array, array.array_indices(), InterleaveSlots::ARRAY_INDICES => Primitive);
    array = require_child!(array, array.row_indices(), InterleaveSlots::ROW_INDICES => Primitive);
    for i in 0..num_values {
        array = require_child!(array, array.value(i), InterleaveSlots::VALUES_OFFSET + i => AnyColumnar);
    }

    let validity = array.as_ref().validity()?;
    let dtype = DecimalDType::try_from(array.dtype())?;
    let output = match_each_decimal_value_type!(dtype.canonical_values_type(), |T| {
        DecimalArray::new(gather_values::<T>(&array, dtype, ctx)?, dtype, validity)
    });

    Ok(ExecutionResult::done(output))
}

fn gather_values<T: NativeDecimalType>(
    array: &Array<Interleave>,
    dtype: DecimalDType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Buffer<T>> {
    let sources = (0..array.num_values())
        .map(|i| prepare_source::<T>(array.value(i), dtype, ctx))
        .collect::<VortexResult<Vec<_>>>()?;

    gather(array, &sources, ctx.allocator())
}

/// Normalizes one decimal source while retaining a single payload for a constant.
fn prepare_source<T: NativeDecimalType>(
    value: &ArrayRef,
    dtype: DecimalDType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Source<T>> {
    if let Some(constant) = value.as_opt::<Constant>() {
        let payload = match constant.scalar().as_decimal().decimal_value() {
            Some(value) => value.cast::<T>().ok_or_else(|| {
                vortex_err!("interleave decimal constant must fit {dtype}, got {value}")
            })?,
            None => T::default(),
        };

        return Ok(Source::constant(payload, value.len(), ctx.allocator()));
    }

    let decimal = value.as_::<Decimal>();
    let data = if decimal.values_type() == T::DECIMAL_TYPE {
        decimal.buffer::<T>()
    } else {
        // A source can still store its values at a width other than the canonical width of its
        // dtype (#10701). The copy widens such values, or checks their precision and narrows them
        // with zeros in the null slots.
        let mask = decimal.validity()?.execute_mask(value.len(), ctx)?;
        let copy = match_each_decimal_value_type!(decimal.values_type(), |S| {
            copy_to_canonical_width(decimal.buffer::<S>().as_slice(), &mask, dtype)?
        });
        Buffer::<T>::from_byte_buffer(copy.unwrap_host())
    };

    Ok(Source::decoded(data))
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::arrays::DecimalArray;
    use crate::arrays::InterleaveArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::decimal::DecimalArrayExt;
    use crate::assert_arrays_eq;
    use crate::dtype::BigCast;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::NativeDecimalType;
    use crate::dtype::Nullability;
    use crate::dtype::i256;
    use crate::match_each_decimal_value_type;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    /// Interleaves `values` with one `(array_index, row_index)` pair for each output row, and
    /// checks that the result has the canonical width of its dtype and equals `expected`.
    #[track_caller]
    fn assert_interleaved(
        values: Vec<ArrayRef>,
        indices: &[(u8, u8)],
        expected: DecimalArray,
    ) -> VortexResult<()> {
        let branches = PrimitiveArray::from_iter(indices.iter().map(|&(branch, _)| branch));
        let rows = PrimitiveArray::from_iter(indices.iter().map(|&(_, row)| row));
        let array = InterleaveArray::try_new(values, branches.into_array(), rows.into_array())?;
        let mut ctx = SESSION.create_execution_ctx();

        let output = array.into_array().execute::<DecimalArray>(&mut ctx)?;
        assert_eq!(
            output.values_type(),
            expected.decimal_dtype().canonical_values_type()
        );
        assert_eq!(output.dtype(), expected.dtype());
        assert_arrays_eq!(output.into_array(), expected.into_array(), &mut ctx);

        Ok(())
    }

    /// Gathers rows out of order, with repeats and nulls, at each canonical width. The null slot of
    /// the nullable source holds `T::MIN`, which can be outside the precision.
    #[rstest]
    #[case::i8(2)]
    #[case::i16(4)]
    #[case::i32(9)]
    #[case::i64(18)]
    #[case::i128(38)]
    #[case::i256(76)]
    fn test_decimal_interleave(#[case] precision: u8) -> VortexResult<()> {
        let dtype = DecimalDType::new(precision, 2);
        match_each_decimal_value_type!(dtype.canonical_values_type(), |T| {
            let min = T::MIN_BY_PRECISION[usize::from(precision)];
            let max = T::MAX_BY_PRECISION[usize::from(precision)];
            let zero = T::default();
            let nullable = DecimalArray::new(
                Buffer::copy_from([min, T::MIN, max]),
                dtype,
                Validity::from_iter([true, false, true]),
            );
            let non_nullable = DecimalArray::new(
                Buffer::copy_from([max, min, zero]),
                dtype,
                Validity::NonNullable,
            );
            let expected = DecimalArray::new(
                Buffer::copy_from([zero, min, zero, max, max, min]),
                dtype,
                Validity::from_iter([true, true, false, true, true, true]),
            );

            assert_interleaved(
                vec![nullable.into_array(), non_nullable.into_array()],
                &[(1, 2), (0, 0), (0, 1), (1, 0), (0, 2), (0, 0)],
                expected,
            )
        })
    }

    /// A constant keeps one payload at the canonical width, also when its scalar stores the value
    /// at another width.
    #[rstest]
    #[case::narrower_scalar(76, Some(DecimalValue::I8(-7)))]
    #[case::wider_scalar(2, Some(DecimalValue::I256(i256::from_i128(-7))))]
    #[case::null(9, None)]
    fn test_decimal_interleave_constant(
        #[case] precision: u8,
        #[case] payload: Option<DecimalValue>,
    ) -> VortexResult<()> {
        let dtype = DecimalDType::new(precision, 2);
        let (scalar, validity) = match payload {
            Some(payload) => (
                Scalar::decimal(payload, dtype, Nullability::NonNullable),
                Validity::NonNullable,
            ),
            None => (
                Scalar::null(DType::Decimal(dtype, Nullability::Nullable)),
                Validity::from_iter([false, true, false, false]),
            ),
        };
        let constant = ConstantArray::new(scalar, 3).into_array();
        match_each_decimal_value_type!(dtype.canonical_values_type(), |T| {
            let seven = <T as BigCast>::from(7i8).vortex_expect("7 fits every decimal type");
            let minus_seven =
                <T as BigCast>::from(-7i8).vortex_expect("-7 fits every decimal type");
            let column = DecimalArray::new(buffer![seven], dtype, Validity::NonNullable);
            let expected = DecimalArray::new(
                buffer![minus_seven, seven, minus_seven, minus_seven],
                dtype,
                validity,
            );

            assert_interleaved(
                vec![constant, column.into_array()],
                &[(0, 2), (1, 0), (0, 1), (0, 2)],
                expected,
            )
        })
    }

    /// Sources that store their values at another width are copied to the canonical width first.
    /// The `i8` source widens, and the `i256` source narrows with garbage in its null slots.
    #[test]
    fn test_decimal_interleave_mixed_stored_widths() -> VortexResult<()> {
        let dtype = DecimalDType::new(9, 0);
        let narrow = DecimalArray::new(buffer![1i8, -2], dtype, Validity::NonNullable);
        let wide = DecimalArray::new(
            buffer![i256::from_i128(3), i256::MIN, i256::MAX],
            dtype,
            Validity::from_iter([true, false, false]),
        );
        let expected = DecimalArray::new(
            buffer![3i32, -2, 0, 1, 0],
            dtype,
            Validity::from_iter([true, true, false, true, false]),
        );

        assert_interleaved(
            vec![narrow.into_array(), wide.into_array()],
            &[(1, 0), (0, 1), (1, 1), (0, 0), (1, 2)],
            expected,
        )
    }

    #[test]
    fn test_decimal_interleave_empty() -> VortexResult<()> {
        let dtype = DecimalDType::new(9, 0);
        let empty = DecimalArray::new(Buffer::<i8>::empty(), dtype, Validity::NonNullable);
        let expected = DecimalArray::new(Buffer::<i32>::empty(), dtype, Validity::NonNullable);

        assert_interleaved(
            vec![empty.clone().into_array(), empty.into_array()],
            &[],
            expected,
        )
    }

    #[test]
    fn test_decimal_interleave_rejects_values_outside_precision() -> VortexResult<()> {
        let source = DecimalArray::new(
            buffer![100i16],
            DecimalDType::new(2, 0),
            Validity::NonNullable,
        )
        .into_array();
        let array = InterleaveArray::try_new(
            vec![source.clone(), source],
            PrimitiveArray::from_iter([0u8]).into_array(),
            PrimitiveArray::from_iter([0u8]).into_array(),
        )?
        .into_array();

        let error = array
            .execute::<DecimalArray>(&mut SESSION.create_execution_ctx())
            .expect_err("the narrowing copy to `i8` rejects 100 at precision 2");
        assert!(error.to_string().contains("precision"), "{error}");

        Ok(())
    }
}
