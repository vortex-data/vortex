// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal-value execution: the optimized [`Interleave`](super::super::Interleave) path for
//! decimal values.

use num_traits::AsPrimitive;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::super::Interleave;
use super::super::InterleaveArrayExt;
use super::check_selector_bounds;
use crate::array::Array;
use crate::array::ArrayView;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::Primitive;
use crate::dtype::BigCast;
use crate::dtype::NativeDecimalType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::match_each_decimal_value_type;
use crate::match_each_unsigned_integer_ptype;
use crate::require_child;
use crate::validity::Validity;

/// Gathers `N` decimal values under unsigned `array_indices` / `row_indices` selectors.
///
/// The values share a [`DecimalDType`](crate::dtype::DecimalDType) but may store their scaled
/// integers at different physical widths; each value's buffer is widened to the widest width among
/// the values once up front, so the gather itself is monomorphized on a single value type.
#[expect(
    clippy::cognitive_complexity,
    reason = "the gather dispatches over the decimal value type and both selector widths"
)]
pub(super) fn execute(
    array: Array<Interleave>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionResult> {
    let num_values = array.num_values();

    // Drive every value and both selectors to canonical encodings so we can operate on raw
    // buffers.
    let mut array = array;
    for i in 0..num_values {
        array = require_child!(array, array.value(i), i => Decimal);
    }
    array = require_child!(array, array.array_indices(), num_values => Primitive);
    array = require_child!(array, array.row_indices(), num_values + 1 => Primitive);

    let dtype = array.as_ref().dtype().clone();
    let len = array.as_ref().len();
    let nullable = dtype.is_nullable();

    let decimal_dtype = array.value(0).as_::<Decimal>().decimal_dtype();
    let values_type = (0..num_values)
        .map(|i| array.value(i).as_::<Decimal>().values_type())
        .max()
        .vortex_expect("interleave has at least 2 values");

    let mut value_validity = Vec::with_capacity(num_values);
    for i in 0..num_values {
        let value = array.value(i).as_::<Decimal>();
        let validity = nullable
            .then(|| value.validity()?.execute_mask(value.len(), ctx))
            .transpose()?;
        value_validity.push(validity);
    }

    let array_indices = array.array_indices().as_::<Primitive>();
    let row_indices = array.row_indices().as_::<Primitive>();
    let decimal = match_each_decimal_value_type!(values_type, |D| {
        let value_buffers: Vec<Buffer<D>> = (0..num_values)
            .map(|i| widen::<D>(array.value(i).as_::<Decimal>()))
            .collect();
        let (values, validity) = match_each_unsigned_integer_ptype!(array_indices.ptype(), |A| {
            match_each_unsigned_integer_ptype!(row_indices.ptype(), |R| {
                gather(
                    len,
                    &value_buffers,
                    &value_validity,
                    array_indices.as_slice::<A>(),
                    row_indices.as_slice::<R>(),
                    nullable,
                )?
            })
        });
        let validity = match validity {
            Some(bits) => Validity::from(bits.freeze()),
            None => Validity::NonNullable,
        };
        DecimalArray::try_new(values, decimal_dtype, validity)?
    });

    Ok(ExecutionResult::done(decimal))
}

/// Materializes a value's scaled integers at width `D`, the widest values type among the
/// interleaved values. Zero-copy when the value is already stored at width `D`.
fn widen<D: NativeDecimalType>(value: ArrayView<'_, Decimal>) -> Buffer<D> {
    if value.values_type() == D::DECIMAL_TYPE {
        return value.buffer::<D>();
    }
    match_each_decimal_value_type!(value.values_type(), |S| {
        value
            .buffer::<S>()
            .iter()
            .map(|&v| {
                <D as BigCast>::from(v).vortex_expect("widening decimal cast cannot overflow")
            })
            .collect()
    })
}

/// The scatter, monomorphized on the value width and the selector integer widths.
fn gather<D: NativeDecimalType, A: AsPrimitive<usize>, R: AsPrimitive<usize>>(
    len: usize,
    value_buffers: &[Buffer<D>],
    value_validity: &[Option<Mask>],
    branches: &[A],
    rows: &[R],
    nullable: bool,
) -> VortexResult<(Buffer<D>, Option<BitBufferMut>)> {
    let value_lens: Vec<usize> = value_buffers.iter().map(|b| b.len()).collect();
    check_selector_bounds(branches, rows, &value_lens)?;

    let values = (0..len)
        .map(|i| value_buffers[branches[i].as_()][rows[i].as_()])
        .collect();

    // A missing per-value mask means every row of that value is valid; only materialized when the
    // output can be null.
    let validity = nullable.then(|| {
        BitBufferMut::collect_bool(len, |i| {
            value_validity[branches[i].as_()]
                .as_ref()
                .is_none_or(|mask| mask.value(rows[i].as_()))
        })
    });

    Ok((values, validity))
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::DecimalArray;
    use crate::arrays::InterleaveArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DecimalDType;
    use crate::validity::Validity;

    fn selectors(indices: &[(u32, u32)]) -> (ArrayRef, ArrayRef) {
        (
            PrimitiveArray::from_iter(indices.iter().map(|&(a, _)| a)).into_array(),
            PrimitiveArray::from_iter(indices.iter().map(|&(_, r)| r)).into_array(),
        )
    }

    #[test]
    fn interleave_decimal_reorders_and_repeats() -> VortexResult<()> {
        let ddtype = DecimalDType::new(5, 2);
        let v0 = DecimalArray::from_iter([100i32, 200, 300], ddtype).into_array();
        let v1 = DecimalArray::from_iter([-400i32, 500], ddtype).into_array();
        let (array_indices, row_indices) = selectors(&[(1, 0), (0, 2), (0, 0), (1, 1), (0, 0)]);

        let interleaved = InterleaveArray::try_new(vec![v0, v1], array_indices, row_indices)?;

        let expected = DecimalArray::from_iter([-400i32, 300, 100, 500, 100], ddtype);
        assert_arrays_eq!(interleaved, expected);
        Ok(())
    }

    #[test]
    fn interleave_decimal_nullable() -> VortexResult<()> {
        let ddtype = DecimalDType::new(10, 1);
        let v0 = DecimalArray::from_option_iter([Some(10i64), None], ddtype).into_array();
        let v1 = DecimalArray::from_option_iter([None, Some(20i64)], ddtype).into_array();
        let (array_indices, row_indices) = selectors(&[(1, 1), (0, 1), (1, 0), (0, 0)]);

        let interleaved = InterleaveArray::try_new(vec![v0, v1], array_indices, row_indices)?;

        let expected =
            DecimalArray::from_option_iter([Some(20i64), None, None, Some(10i64)], ddtype);
        assert_arrays_eq!(interleaved, expected);
        Ok(())
    }

    #[test]
    fn interleave_decimal_mixed_value_widths() -> VortexResult<()> {
        // Same logical decimal type stored at different physical widths; the gather widens to the
        // widest width (i64) before stitching.
        let ddtype = DecimalDType::new(10, 0);
        let v0 = DecimalArray::new(buffer![1i32, 2], ddtype, Validity::NonNullable).into_array();
        let v1 = DecimalArray::new(buffer![3i64, 4], ddtype, Validity::NonNullable).into_array();
        let (array_indices, row_indices) = selectors(&[(0, 0), (1, 1), (1, 0), (0, 1)]);

        let interleaved = InterleaveArray::try_new(vec![v0, v1], array_indices, row_indices)?;

        let expected = DecimalArray::from_iter([1i64, 4, 3, 2], ddtype);
        assert_arrays_eq!(interleaved, expected);
        Ok(())
    }

    #[test]
    fn interleave_decimal_row_out_of_bounds() -> VortexResult<()> {
        let ddtype = DecimalDType::new(5, 0);
        let v0 = DecimalArray::from_iter([1i32], ddtype).into_array();
        let v1 = DecimalArray::from_iter([2i32], ddtype).into_array();
        let (array_indices, row_indices) = selectors(&[(0, 0), (1, 1)]);

        let interleaved = InterleaveArray::try_new(vec![v0, v1], array_indices, row_indices)?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let err = interleaved
            .into_array()
            .execute::<Canonical>(&mut ctx)
            .err()
            .vortex_expect("expected out-of-bounds row index to error");
        assert!(err.to_string().contains("out of bounds"), "{err}");
        Ok(())
    }
}
