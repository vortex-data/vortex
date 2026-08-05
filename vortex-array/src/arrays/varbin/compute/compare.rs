// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;

use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::BoolArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBin;
use crate::arrays::varbin::VarBinArraySlotsExt;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::match_each_integer_ptype;
use crate::scalar_fn::fns::binary::CompareKernel;
use crate::scalar_fn::fns::operators::CompareOperator;

// This implementation exists so we can compare against a constant in encoded space, without
// canonicalizing the VarBin array to VarBinView.
impl CompareKernel for VarBin {
    fn compare(
        lhs: ArrayView<'_, VarBin>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(rhs_const) = rhs.as_constant() else {
            return Ok(None);
        };

        let len = lhs.len();

        // The compare adaptor resolves null constants before dispatching to this kernel, so
        // the scalar always carries a value.
        let rhs_bytes: &[u8] = match rhs_const.dtype() {
            DType::Binary(_) => rhs_const
                .as_binary()
                .value()
                .vortex_expect("RHS should not be null")
                .as_slice(),
            DType::Utf8(_) => rhs_const
                .as_utf8()
                .value()
                .vortex_expect("RHS should not be null")
                .as_str()
                .as_bytes(),
            _ => vortex_bail!("VarBinArray can only have type of Binary or Utf8"),
        };

        let buffer = if rhs_bytes.is_empty() {
            // Comparisons against "" only need the value lengths, i.e. the offset deltas.
            match operator {
                CompareOperator::Gte => BitBuffer::new_set(len), /* Every possible value is >= "" */
                CompareOperator::Lt => BitBuffer::new_unset(len), // No value is < ""
                CompareOperator::Eq | CompareOperator::Lte => {
                    let lhs_offsets = lhs.offsets().clone().execute::<PrimitiveArray>(ctx)?;
                    match_each_integer_ptype!(lhs_offsets.ptype(), |P| {
                        compare_offsets_to_empty::<P>(lhs_offsets, true)
                    })
                }
                CompareOperator::NotEq | CompareOperator::Gt => {
                    let lhs_offsets = lhs.offsets().clone().execute::<PrimitiveArray>(ctx)?;
                    match_each_integer_ptype!(lhs_offsets.ptype(), |P| {
                        compare_offsets_to_empty::<P>(lhs_offsets, false)
                    })
                }
            }
        } else {
            let lhs_offsets = lhs.offsets().clone().execute::<PrimitiveArray>(ctx)?;
            match_each_integer_ptype!(lhs_offsets.ptype(), |P| {
                compare_bytes_to_constant(
                    lhs_offsets.as_slice::<P>(),
                    lhs.bytes().as_slice(),
                    rhs_bytes,
                    operator,
                )
            })
        };

        Ok(Some(
            BoolArray::new(
                buffer,
                lhs.validity()?.union_nullability(rhs.dtype().nullability()),
            )
            .into_array(),
        ))
    }
}

fn compare_offsets_to_empty<P: IntegerPType>(offsets: PrimitiveArray, eq: bool) -> BitBuffer {
    let fn_ = if eq { P::eq } else { P::ne };
    let offsets = offsets.as_slice::<P>();
    BitBuffer::collect_bool(offsets.len() - 1, |idx| {
        let left = unsafe { offsets.get_unchecked(idx) };
        let right = unsafe { offsets.get_unchecked(idx + 1) };
        fn_(left, right)
    })
}

/// Compare every value in a VarBin array against a constant, resolving values through the
/// offsets. Dispatches the operator outside the lane loop so each predicate inlines into its
/// own loop.
fn compare_bytes_to_constant<P: IntegerPType>(
    offsets: &[P],
    bytes: &[u8],
    constant: &[u8],
    operator: CompareOperator,
) -> BitBuffer {
    match operator {
        CompareOperator::Eq => {
            collect_lane_bits(offsets, |start, end| value_eq(bytes, start, end, constant))
        }
        CompareOperator::NotEq => {
            collect_lane_bits(offsets, |start, end| !value_eq(bytes, start, end, constant))
        }
        CompareOperator::Gt => collect_lane_bits(offsets, |start, end| {
            value_cmp(bytes, start, end, constant).is_gt()
        }),
        CompareOperator::Gte => collect_lane_bits(offsets, |start, end| {
            value_cmp(bytes, start, end, constant).is_ge()
        }),
        CompareOperator::Lt => collect_lane_bits(offsets, |start, end| {
            value_cmp(bytes, start, end, constant).is_lt()
        }),
        CompareOperator::Lte => collect_lane_bits(offsets, |start, end| {
            value_cmp(bytes, start, end, constant).is_le()
        }),
    }
}

/// Bit-pack `predicate(offsets[i], offsets[i + 1])` over each lane of a VarBin array.
fn collect_lane_bits<P: IntegerPType>(
    offsets: &[P],
    predicate: impl Fn(usize, usize) -> bool,
) -> BitBuffer {
    BitBuffer::collect_bool(offsets.len() - 1, |idx| {
        // SAFETY: `collect_bool` yields idx < offsets.len() - 1.
        let start = unsafe { offsets.get_unchecked(idx) }.as_();
        let end = unsafe { offsets.get_unchecked(idx + 1) }.as_();
        predicate(start, end)
    })
}

/// Whether `bytes[start..end]` equals `constant`, comparing lengths first so lanes of a
/// different length never touch the value bytes.
///
/// Offsets at null positions are not validated, so an out-of-bounds or inverted range is
/// possible there; such lanes answer `false`, and validity masks them out of the result anyway.
#[inline(always)]
fn value_eq(bytes: &[u8], start: usize, end: usize, constant: &[u8]) -> bool {
    // A lane can only match when its length equals the constant's, so lanes of a different
    // length answer without touching the value bytes. An inverted garbage range (start > end)
    // wraps to a huge value that never equals `constant.len()`.
    end.wrapping_sub(start) == constant.len()
        && bytes.get(start..end).is_some_and(|value| value == constant)
}

/// Order `bytes[start..end]` against `constant`, treating the unvalidated garbage ranges that
/// can appear at null positions as empty; validity masks those lanes out of the result anyway.
#[inline(always)]
fn value_cmp(bytes: &[u8], start: usize, end: usize, constant: &[u8]) -> Ordering {
    bytes.get(start..end).unwrap_or_default().cmp(constant)
}

#[cfg(test)]
mod test {
    use vortex_buffer::BitBuffer;
    use vortex_buffer::ByteBuffer;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::bool::BoolArrayExt;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::operators::Operator;

    #[test]
    fn test_binary_compare() {
        let mut ctx = array_session().create_execution_ctx();
        let array = VarBinArray::from_iter(
            [Some(b"abc".to_vec()), None, Some(b"def".to_vec())],
            DType::Binary(Nullability::Nullable),
        );
        let result = array
            .into_array()
            .binary(
                ConstantArray::new(
                    Scalar::binary(ByteBuffer::copy_from(b"abc"), Nullability::Nullable),
                    3,
                )
                .into_array(),
                Operator::Eq,
            )
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap();

        assert_eq!(
            &result
                .as_ref()
                .validity()
                .unwrap()
                .execute_mask(result.as_ref().len(), &mut ctx)
                .unwrap()
                .to_bit_buffer(),
            &BitBuffer::from_iter([true, false, true])
        );
        assert_eq!(
            result.to_bit_buffer(),
            BitBuffer::from_iter([true, false, false])
        );
    }

    #[test]
    fn varbinview_compare() {
        let mut ctx = array_session().create_execution_ctx();
        let array = VarBinArray::from_iter(
            [Some(b"abc".to_vec()), None, Some(b"def".to_vec())],
            DType::Binary(Nullability::Nullable),
        );
        let vbv = VarBinViewArray::from_iter(
            [None, None, Some(b"def".to_vec())],
            DType::Binary(Nullability::Nullable),
        );
        let result = array
            .into_array()
            .binary(vbv.into_array(), Operator::Eq)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap();

        assert_eq!(
            result
                .as_ref()
                .validity()
                .unwrap()
                .execute_mask(result.as_ref().len(), &mut ctx)
                .unwrap()
                .to_bit_buffer(),
            BitBuffer::from_iter([false, false, true])
        );
        assert_eq!(
            result.to_bit_buffer(),
            BitBuffer::from_iter([false, true, true])
        );
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::ByteBuffer;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::varbin::builder::VarBinBuilder;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::operators::Operator;

    #[test]
    fn test_null_compare() {
        let arr = VarBinArray::from_iter([Some("h")], DType::Utf8(Nullability::NonNullable));

        let const_ = ConstantArray::new(Scalar::utf8("", Nullability::Nullable), 1);

        assert_eq!(
            arr.into_array()
                .binary(const_.into_array(), Operator::Eq)
                .unwrap()
                .dtype(),
            &DType::Bool(Nullability::Nullable)
        );
    }

    /// Regression: [`CompareKernel`] must handle every offset width; a `VarBinArray` built with
    /// `i64` offsets once failed the constant comparison. Triggering this only requires `i64`
    /// offsets, not large data.
    ///
    /// [`CompareKernel`]: super::CompareKernel
    #[test]
    fn varbin_i64_offsets_compare_constant() {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder =
            VarBinBuilder::<i64>::with_capacity(DType::Utf8(Nullability::NonNullable), 3);
        builder.append_value(b"abc");
        builder.append_value(b"xyz");
        builder.append_value(b"abc");
        let array = builder.finish_into_varbin();

        let result = array
            .into_array()
            .binary(
                ConstantArray::new(Scalar::utf8("abc", Nullability::NonNullable), 3).into_array(),
                Operator::Eq,
            )
            .unwrap();

        let expected = BoolArray::from_iter([true, false, true]);
        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[test]
    fn varbin_i64_offsets_compare_constant_binary() {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder =
            VarBinBuilder::<i64>::with_capacity(DType::Binary(Nullability::NonNullable), 3);
        builder.append_value(b"abc");
        builder.append_value(b"xyz");
        builder.append_value(b"abc");
        let array = builder.finish_into_varbin();

        let result = array
            .into_array()
            .binary(
                ConstantArray::new(
                    Scalar::binary(ByteBuffer::copy_from(b"abc"), Nullability::NonNullable),
                    3,
                )
                .into_array(),
                Operator::Eq,
            )
            .unwrap();

        let expected = BoolArray::from_iter([true, false, true]);
        assert_arrays_eq!(result, expected, &mut ctx);
    }
}
