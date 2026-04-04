// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::PValue;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::array::Sequence;

impl CompareKernel for Sequence {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if operator == CompareOperator::NotEq {
            return Ok(None);
        }

        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };

        let intercept = constant
            .as_primitive()
            .pvalue()
            .vortex_expect("null constant handled in adaptor");
        let Ok(true_range) =
            find_true_range_scalar(lhs.base(), lhs.multiplier(), lhs.len(), intercept, operator)
        else {
            return Ok(None);
        };

        let nullability = lhs.dtype().nullability() | rhs.dtype().nullability();
        let validity = match nullability {
            Nullability::NonNullable => vortex_array::validity::Validity::NonNullable,
            Nullability::Nullable => vortex_array::validity::Validity::AllValid,
        };

        Ok(Some(
            BoolArray::new(range_to_bit_buffer(lhs.len(), true_range), validity).into_array(),
        ))
    }
}

fn range_to_bit_buffer(len: usize, true_range: Range<usize>) -> BitBuffer {
    if true_range.start == true_range.end {
        return BitBuffer::new_unset(len);
    }
    if true_range.start == 0 && true_range.end == len {
        return BitBuffer::new_set(len);
    }

    let mut buffer = BitBufferMut::new_unset(len);
    buffer.fill_range(true_range.start, true_range.end, true);
    buffer.freeze()
}

fn empty_range() -> Range<usize> {
    0..0
}

fn full_range(len: usize) -> Range<usize> {
    0..len
}

fn prefix_range(end: usize) -> Range<usize> {
    0..end
}

fn suffix_range(start: usize, len: usize) -> Range<usize> {
    start..len
}

fn singleton_range(index: usize) -> Range<usize> {
    index..index + 1
}

fn comparison_matches(ordering: Ordering, operator: CompareOperator) -> bool {
    match operator {
        CompareOperator::Eq => ordering.is_eq(),
        CompareOperator::NotEq => ordering.is_ne(),
        CompareOperator::Gt => ordering.is_gt(),
        CompareOperator::Gte => ordering.is_ge(),
        CompareOperator::Lt => ordering.is_lt(),
        CompareOperator::Lte => ordering.is_le(),
    }
}

fn constant_true_range(len: usize, ordering: Ordering, operator: CompareOperator) -> Range<usize> {
    if comparison_matches(ordering, operator) {
        full_range(len)
    } else {
        empty_range()
    }
}

fn usize_to_u128(value: usize) -> VortexResult<u128> {
    u128::try_from(value).map_err(|_| vortex_err!("Cannot represent {value} as u128"))
}

fn usize_to_i128(value: usize) -> VortexResult<i128> {
    i128::try_from(value).map_err(|_| vortex_err!("Cannot represent {value} as i128"))
}

fn ceil_div_positive_u128(lhs: u128, rhs: u128) -> u128 {
    debug_assert!(rhs > 0);
    if lhs == 0 { 0 } else { ((lhs - 1) / rhs) + 1 }
}

fn ceil_div_positive_i128(lhs: i128, rhs: i128) -> i128 {
    debug_assert!(lhs >= 0);
    debug_assert!(rhs > 0);
    if lhs == 0 { 0 } else { ((lhs - 1) / rhs) + 1 }
}

/// Find the first index where `base + idx * multiplier == intercept`, if one exists.
///
/// # Errors
/// Return `VortexError` if:
/// - `len` is 0
/// - `intercept` or `multiplier` can't be cast to `base`'s PType
/// - `intercept` is outside the range of the sequence
/// - `intercept` doesn't fall exactly on a sequence value
pub(crate) fn find_intersection_scalar(
    base: PValue,
    multiplier: PValue,
    len: usize,
    intercept: PValue,
) -> VortexResult<usize> {
    match_each_integer_ptype!(base.ptype(), |P| {
        let intercept = intercept.cast::<P>()?;
        let base = base.cast::<P>()?;
        let multiplier = multiplier.cast::<P>()?;
        find_intersection(base, multiplier, len, intercept)
    })
}

fn find_intersection<P: IntegerPType>(
    base: P,
    multiplier: P,
    len: usize,
    intercept: P,
) -> VortexResult<usize> {
    let true_range = find_true_range(base, multiplier, len, intercept, CompareOperator::Eq)?;
    if true_range.start == true_range.end {
        vortex_bail!("{intercept} does not intersect the sequence");
    }
    Ok(true_range.start)
}

fn find_true_range_scalar(
    base: PValue,
    multiplier: PValue,
    len: usize,
    intercept: PValue,
    operator: CompareOperator,
) -> VortexResult<Range<usize>> {
    match_each_integer_ptype!(base.ptype(), |P| {
        let intercept = intercept.cast::<P>()?;
        let base = base.cast::<P>()?;
        let multiplier = multiplier.cast::<P>()?;
        find_true_range(base, multiplier, len, intercept, operator)
    })
}

fn find_true_range<P: IntegerPType>(
    base: P,
    multiplier: P,
    len: usize,
    intercept: P,
    operator: CompareOperator,
) -> VortexResult<Range<usize>> {
    if len == 0 {
        vortex_bail!("len == 0");
    }

    if P::PTYPE.is_signed_int() {
        signed_true_range(
            base.to_i128()
                .ok_or_else(|| vortex_err!("Cannot represent {base} as i128"))?,
            multiplier
                .to_i128()
                .ok_or_else(|| vortex_err!("Cannot represent {multiplier} as i128"))?,
            len,
            intercept
                .to_i128()
                .ok_or_else(|| vortex_err!("Cannot represent {intercept} as i128"))?,
            operator,
        )
    } else {
        unsigned_true_range(
            base.to_u128()
                .ok_or_else(|| vortex_err!("Cannot represent {base} as u128"))?,
            multiplier
                .to_u128()
                .ok_or_else(|| vortex_err!("Cannot represent {multiplier} as u128"))?,
            len,
            intercept
                .to_u128()
                .ok_or_else(|| vortex_err!("Cannot represent {intercept} as u128"))?,
            operator,
        )
    }
}

#[allow(clippy::manual_is_multiple_of)]
fn unsigned_true_range(
    base: u128,
    multiplier: u128,
    len: usize,
    intercept: u128,
    operator: CompareOperator,
) -> VortexResult<Range<usize>> {
    if multiplier == 0 {
        return Ok(constant_true_range(len, base.cmp(&intercept), operator));
    }

    let last = base + multiplier * usize_to_u128(len - 1)?;

    let true_range = match operator {
        CompareOperator::Eq => {
            if intercept < base || intercept > last {
                empty_range()
            } else {
                let diff = intercept - base;
                if diff % multiplier == 0 {
                    singleton_range(
                        usize::try_from(diff / multiplier)
                            .map_err(|_| vortex_err!("index does not fit into usize"))?,
                    )
                } else {
                    empty_range()
                }
            }
        }
        CompareOperator::Lt => {
            let end = if intercept <= base {
                0
            } else if intercept > last {
                len
            } else {
                usize::try_from(ceil_div_positive_u128(intercept - base, multiplier))
                    .map_err(|_| vortex_err!("cut-point does not fit into usize"))?
            };
            prefix_range(end)
        }
        CompareOperator::Lte => {
            let end = if intercept < base {
                0
            } else if intercept >= last {
                len
            } else {
                usize::try_from(((intercept - base) / multiplier) + 1)
                    .map_err(|_| vortex_err!("cut-point does not fit into usize"))?
            };
            prefix_range(end)
        }
        CompareOperator::Gt => {
            let start = if intercept < base {
                0
            } else if intercept >= last {
                len
            } else {
                usize::try_from(((intercept - base) / multiplier) + 1)
                    .map_err(|_| vortex_err!("cut-point does not fit into usize"))?
            };
            suffix_range(start, len)
        }
        CompareOperator::Gte => {
            let start = if intercept <= base {
                0
            } else if intercept > last {
                len
            } else {
                usize::try_from(ceil_div_positive_u128(intercept - base, multiplier))
                    .map_err(|_| vortex_err!("cut-point does not fit into usize"))?
            };
            suffix_range(start, len)
        }
        CompareOperator::NotEq => vortex_bail!("NotEq cannot be represented as a single range"),
    };

    Ok(true_range)
}

#[allow(clippy::manual_is_multiple_of)]
fn signed_true_range(
    base: i128,
    multiplier: i128,
    len: usize,
    intercept: i128,
    operator: CompareOperator,
) -> VortexResult<Range<usize>> {
    if multiplier == 0 {
        return Ok(constant_true_range(len, base.cmp(&intercept), operator));
    }

    let last = base + multiplier * usize_to_i128(len - 1)?;

    let true_range = if multiplier > 0 {
        let max = last;
        match operator {
            CompareOperator::Eq => {
                if intercept < base || intercept > max {
                    empty_range()
                } else {
                    let diff = intercept - base;
                    if diff % multiplier == 0 {
                        singleton_range(
                            usize::try_from(diff / multiplier)
                                .map_err(|_| vortex_err!("index does not fit into usize"))?,
                        )
                    } else {
                        empty_range()
                    }
                }
            }
            CompareOperator::Lt => {
                let end = if intercept <= base {
                    0
                } else if intercept > max {
                    len
                } else {
                    usize::try_from(ceil_div_positive_i128(intercept - base, multiplier))
                        .map_err(|_| vortex_err!("cut-point does not fit into usize"))?
                };
                prefix_range(end)
            }
            CompareOperator::Lte => {
                let end = if intercept < base {
                    0
                } else if intercept >= max {
                    len
                } else {
                    usize::try_from(((intercept - base) / multiplier) + 1)
                        .map_err(|_| vortex_err!("cut-point does not fit into usize"))?
                };
                prefix_range(end)
            }
            CompareOperator::Gt => {
                let start = if intercept < base {
                    0
                } else if intercept >= max {
                    len
                } else {
                    usize::try_from(((intercept - base) / multiplier) + 1)
                        .map_err(|_| vortex_err!("cut-point does not fit into usize"))?
                };
                suffix_range(start, len)
            }
            CompareOperator::Gte => {
                let start = if intercept <= base {
                    0
                } else if intercept > max {
                    len
                } else {
                    usize::try_from(ceil_div_positive_i128(intercept - base, multiplier))
                        .map_err(|_| vortex_err!("cut-point does not fit into usize"))?
                };
                suffix_range(start, len)
            }
            CompareOperator::NotEq => {
                vortex_bail!("NotEq cannot be represented as a single range")
            }
        }
    } else {
        let min = last;
        let step = -multiplier;
        match operator {
            CompareOperator::Eq => {
                if intercept < min || intercept > base {
                    empty_range()
                } else {
                    let diff = base - intercept;
                    if diff % step == 0 {
                        singleton_range(
                            usize::try_from(diff / step)
                                .map_err(|_| vortex_err!("index does not fit into usize"))?,
                        )
                    } else {
                        empty_range()
                    }
                }
            }
            CompareOperator::Lt => {
                let start = if base < intercept {
                    0
                } else if min >= intercept {
                    len
                } else {
                    usize::try_from(((base - intercept) / step) + 1)
                        .map_err(|_| vortex_err!("cut-point does not fit into usize"))?
                };
                suffix_range(start, len)
            }
            CompareOperator::Lte => {
                let start = if base <= intercept {
                    0
                } else if min > intercept {
                    len
                } else {
                    usize::try_from(ceil_div_positive_i128(base - intercept, step))
                        .map_err(|_| vortex_err!("cut-point does not fit into usize"))?
                };
                suffix_range(start, len)
            }
            CompareOperator::Gt => {
                let end = if base <= intercept {
                    0
                } else if min > intercept {
                    len
                } else {
                    usize::try_from(ceil_div_positive_i128(base - intercept, step))
                        .map_err(|_| vortex_err!("cut-point does not fit into usize"))?
                };
                prefix_range(end)
            }
            CompareOperator::Gte => {
                let end = if base < intercept {
                    0
                } else if min >= intercept {
                    len
                } else {
                    usize::try_from(((base - intercept) / step) + 1)
                        .map_err(|_| vortex_err!("cut-point does not fit into usize"))?
                };
                prefix_range(end)
            }
            CompareOperator::NotEq => {
                vortex_bail!("NotEq cannot be represented as a single range")
            }
        }
    };

    Ok(true_range)
}

#[cfg(test)]
mod tests {
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::Nullability::Nullable;
    use vortex_array::scalar_fn::fns::binary::CompareKernel;
    use vortex_array::scalar_fn::fns::operators::CompareOperator;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_session::VortexSession;

    use crate::Sequence;

    #[test]
    fn test_compare_match() {
        let lhs = Sequence::try_new_typed(2i64, 1, NonNullable, 4).unwrap();
        let rhs = ConstantArray::new(4i64, lhs.len());
        let result = lhs
            .into_array()
            .binary(rhs.into_array(), Operator::Eq)
            .unwrap();
        let expected = BoolArray::from_iter([false, false, true, false]);
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn test_compare_match_scale() {
        let lhs = Sequence::try_new_typed(2i64, 3, Nullable, 4).unwrap();
        let rhs = ConstantArray::new(8i64, lhs.len());
        let result = lhs
            .into_array()
            .binary(rhs.into_array(), Operator::Eq)
            .unwrap();
        let expected = BoolArray::from_iter([Some(false), Some(false), Some(true), Some(false)]);
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn test_compare_no_match() {
        let lhs = Sequence::try_new_typed(2i64, 1, NonNullable, 4).unwrap();
        let rhs = ConstantArray::new(1i64, lhs.len());
        let result = lhs
            .into_array()
            .binary(rhs.into_array(), Operator::Eq)
            .unwrap();
        let expected = BoolArray::from_iter([false, false, false, false]);
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn test_compare_range_ascending() {
        let lhs = Sequence::try_new_typed(2i64, 3, NonNullable, 5).unwrap();

        let lt = lhs
            .clone()
            .into_array()
            .binary(
                ConstantArray::new(8i64, lhs.len()).into_array(),
                Operator::Lt,
            )
            .unwrap();
        assert_arrays_eq!(lt, BoolArray::from_iter([true, true, false, false, false]));

        let lte = lhs
            .clone()
            .into_array()
            .binary(
                ConstantArray::new(8i64, lhs.len()).into_array(),
                Operator::Lte,
            )
            .unwrap();
        assert_arrays_eq!(lte, BoolArray::from_iter([true, true, true, false, false]));

        let gt = lhs
            .clone()
            .into_array()
            .binary(
                ConstantArray::new(8i64, lhs.len()).into_array(),
                Operator::Gt,
            )
            .unwrap();
        assert_arrays_eq!(gt, BoolArray::from_iter([false, false, false, true, true]));

        let gte = lhs
            .into_array()
            .binary(ConstantArray::new(8i64, 5).into_array(), Operator::Gte)
            .unwrap();
        assert_arrays_eq!(gte, BoolArray::from_iter([false, false, true, true, true]));
    }

    #[test]
    fn test_compare_range_descending() {
        let lhs = Sequence::try_new_typed(14i64, -3, NonNullable, 5).unwrap();

        let lt = lhs
            .clone()
            .into_array()
            .binary(
                ConstantArray::new(8i64, lhs.len()).into_array(),
                Operator::Lt,
            )
            .unwrap();
        assert_arrays_eq!(lt, BoolArray::from_iter([false, false, false, true, true]));

        let lte = lhs
            .clone()
            .into_array()
            .binary(
                ConstantArray::new(8i64, lhs.len()).into_array(),
                Operator::Lte,
            )
            .unwrap();
        assert_arrays_eq!(lte, BoolArray::from_iter([false, false, true, true, true]));

        let gt = lhs
            .clone()
            .into_array()
            .binary(
                ConstantArray::new(8i64, lhs.len()).into_array(),
                Operator::Gt,
            )
            .unwrap();
        assert_arrays_eq!(gt, BoolArray::from_iter([true, true, false, false, false]));

        let gte = lhs
            .into_array()
            .binary(ConstantArray::new(8i64, 5).into_array(), Operator::Gte)
            .unwrap();
        assert_arrays_eq!(gte, BoolArray::from_iter([true, true, true, false, false]));
    }

    #[test]
    fn test_compare_constant_sequence_matches_all() {
        let lhs = Sequence::try_new_typed(7i64, 0, NonNullable, 4).unwrap();

        let eq = lhs
            .clone()
            .into_array()
            .binary(
                ConstantArray::new(7i64, lhs.len()).into_array(),
                Operator::Eq,
            )
            .unwrap();
        assert_arrays_eq!(eq, BoolArray::from_iter([true, true, true, true]));

        let lte = lhs
            .clone()
            .into_array()
            .binary(
                ConstantArray::new(7i64, lhs.len()).into_array(),
                Operator::Lte,
            )
            .unwrap();
        assert_arrays_eq!(lte, BoolArray::from_iter([true, true, true, true]));

        let gt = lhs
            .clone()
            .into_array()
            .binary(
                ConstantArray::new(6i64, lhs.len()).into_array(),
                Operator::Gt,
            )
            .unwrap();
        assert_arrays_eq!(gt, BoolArray::from_iter([true, true, true, true]));

        let lt = lhs
            .into_array()
            .binary(ConstantArray::new(7i64, 4).into_array(), Operator::Lt)
            .unwrap();
        assert_arrays_eq!(lt, BoolArray::from_iter([false, false, false, false]));
    }

    #[test]
    fn test_compare_nullable_range() {
        let lhs = Sequence::try_new_typed(2i64, 3, Nullable, 4).unwrap();
        let rhs = ConstantArray::new(5i64, lhs.len());
        let result = lhs
            .into_array()
            .binary(rhs.into_array(), Operator::Gte)
            .unwrap();
        let expected = BoolArray::from_iter([Some(false), Some(true), Some(true), Some(true)]);
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn test_compare_swapped_operands() {
        let rhs = Sequence::try_new_typed(2i64, 3, NonNullable, 5).unwrap();
        let lhs = ConstantArray::new(8i64, rhs.len());
        let result = lhs
            .into_array()
            .binary(rhs.into_array(), Operator::Gt)
            .unwrap();
        let expected = BoolArray::from_iter([true, true, false, false, false]);
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn test_compare_unsigned_sequence_range() {
        let lhs = Sequence::try_new_typed(2u64, 3, NonNullable, 5).unwrap();
        let rhs = ConstantArray::new(8u64, lhs.len());
        let result = lhs
            .into_array()
            .binary(rhs.into_array(), Operator::Gte)
            .unwrap();
        let expected = BoolArray::from_iter([false, false, true, true, true]);
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn test_compare_non_constant_rhs_returns_none() {
        let lhs = Sequence::try_new_typed(2i64, 3, NonNullable, 5).unwrap();
        let rhs = PrimitiveArray::from_iter([8i64; 5]).into_array();
        let mut ctx = ExecutionCtx::new(VortexSession::empty());

        let result = Sequence::compare(lhs.as_view(), &rhs, CompareOperator::Lt, &mut ctx).unwrap();

        assert!(result.is_none());
    }
}
