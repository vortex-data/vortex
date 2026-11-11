// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::PValue;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::array::Sequence;
use crate::eval;

impl CompareKernel for Sequence {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };

        let value = constant
            .as_primitive()
            .pvalue()
            .vortex_expect("null constant handled in adaptor");

        let len = lhs.len();
        let bits = match operator {
            // A sequence with a non-zero step holds each value at most once, so equality matches
            // at most one row. Inequality is the negation of that answer.
            CompareOperator::Eq | CompareOperator::NotEq => {
                // Fall back for non-integer constants.
                let Some(intersection) =
                    find_intersection(lhs.base(), lhs.multiplier(), len, value)
                else {
                    return Ok(None);
                };
                let negated = operator == CompareOperator::NotEq;
                match intersection {
                    Intersection::None => BitBufferMut::full(negated, len),
                    Intersection::All => BitBufferMut::full(!negated, len),
                    Intersection::At(idx) => {
                        let mut bits = BitBufferMut::full(negated, len);
                        bits.set_to(idx, !negated);
                        bits
                    }
                }
            }
            CompareOperator::Lt
            | CompareOperator::Lte
            | CompareOperator::Gt
            | CompareOperator::Gte => {
                // A sequence is monotonic, so the rows that satisfy an ordering comparison are
                // contiguous and touch one end of the array. See `ordered_run`.
                let Some(run) = ordered_run(lhs.base(), lhs.multiplier(), len, value, operator)
                else {
                    return Ok(None);
                };
                let mut bits = BitBufferMut::new_unset(len);
                match run {
                    TrueRun::Prefix(end) => bits.fill_range(0, end, true),
                    TrueRun::Suffix(start) => bits.fill_range(start, len, true),
                }
                bits
            }
        };

        let nullability = lhs.dtype().nullability() | rhs.dtype().nullability();
        let validity = match nullability {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => Validity::AllValid,
        };
        Ok(Some(BoolArray::new(bits.freeze(), validity).into_array()))
    }
}

/// The intersection of a sequence with a value.
pub(crate) enum Intersection {
    None,
    At(usize),
    All,
}

/// Finds where a sequence equals `value`, or returns `None` for non-integer inputs.
pub(crate) fn find_intersection(
    base: PValue,
    multiplier: PValue,
    len: usize,
    value: PValue,
) -> Option<Intersection> {
    if !value.ptype().is_int() || len == 0 {
        return (len == 0).then_some(Intersection::None);
    }
    let (ascending, magnitude) = eval::step_parts(multiplier)?;

    // Use the base's signedness to preserve values above `i64::MAX`.
    let (towards, offset) = if base.ptype().is_signed_int() {
        let base = base.cast::<i64>().vortex_expect("base fits its ptype");
        let Ok(value) = value.cast::<i64>() else {
            return Some(Intersection::None);
        };
        (value >= base, base.abs_diff(value))
    } else {
        let base = base.cast::<u64>().vortex_expect("base fits its ptype");
        let Ok(value) = value.cast::<u64>() else {
            return Some(Intersection::None);
        };
        (value >= base, base.abs_diff(value))
    };

    if offset == 0 {
        return Some(if magnitude == 0 {
            Intersection::All
        } else {
            Intersection::At(0)
        });
    }
    if magnitude == 0 || towards != ascending || offset % magnitude != 0 {
        return Some(Intersection::None);
    }

    Some(match usize::try_from(offset / magnitude) {
        Ok(idx) if idx < len => Intersection::At(idx),
        _ => Intersection::None,
    })
}

/// The rows for which an ordering comparison holds. A sequence is monotonic, so they form a
/// prefix or a suffix.
#[derive(Debug, PartialEq, Eq)]
enum TrueRun {
    /// Rows `0..end` hold.
    Prefix(usize),
    /// Rows `start..len` hold.
    Suffix(usize),
}

/// Finds the rows of a monotonic integer sequence for which `value[i] OP constant` holds.
///
/// Counts the rows that come before `constant` in the direction of the sequence. Each operator
/// uses that count as the end of a prefix or the start of a suffix. For example, `2, 5, 8, 11`
/// has two rows before `8`, so `< 8` holds on rows `0..2`.
///
/// The arithmetic runs in `i128`, so it is exact for all integer inputs. Returns `None` for a
/// non-integer step or constant.
fn ordered_run(
    base: PValue,
    multiplier: PValue,
    len: usize,
    constant: PValue,
    operator: CompareOperator,
) -> Option<TrueRun> {
    let base = base.as_i128()?;
    let step = multiplier.as_i128()?;
    let constant = constant.as_i128()?;

    let clamp = |rows: i128| usize::try_from(rows.clamp(0, len as i128)).vortex_expect("clamped");

    // All rows hold the same value, so the comparison is true for all rows or for none.
    if step == 0 {
        let holds = match operator {
            CompareOperator::Lt => base < constant,
            CompareOperator::Lte => base <= constant,
            CompareOperator::Gt => base > constant,
            CompareOperator::Gte => base >= constant,
            CompareOperator::Eq | CompareOperator::NotEq => {
                unreachable!("equality is answered by find_intersection")
            }
        };
        return Some(TrueRun::Prefix(if holds { len } else { 0 }));
    }

    let ascending = step > 0;
    let (distance, magnitude) = if ascending {
        (constant - base, step)
    } else {
        (base - constant, -step)
    };

    // `magnitude` is positive, so `div_euclid` rounds down and a non-zero remainder rounds up.
    let rows_strictly_before =
        clamp(distance.div_euclid(magnitude) + i128::from(distance.rem_euclid(magnitude) != 0));
    let rows_at_or_before = clamp(distance.div_euclid(magnitude) + 1);

    // When ascending, the rows before `constant` are smaller than it. When descending, they are
    // larger than it. The rows after are the complement, so they form the matching suffix.
    Some(match (ascending, operator) {
        (true, CompareOperator::Lt) => TrueRun::Prefix(rows_strictly_before),
        (true, CompareOperator::Lte) => TrueRun::Prefix(rows_at_or_before),
        (true, CompareOperator::Gt) => TrueRun::Suffix(rows_at_or_before),
        (true, CompareOperator::Gte) => TrueRun::Suffix(rows_strictly_before),
        (false, CompareOperator::Gt) => TrueRun::Prefix(rows_strictly_before),
        (false, CompareOperator::Gte) => TrueRun::Prefix(rows_at_or_before),
        (false, CompareOperator::Lt) => TrueRun::Suffix(rows_at_or_before),
        (false, CompareOperator::Lte) => TrueRun::Suffix(rows_strictly_before),
        (_, CompareOperator::Eq | CompareOperator::NotEq) => {
            unreachable!("equality is answered by find_intersection")
        }
    })
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::Nullability::Nullable;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::PValue;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::binary::CompareKernel;
    use vortex_array::scalar_fn::fns::operators::CompareOperator;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Sequence;
    use crate::SequenceArray;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    #[test]
    fn test_compare_match() {
        let lhs = Sequence::try_new_typed(2i64, 1, NonNullable, 4).unwrap();
        let rhs = ConstantArray::new(4i64, lhs.len());
        let result = lhs
            .into_array()
            .binary(rhs.into_array(), Operator::Eq)
            .unwrap();
        let expected = BoolArray::from_iter([false, false, true, false]);
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
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
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
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
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_compare_descending_unsigned() -> VortexResult<()> {
        let lhs = Sequence::try_new(
            PValue::from(100i32),
            PValue::from(-10i32),
            PType::U8,
            NonNullable,
            5,
        )?;
        let rhs = ConstantArray::new(80u8, lhs.len());
        let result = lhs.into_array().binary(rhs.into_array(), Operator::Eq)?;
        let expected = BoolArray::from_iter([false, false, true, false, false]);
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());

        Ok(())
    }

    #[test]
    fn test_compare_past_i64_max() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let step = (1u64 << 63) + 1;
        let lhs = Sequence::try_new(
            PValue::from(1u64 << 62),
            PValue::from(step),
            PType::U64,
            NonNullable,
            2,
        )?;

        let hit = lhs.clone().into_array().binary(
            ConstantArray::new((1u64 << 62) + step, lhs.len()).into_array(),
            Operator::Eq,
        )?;
        assert_arrays_eq!(hit, BoolArray::from_iter([false, true]), &mut ctx);

        let miss = lhs
            .into_array()
            .binary(ConstantArray::new(u64::MAX, 2).into_array(), Operator::Eq)?;
        assert_arrays_eq!(miss, BoolArray::from_iter([false, false]), &mut ctx);

        Ok(())
    }

    #[test]
    fn test_compare_constant_sequence() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = Sequence::try_new_typed(100i32, 0i32, NonNullable, 5)?;

        let matches = lhs.clone().into_array().binary(
            ConstantArray::new(100i32, lhs.len()).into_array(),
            Operator::Eq,
        )?;
        assert_arrays_eq!(matches, BoolArray::from_iter([true; 5]), &mut ctx);

        let misses = lhs
            .into_array()
            .binary(ConstantArray::new(7i32, 5).into_array(), Operator::Eq)?;
        assert_arrays_eq!(misses, BoolArray::from_iter([false; 5]), &mut ctx);

        Ok(())
    }

    const ALL_OPERATORS: [CompareOperator; 6] = [
        CompareOperator::Eq,
        CompareOperator::NotEq,
        CompareOperator::Lt,
        CompareOperator::Lte,
        CompareOperator::Gt,
        CompareOperator::Gte,
    ];

    /// Asserts the kernel engages for every operator and matches the comparison over the
    /// materialized values.
    fn assert_matches_primitive(
        sequence: &SequenceArray,
        constants: impl IntoIterator<Item = Scalar>,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = sequence
            .clone()
            .into_array()
            .execute::<PrimitiveArray>(&mut ctx)?;
        for constant in constants {
            let rhs = ConstantArray::new(constant, sequence.len()).into_array();
            for operator in ALL_OPERATORS {
                let actual = <Sequence as CompareKernel>::compare(
                    sequence.as_view(),
                    &rhs,
                    operator,
                    &mut ctx,
                )?
                .unwrap_or_else(|| panic!("sequence compare must engage for {operator:?}"));
                let expected = values
                    .clone()
                    .into_array()
                    .binary(rhs.clone(), Operator::from(operator))?;
                assert_arrays_eq!(actual, expected, &mut ctx);
            }
        }
        Ok(())
    }

    #[test]
    fn ascending_matches_primitive() -> VortexResult<()> {
        let sequence = Sequence::try_new_typed(2i64, 3, NonNullable, 10)?;
        let constants = [i64::MIN, -100, 1, 2, 3, 4, 17, 28, 29, 30, i64::MAX];
        assert_matches_primitive(&sequence, constants.map(Scalar::from))
    }

    #[test]
    fn descending_matches_primitive() -> VortexResult<()> {
        let sequence = Sequence::try_new_typed(100i32, -7, Nullable, 8)?;
        let constants = [i32::MIN, 0, 50, 51, 52, 79, 99, 100, 101, i32::MAX];
        assert_matches_primitive(&sequence, constants.map(Scalar::from))
    }

    #[test]
    fn constant_sequence_matches_primitive() -> VortexResult<()> {
        let sequence = Sequence::try_new_typed(42i32, 0, NonNullable, 5)?;
        assert_matches_primitive(&sequence, [41i32, 42, 43].map(Scalar::from))
    }

    #[test]
    fn single_row_matches_primitive() -> VortexResult<()> {
        let sequence = Sequence::try_new_typed(5i16, 4, NonNullable, 1)?;
        assert_matches_primitive(&sequence, [4i16, 5, 6].map(Scalar::from))
    }

    #[test]
    fn descending_u8_matches_primitive() -> VortexResult<()> {
        let sequence = Sequence::try_new(
            PValue::from(200i32),
            PValue::from(-3i32),
            PType::U8,
            NonNullable,
            60,
        )?;
        let constants = [0u8, 22, 23, 24, 25, 199, 200, 201, 255];
        assert_matches_primitive(&sequence, constants.map(Scalar::from))
    }

    #[test]
    fn past_i64_max_matches_primitive() -> VortexResult<()> {
        let step = (1u64 << 63) + 1;
        let sequence = Sequence::try_new(
            PValue::from(1u64 << 62),
            PValue::from(step),
            PType::U64,
            NonNullable,
            2,
        )?;
        let constants = [0u64, 1 << 62, (1 << 62) + 1, (1 << 62) + step, u64::MAX];
        assert_matches_primitive(&sequence, constants.map(Scalar::from))
    }
}
