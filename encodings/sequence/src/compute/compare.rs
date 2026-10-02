// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_pvalue;
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
            CompareOperator::Eq | CompareOperator::NotEq => {
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

pub(crate) enum Intersection {
    None,
    At(usize),
    All,
}

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

/// Solves `base + i * step OP value` for a monotonic integer sequence.
///
/// Returns `None` for a non-integer step or constant. The arithmetic runs in `i128`, so every
/// integer base, step, and constant is exact.
fn ordered_run(
    base: PValue,
    multiplier: PValue,
    len: usize,
    value: PValue,
    operator: CompareOperator,
) -> Option<TrueRun> {
    let base = integer_to_i128(base)?;
    let step = integer_to_i128(multiplier)?;
    let value = integer_to_i128(value)?;

    let clamp = |rows: i128| usize::try_from(rows.clamp(0, len as i128)).vortex_expect("clamped");

    if step == 0 {
        let holds = match operator {
            CompareOperator::Lt => base < value,
            CompareOperator::Lte => base <= value,
            CompareOperator::Gt => base > value,
            CompareOperator::Gte => base >= value,
            CompareOperator::Eq | CompareOperator::NotEq => {
                unreachable!("equality is answered by find_intersection")
            }
        };
        return Some(TrueRun::Prefix(if holds { len } else { 0 }));
    }

    // Count the rows whose value lies strictly before, and at or before, `value` along the
    // direction of travel. The rows before `value` are a prefix when ascending and a suffix
    // when descending.
    let (distance, magnitude) = if step > 0 {
        (value - base, step)
    } else {
        (base - value, -step)
    };
    let strictly_before =
        clamp(distance.div_euclid(magnitude) + i128::from(distance.rem_euclid(magnitude) != 0));
    let at_or_before = clamp(distance.div_euclid(magnitude) + 1);

    Some(match (step > 0, operator) {
        (true, CompareOperator::Lt) => TrueRun::Prefix(strictly_before),
        (true, CompareOperator::Lte) => TrueRun::Prefix(at_or_before),
        (true, CompareOperator::Gt) => TrueRun::Suffix(at_or_before),
        (true, CompareOperator::Gte) => TrueRun::Suffix(strictly_before),
        (false, CompareOperator::Gt) => TrueRun::Prefix(strictly_before),
        (false, CompareOperator::Gte) => TrueRun::Prefix(at_or_before),
        (false, CompareOperator::Lt) => TrueRun::Suffix(at_or_before),
        (false, CompareOperator::Lte) => TrueRun::Suffix(strictly_before),
        (_, CompareOperator::Eq | CompareOperator::NotEq) => {
            unreachable!("equality is answered by find_intersection")
        }
    })
}

fn integer_to_i128(value: PValue) -> Option<i128> {
    match_each_pvalue!(
        value,
        uint: |v| {
            let v: u64 = v.as_();
            Some(i128::from(v))
        },
        int: |v| {
            let v: i64 = v.as_();
            Some(i128::from(v))
        },
        float: |_v| { None }
    )
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
        let session = vortex_array::array_session();
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
