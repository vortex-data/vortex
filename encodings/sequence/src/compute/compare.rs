// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::arith;
use crate::array::Sequence;

impl CompareKernel for Sequence {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // TODO(joe): support other operators (NotEq, Lt, Lte, Gt, Gte) in encoded space.
        if operator != CompareOperator::Eq {
            return Ok(None);
        }

        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };

        let value = constant
            .as_primitive()
            .pvalue()
            .vortex_expect("null constant handled in adaptor");

        // A non-integer constant, say a float compared against an integer sequence, is left to
        // the canonical compare.
        let Some(intersection) = find_intersection(lhs.base(), lhs.multiplier(), lhs.len(), value)
        else {
            return Ok(None);
        };

        let nullability = lhs.dtype().nullability() | rhs.dtype().nullability();
        let validity = match nullability {
            Nullability::NonNullable => vortex_array::validity::Validity::NonNullable,
            Nullability::Nullable => vortex_array::validity::Validity::AllValid,
        };

        let array = match intersection {
            Intersection::None => {
                ConstantArray::new(Scalar::bool(false, nullability), lhs.len()).into_array()
            }
            Intersection::All => {
                ConstantArray::new(Scalar::bool(true, nullability), lhs.len()).into_array()
            }
            Intersection::At(idx) => {
                let mut buffer = BitBufferMut::new_unset(lhs.len());
                buffer.set(idx);
                BoolArray::new(buffer.freeze(), validity).into_array()
            }
        };

        Ok(Some(array))
    }
}

/// Which elements of a sequence equal a given value.
pub(crate) enum Intersection {
    /// No element of the sequence equals the value.
    None,
    /// Only the element at this index equals the value.
    At(usize),
    /// Every element equals the value, i.e. the sequence is constant.
    All,
}

/// Finds the elements of `base + i * multiplier`, `i` in `0..len`, that equal `value`.
///
/// Returns `None` if any of the values is not an integer, when the caller has to fall back to
/// comparing the sequence's values.
pub(crate) fn find_intersection(
    base: PValue,
    multiplier: PValue,
    len: usize,
    value: PValue,
) -> Option<Intersection> {
    if !value.ptype().is_int() || len == 0 {
        return (len == 0).then_some(Intersection::None);
    }
    let (ascending, magnitude) = arith::step_parts(multiplier)?;

    // Work in the domain of `base`'s ptype signedness, which every sequence value fits: `u64`
    // for unsigned ptypes - keeping values above `i64::MAX` exact - and `i64` for signed ones.
    // A `value` outside that domain cannot equal any sequence value.
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
        // `value` is the first element, which a non-constant sequence never revisits.
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

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::Nullability::Nullable;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::PValue;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Sequence;

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

    /// A step that is not representable in the output ptype still intersects exactly.
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

    /// A base and step above `i64::MAX` intersect exactly, without being routed through `i64`.
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

    /// Every element of a constant sequence equals its base.
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
}
