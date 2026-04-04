// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::CheckedAdd;
use num_traits::CheckedMul;
use num_traits::CheckedSub;
use num_traits::Zero;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ScalarFnVTable;
use vortex_array::arrays::scalar_fn::AnyScalarFn;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::dtype::DType;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar::PValue;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_array::scalar_fn::fns::list_contains::ListContainsElementReduceAdaptor;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexResult;

use crate::Sequence;

pub(crate) static RULES: ParentRuleSet<Sequence> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(Sequence)),
    ParentRuleSet::lift(&SequenceAffineScalarFnRule),
    ParentRuleSet::lift(&ListContainsElementReduceAdaptor(Sequence)),
    ParentRuleSet::lift(&SliceReduceAdaptor(Sequence)),
]);

#[derive(Debug)]
struct SequenceAffineScalarFnRule;

impl ArrayParentReduceRule<Sequence> for SequenceAffineScalarFnRule {
    type Parent = AnyScalarFn;

    fn reduce_parent(
        &self,
        sequence: ArrayView<'_, Sequence>,
        parent: ArrayView<'_, ScalarFnVTable>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        if parent.nchildren() != 2 || child_idx > 1 {
            return Ok(None);
        }

        let Some(operator) = parent.scalar_fn().as_opt::<Binary>().copied() else {
            return Ok(None);
        };

        let DType::Primitive(result_ptype, nullability) = parent.dtype() else {
            return Ok(None);
        };
        if !result_ptype.is_int() {
            return Ok(None);
        }

        let Some(sibling) = parent.iter_children().nth(child_idx ^ 1) else {
            return Ok(None);
        };
        let Some(constant) = sibling.as_opt::<Constant>() else {
            return Ok(None);
        };
        let Some(constant_value) = constant
            .scalar()
            .as_primitive_opt()
            .and_then(|c| c.pvalue())
        else {
            return Ok(None);
        };

        let Some((base, multiplier)) = affine_sequence_parts(
            sequence.base(),
            sequence.multiplier(),
            constant_value,
            *result_ptype,
            operator,
            child_idx == 0,
        ) else {
            return Ok(None);
        };

        Ok(Sequence::try_new(
            base,
            multiplier,
            *result_ptype,
            *nullability,
            sequence.len(),
        )
        .ok()
        .map(|sequence| sequence.into_array()))
    }
}

fn affine_sequence_parts(
    base: PValue,
    multiplier: PValue,
    constant: PValue,
    result_ptype: PType,
    operator: Operator,
    sequence_on_lhs: bool,
) -> Option<(PValue, PValue)> {
    match_each_integer_ptype!(result_ptype, |P| {
        let base = base.cast::<P>().ok()?;
        let multiplier = multiplier.cast::<P>().ok()?;
        let constant = constant.cast::<P>().ok()?;

        affine_sequence_parts_typed(base, multiplier, constant, operator, sequence_on_lhs)
            .map(|(base, multiplier)| (PValue::from(base), PValue::from(multiplier)))
    })
}

fn affine_sequence_parts_typed<P>(
    base: P,
    multiplier: P,
    constant: P,
    operator: Operator,
    sequence_on_lhs: bool,
) -> Option<(P, P)>
where
    P: IntegerPType + CheckedAdd + CheckedSub + CheckedMul + Zero + Copy,
    PValue: From<P>,
{
    match (operator, sequence_on_lhs) {
        (Operator::Add, _) => Some((base.checked_add(&constant)?, multiplier)),
        (Operator::Sub, true) => Some((base.checked_sub(&constant)?, multiplier)),
        (Operator::Sub, false) => Some((
            constant.checked_sub(&base)?,
            P::zero().checked_sub(&multiplier)?,
        )),
        (Operator::Mul, _) => Some((
            base.checked_mul(&constant)?,
            multiplier.checked_mul(&constant)?,
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_array::scalar_fn::fns::binary::Binary;
    use vortex_array::scalar_fn::fns::operators::Operator;

    use crate::Sequence;
    use crate::SequenceArray;

    #[rstest]
    #[case::seq_plus_const(
        Operator::Add,
        true,
        Sequence::try_new_typed(12i64, 3, NonNullable, 5).unwrap(),
    )]
    #[case::const_plus_seq(
        Operator::Add,
        false,
        Sequence::try_new_typed(12i64, 3, NonNullable, 5).unwrap(),
    )]
    #[case::seq_minus_const(
        Operator::Sub,
        true,
        Sequence::try_new_typed(2i64, 3, NonNullable, 5).unwrap(),
    )]
    #[case::const_minus_seq(
        Operator::Sub,
        false,
        Sequence::try_new_typed(-2i64, -3, NonNullable, 5).unwrap(),
    )]
    #[case::seq_times_const(
        Operator::Mul,
        true,
        Sequence::try_new_typed(35i64, 15, NonNullable, 5).unwrap(),
    )]
    #[case::const_times_seq(
        Operator::Mul,
        false,
        Sequence::try_new_typed(35i64, 15, NonNullable, 5).unwrap(),
    )]
    fn rewrites_affine_binary_ops_to_sequence(
        #[case] operator: Operator,
        #[case] sequence_on_lhs: bool,
        #[case] expected: SequenceArray,
    ) {
        let sequence = Sequence::try_new_typed(7i64, 3, NonNullable, 5)
            .unwrap()
            .into_array();
        let constant = ConstantArray::new(5i64, sequence.len()).into_array();

        let optimized = optimize_binary(sequence, constant, operator, sequence_on_lhs);

        assert!(optimized.is::<Sequence>());
        assert_arrays_eq!(optimized, expected.into_array());
    }

    #[test]
    fn falls_back_for_overflow_prone_const_minus_seq() {
        let sequence = Sequence::try_new_typed(1i8, i8::MIN, NonNullable, 2)
            .unwrap()
            .into_array();
        let constant = ConstantArray::new(0i8, sequence.len()).into_array();

        let optimized = optimize_binary(sequence, constant, Operator::Sub, false);

        assert!(!optimized.is::<Sequence>());
        assert_arrays_eq!(
            optimized,
            PrimitiveArray::from_iter([-1i8, 127]).into_array()
        );
    }

    #[test]
    fn keeps_division_on_the_fallback_path() {
        let sequence = Sequence::try_new_typed(8i64, 4, NonNullable, 4)
            .unwrap()
            .into_array();
        let constant = ConstantArray::new(2i64, sequence.len()).into_array();

        let optimized = optimize_binary(sequence, constant, Operator::Div, true);

        assert!(!optimized.is::<Sequence>());
        assert_arrays_eq!(
            optimized,
            PrimitiveArray::from_iter([4i64, 6, 8, 10]).into_array()
        );
    }

    fn optimize_binary(
        sequence: ArrayRef,
        constant: ArrayRef,
        operator: Operator,
        sequence_on_lhs: bool,
    ) -> ArrayRef {
        let children = if sequence_on_lhs {
            vec![sequence, constant]
        } else {
            vec![constant, sequence]
        };

        Binary
            .try_new_array(children[0].len(), operator, children)
            .unwrap()
            .optimize()
            .unwrap()
    }
}
