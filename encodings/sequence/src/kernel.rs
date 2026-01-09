// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VectorExecutor;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ConstantVTable;
use vortex_array::arrays::ExactScalarFn;
use vortex_array::arrays::ScalarFnArrayView;
use vortex_array::arrays::ScalarFnVTable;
use vortex_array::compute::Operator;
use vortex_array::expr::Binary;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::kernel::ParentKernelSet;
use vortex_buffer::buffer;
use vortex_dtype::DType;
use vortex_dtype::NativePType;
use vortex_dtype::Nullability;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_runend::RunEndArray;
use vortex_scalar::PValue;
use vortex_scalar::Scalar;
use vortex_vector::Vector;

use crate::SequenceArray;
use crate::SequenceVTable;
use crate::compute::compare::find_intersection_scalar;

pub(crate) const PARENT_KERNELS: ParentKernelSet<SequenceVTable> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&SequenceCompareKernel)]);

/// Kernel to execute comparison operations directly on a sequence array.
#[derive(Debug)]
struct SequenceCompareKernel;

impl ExecuteParentKernel<SequenceVTable> for SequenceCompareKernel {
    type Parent = ExactScalarFn<Binary>;

    fn parent(&self) -> Self::Parent {
        ExactScalarFn::from(&Binary)
    }

    fn execute_parent(
        &self,
        array: &SequenceArray,
        parent: ScalarFnArrayView<'_, Binary>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Vector>> {
        // Only handle comparison operators
        let Some(cmp_op) = parent.options.maybe_cmp_operator() else {
            return Ok(None);
        };

        // Get the ScalarFnArray to access children
        let Some(scalar_fn_array) = parent.as_opt::<ScalarFnVTable>() else {
            return Ok(None);
        };
        let children = scalar_fn_array.children();

        // Determine which operand is the constant and which is the sequence
        let (cmp_op, constant) = match child_idx {
            0 => {
                // sequence is lhs, check if rhs is constant
                let rhs = &children[1];
                let Some(constant) = rhs.as_opt::<ConstantVTable>() else {
                    return Ok(None);
                };
                (cmp_op, constant)
            }
            1 => {
                // sequence is rhs, swap the operator and check if lhs is constant
                let lhs = &children[0];
                let Some(constant) = lhs.as_opt::<ConstantVTable>() else {
                    return Ok(None);
                };
                // Swap the operator since we're reversing operand order
                (cmp_op.swap(), constant)
            }
            _ => return Ok(None),
        };

        let constant_pvalue = constant.scalar().as_primitive().pvalue();
        let Some(constant_pvalue) = constant_pvalue else {
            // Constant is null - result is all null for comparisons
            let nullability = array.dtype().nullability() | constant.dtype().nullability();
            let result_array =
                ConstantArray::new(Scalar::null(DType::Bool(nullability)), array.length).to_array();
            return Ok(Some(result_array.execute(ctx)?));
        };

        let nullability = array.dtype().nullability() | constant.dtype().nullability();

        // For Eq and NotEq, use specialized logic
        if cmp_op == Operator::Eq {
            return compare_eq_neq(array, constant_pvalue, nullability, false, ctx);
        }
        if cmp_op == Operator::NotEq {
            return compare_eq_neq(array, constant_pvalue, nullability, true, ctx);
        }

        // For ordering comparisons, find the transition point
        compare_ordering(array, constant_pvalue, cmp_op, nullability, ctx)
    }
}

/// Compare sequence to constant for equality/inequality.
/// When `negate` is false, returns true where sequence == constant.
/// When `negate` is true, returns true where sequence != constant.
fn compare_eq_neq(
    array: &SequenceArray,
    constant: PValue,
    nullability: Nullability,
    negate: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Vector>> {
    // For Eq: match_val=true, default_val=false
    // For NotEq: match_val=false, default_val=true
    let match_val = !negate;
    let not_match_val = negate;

    // Check if there exists an integer solution to const = base + idx * multiplier
    let Some(set_idx) =
        find_intersection_scalar(array.base(), array.multiplier(), array.length, constant)
    else {
        let result_array = ConstantArray::new(
            Scalar::new(DType::Bool(nullability), not_match_val.into()),
            array.length,
        )
        .to_array();
        return Ok(Some(result_array.execute(ctx)?));
    };
    let idx = set_idx as u64;
    let len = array.length as u64;

    if len == 1 && set_idx == 0 {
        let result_array = ConstantArray::new(
            Scalar::new(DType::Bool(nullability), match_val.into()),
            array.length,
        )
        .to_array();
        return Ok(Some(result_array.execute(ctx)?));
    }

    let (ends, values) = if idx == 0 {
        let ends = buffer![1u64, len].into_array();
        let values = BoolArray::from_iter([match_val, not_match_val]).into_array();
        (ends, values)
    } else if idx == len - 1 {
        let ends = buffer![idx, len].into_array();
        let values = BoolArray::from_iter([not_match_val, match_val]).into_array();
        (ends, values)
    } else {
        let ends = buffer![idx, idx + 1, len].into_array();
        let values = BoolArray::from_iter([not_match_val, match_val, not_match_val]).into_array();
        (ends, values)
    };
    let result_array = RunEndArray::try_new(ends, values)?.into_array();
    Ok(Some(result_array.execute(ctx)?))
}

fn compare_ordering(
    array: &SequenceArray,
    constant: PValue,
    operator: Operator,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Vector>> {
    let transition = find_transition_point(
        array.base(),
        array.multiplier(),
        array.length,
        constant,
        operator,
    );

    let result_array = match transition {
        Transition::AllTrue => ConstantArray::new(
            Scalar::new(DType::Bool(nullability), true.into()),
            array.length,
        )
        .to_array(),
        Transition::AllFalse => ConstantArray::new(
            Scalar::new(DType::Bool(nullability), false.into()),
            array.length,
        )
        .to_array(),
        Transition::FalseToTrue(idx) => {
            // [0..idx) is false, [idx..len) is true
            let ends = buffer![idx as u64, array.length as u64].into_array();
            let values = BoolArray::from_iter([false, true]).into_array();
            RunEndArray::try_new(ends, values)?.into_array()
        }
        Transition::TrueToFalse(idx) => {
            // [0..idx) is true, [idx..len) is false
            let ends = buffer![idx as u64, array.length as u64].into_array();
            let values = BoolArray::from_iter([true, false]).into_array();
            RunEndArray::try_new(ends, values)?.into_array()
        }
    };

    Ok(Some(result_array.execute(ctx)?))
}

enum Transition {
    AllTrue,
    AllFalse,
    FalseToTrue(usize),
    TrueToFalse(usize),
}

fn find_transition_point(
    base: PValue,
    multiplier: PValue,
    len: usize,
    constant: PValue,
    operator: Operator,
) -> Transition {
    match_each_integer_ptype!(base.ptype(), |P| {
        find_transition_point_typed::<P>(
            base.cast::<P>(),
            multiplier.cast::<P>(),
            len,
            constant.cast::<P>(),
            operator,
        )
    })
}

fn find_transition_point_typed<P: NativePType>(
    base: P,
    multiplier: P,
    len: usize,
    constant: P,
    operator: Operator,
) -> Transition {
    if len == 0 {
        return Transition::AllFalse;
    }

    let last_idx = P::from_usize(len - 1).vortex_expect("len must fit into type");
    let first_value = base;
    let last_value = base + multiplier * last_idx;

    let first_result = eval_comparison(first_value, constant, operator);
    let last_result = eval_comparison(last_value, constant, operator);

    if first_result && last_result {
        return Transition::AllTrue;
    }
    if !first_result && !last_result {
        return Transition::AllFalse;
    }

    // There's a transition point - find it using binary search
    let transition_idx = binary_search_transition(base, multiplier, len, constant, operator);

    if first_result {
        Transition::TrueToFalse(transition_idx)
    } else {
        Transition::FalseToTrue(transition_idx)
    }
}

fn eval_comparison<P: NativePType>(lhs: P, rhs: P, operator: Operator) -> bool {
    match operator {
        Operator::Lt => lhs.is_lt(rhs),
        Operator::Lte => lhs.is_le(rhs),
        Operator::Gt => lhs.is_gt(rhs),
        Operator::Gte => lhs.is_ge(rhs),
        Operator::Eq => lhs.is_eq(rhs),
        Operator::NotEq => !lhs.is_eq(rhs),
    }
}

fn binary_search_transition<P: NativePType>(
    base: P,
    multiplier: P,
    len: usize,
    constant: P,
    operator: Operator,
) -> usize {
    let first_result = eval_comparison(base, constant, operator);

    let mut lo = 0usize;
    let mut hi = len;

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let mid_p = P::from_usize(mid).vortex_expect("idx must fit into type");
        let value = base + multiplier * mid_p;
        let result = eval_comparison(value, constant, operator);

        if result == first_result {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    lo
}

#[cfg(test)]
mod tests {
    use vortex_array::VectorExecutor;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::ScalarFnArrayExt;
    use vortex_array::expr::Binary;
    use vortex_array::expr::Operator as ExprOperator;
    use vortex_buffer::BitBuffer;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::SequenceArray;

    #[test]
    fn test_sequence_eq_neq_constant() -> VortexResult<()> {
        let len = 1;
        let session = VortexSession::empty();
        let seq = SequenceArray::typed_new(5i64, 1, NonNullable, len)?.to_array();
        let constant = ConstantArray::new(5i64, len).to_array();

        let compare_array =
            Binary.try_new_array(len, ExprOperator::NotEq, [seq.clone(), constant.clone()])?;

        let result = compare_array.execute_vector(&session)?;
        let bool_result = result.into_bool();

        let expected = BitBuffer::from(vec![false]);
        assert_eq!(bool_result.bits(), &expected);

        let compare_array = Binary.try_new_array(len, ExprOperator::Eq, [seq, constant])?;

        let result = compare_array.execute_vector(&session)?;
        let bool_result = result.into_bool();

        let expected = BitBuffer::from(vec![true]);
        assert_eq!(bool_result.bits(), &expected);
        Ok(())
    }

    #[test]
    fn test_sequence_gte_constant() -> VortexResult<()> {
        let session = VortexSession::empty();
        let seq = SequenceArray::typed_new(0i64, 1, NonNullable, 10)?.to_array();
        let constant = ConstantArray::new(5i64, 10).to_array();

        let compare_array = Binary.try_new_array(10, ExprOperator::Gte, [seq, constant])?;

        let result = compare_array.execute_vector(&session)?;
        let bool_result = result.into_bool();

        let expected = BitBuffer::from(vec![
            false, false, false, false, false, true, true, true, true, true,
        ]);
        assert_eq!(bool_result.bits(), &expected);
        Ok(())
    }

    #[test]
    fn test_sequence_lt_constant() -> VortexResult<()> {
        let session = VortexSession::empty();
        let seq = SequenceArray::typed_new(0i64, 1, NonNullable, 10)?.to_array();
        let constant = ConstantArray::new(5i64, 10).to_array();

        let compare_array = Binary.try_new_array(10, ExprOperator::Lt, [seq, constant])?;

        let result = compare_array.execute_vector(&session)?;
        let bool_result = result.into_bool();

        let expected = BitBuffer::from(vec![
            true, true, true, true, true, false, false, false, false, false,
        ]);
        assert_eq!(bool_result.bits(), &expected);
        Ok(())
    }

    #[test]
    fn test_sequence_lte_constant() -> VortexResult<()> {
        let session = VortexSession::empty();
        let seq = SequenceArray::typed_new(0i64, 1, NonNullable, 10)?.to_array();
        let constant = ConstantArray::new(5i64, 10).to_array();

        let compare_array = Binary.try_new_array(10, ExprOperator::Lte, [seq, constant])?;

        let result = compare_array.execute_vector(&session)?;
        let bool_result = result.into_bool();

        // [0,1,2,3,4,5,6,7,8,9] <= 5
        let expected = BitBuffer::from(vec![
            true, true, true, true, true, true, false, false, false, false,
        ]);
        assert_eq!(bool_result.bits(), &expected);
        Ok(())
    }

    #[test]
    fn test_sequence_gt_constant() -> VortexResult<()> {
        let session = VortexSession::empty();
        let seq = SequenceArray::typed_new(0i64, 1, NonNullable, 10)?.to_array();
        let constant = ConstantArray::new(5i64, 10).to_array();

        let compare_array = Binary.try_new_array(10, ExprOperator::Gt, [seq, constant])?;

        let result = compare_array.execute_vector(&session)?;
        let bool_result = result.into_bool();

        // [0,1,2,3,4,5,6,7,8,9] > 5
        let expected = BitBuffer::from(vec![
            false, false, false, false, false, false, true, true, true, true,
        ]);
        assert_eq!(bool_result.bits(), &expected);
        Ok(())
    }

    #[test]
    fn test_constant_gte_sequence() -> VortexResult<()> {
        // Test when constant is on the left side
        let session = VortexSession::empty();
        let constant = ConstantArray::new(5i64, 10).to_array();
        let seq = SequenceArray::typed_new(0i64, 1, NonNullable, 10)?.to_array();

        let compare_array = Binary.try_new_array(10, ExprOperator::Gte, [constant, seq])?;

        let result = compare_array.execute_vector(&session)?;
        let bool_result = result.into_bool();

        // 5 >= [0,1,2,3,4,5,6,7,8,9]
        let expected = BitBuffer::from(vec![
            true, true, true, true, true, true, false, false, false, false,
        ]);
        assert_eq!(bool_result.bits(), &expected);
        Ok(())
    }

    #[test]
    fn test_sequence_eq_constant() -> VortexResult<()> {
        let session = VortexSession::empty();
        let seq = SequenceArray::typed_new(0i64, 1, NonNullable, 10)?.to_array();
        let constant = ConstantArray::new(5i64, 10).to_array();

        let compare_array = Binary.try_new_array(10, ExprOperator::Eq, [seq, constant])?;

        let result = compare_array.execute_vector(&session)?;
        let bool_result = result.into_bool();

        let expected = BitBuffer::from(vec![
            false, false, false, false, false, true, false, false, false, false,
        ]);
        assert_eq!(bool_result.bits(), &expected);
        Ok(())
    }

    #[test]
    fn test_sequence_not_eq_constant() -> VortexResult<()> {
        let session = VortexSession::empty();
        let seq = SequenceArray::typed_new(0i64, 1, NonNullable, 10)?.to_array();
        let constant = ConstantArray::new(5i64, 10).to_array();

        let compare_array = Binary.try_new_array(10, ExprOperator::NotEq, [seq, constant])?;

        let result = compare_array.execute_vector(&session)?;
        let bool_result = result.into_bool();

        let expected = BitBuffer::from(vec![
            true, true, true, true, true, false, true, true, true, true,
        ]);
        assert_eq!(bool_result.bits(), &expected);
        Ok(())
    }

    #[test]
    fn test_sequence_all_true() -> VortexResult<()> {
        let session = VortexSession::empty();
        let seq = SequenceArray::typed_new(10i64, 1, NonNullable, 5)?.to_array();
        let constant = ConstantArray::new(5i64, 5).to_array();

        let compare_array = Binary.try_new_array(5, ExprOperator::Gt, [seq, constant])?;

        let result = compare_array.execute_vector(&session)?;
        let bool_result = result.into_bool();

        let expected = BitBuffer::from(vec![true, true, true, true, true]);
        assert_eq!(bool_result.bits(), &expected);
        Ok(())
    }

    #[test]
    fn test_sequence_all_false() -> VortexResult<()> {
        let session = VortexSession::empty();
        let seq = SequenceArray::typed_new(0i64, 1, NonNullable, 5)?.to_array();
        let constant = ConstantArray::new(100i64, 5).to_array();

        let compare_array = Binary.try_new_array(5, ExprOperator::Gt, [seq, constant])?;

        let result = compare_array.execute_vector(&session)?;
        let bool_result = result.into_bool();

        let expected = BitBuffer::from(vec![false, false, false, false, false]);
        assert_eq!(bool_result.bits(), &expected);
        Ok(())
    }

    #[test]
    fn test_sequence_multiplier_2_gte() -> VortexResult<()> {
        // Sequence: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        let session = VortexSession::empty();
        let seq = SequenceArray::typed_new(0i64, 2, NonNullable, 10)?.to_array();
        let constant = ConstantArray::new(10i64, 10).to_array();

        let compare_array = Binary.try_new_array(10, ExprOperator::Gte, [seq, constant])?;

        let result = compare_array.execute_vector(&session)?;
        let bool_result = result.into_bool();

        // [0, 2, 4, 6, 8, 10, 12, 14, 16, 18] >= 10
        let expected = BitBuffer::from(vec![
            false, false, false, false, false, true, true, true, true, true,
        ]);
        assert_eq!(bool_result.bits(), &expected);
        Ok(())
    }

    #[test]
    fn test_sequence_multiplier_3_eq() -> VortexResult<()> {
        // Sequence: [5, 8, 11, 14, 17, 20, 23, 26]
        let session = VortexSession::empty();
        let seq = SequenceArray::typed_new(5i64, 3, NonNullable, 8)?.to_array();
        let constant = ConstantArray::new(14i64, 8).to_array();

        let compare_array = Binary.try_new_array(8, ExprOperator::Eq, [seq, constant])?;

        let result = compare_array.execute_vector(&session)?;
        let bool_result = result.into_bool();

        // 14 is at index 3: (14 - 5) / 3 = 3
        let expected = BitBuffer::from(vec![false, false, false, true, false, false, false, false]);
        assert_eq!(bool_result.bits(), &expected);
        Ok(())
    }

    #[test]
    fn test_sequence_negative_multiplier_lt() -> VortexResult<()> {
        // Sequence: [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        let session = VortexSession::empty();
        let seq = SequenceArray::typed_new(100i64, -10, NonNullable, 10)?.to_array();
        let constant = ConstantArray::new(50i64, 10).to_array();

        let compare_array = Binary.try_new_array(10, ExprOperator::Lt, [seq, constant])?;

        let result = compare_array.execute_vector(&session)?;
        let bool_result = result.into_bool();

        // [100, 90, 80, 70, 60, 50, 40, 30, 20, 10] < 50
        let expected = BitBuffer::from(vec![
            false, false, false, false, false, false, true, true, true, true,
        ]);
        assert_eq!(bool_result.bits(), &expected);
        Ok(())
    }
}
