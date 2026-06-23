// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar_fn::fns::binary::BooleanKernel;
use vortex_array::scalar_fn::fns::binary::kleene_boolean_buffer_scalar;
use vortex_array::scalar_fn::fns::binary::kleene_boolean_buffers;
use vortex_array::scalar_fn::fns::cast::CastKernel;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_array::scalar_fn::fns::mask::MaskReduce;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::ByteBool;

impl CastReduce for ByteBool {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // ByteBool is essentially a bool array stored as bytes
        // The main difference from BoolArray is the storage format
        // For casting, we can decode to canonical (BoolArray) and let it handle the cast
        // If just changing nullability, we can optimize
        if !dtype.is_boolean() {
            return Ok(None);
        }

        let Some(new_validity) = array
            .validity()?
            .trivially_cast_nullability(dtype.nullability(), array.len())?
        else {
            return Ok(None);
        };

        Ok(Some(
            ByteBool::new(array.buffer().clone(), new_validity).into_array(),
        ))
    }
}

impl CastKernel for ByteBool {
    fn cast(
        array: ArrayView<'_, Self>,
        dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only handle nullability changes here; non-bool targets fall through to canonicalization.
        if !dtype.is_boolean() {
            return Ok(None);
        }

        let new_validity =
            array
                .validity()?
                .cast_nullability(dtype.nullability(), array.len(), ctx)?;

        Ok(Some(
            ByteBool::new(array.buffer().clone(), new_validity).into_array(),
        ))
    }
}

impl MaskReduce for ByteBool {
    fn mask(array: ArrayView<'_, Self>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            ByteBool::new(
                array.buffer().clone(),
                array.validity()?.and(Validity::Array(mask.clone()))?,
            )
            .into_array(),
        ))
    }
}

impl TakeExecute for ByteBool {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let values = array.truthy_bytes();

        // This handles combining validity from both source array and nullable indices
        let validity = array.validity()?.take(&indices.clone().into_array())?;

        let taken = match_each_integer_ptype!(indices.ptype(), |I| {
            indices
                .as_slice::<I>()
                .iter()
                .map(|&idx| {
                    let idx: usize = idx.as_();
                    values[idx]
                })
                .collect::<ByteBuffer>()
        });

        Ok(Some(
            ByteBool::new(BufferHandle::new_host(taken), validity).into_array(),
        ))
    }
}

impl BooleanKernel for ByteBool {
    fn boolean(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: Operator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let nullability = lhs.dtype().nullability() | rhs.dtype().nullability();
        let lhs_values = truthy_bit_buffer(lhs);

        if let Some(rhs) = rhs.as_opt::<Constant>() {
            let rhs = rhs
                .scalar()
                .as_bool_opt()
                .ok_or_else(|| vortex_err!("expected boolean scalar"))?;
            return kleene_boolean_buffer_scalar(
                lhs_values,
                lhs.validity()?,
                &rhs,
                operator,
                nullability,
                ctx,
            )
            .map(Some);
        }

        let Some(rhs) = rhs.as_opt::<ByteBool>() else {
            return Ok(None);
        };

        kleene_boolean_buffers(
            lhs_values,
            lhs.validity()?,
            truthy_bit_buffer(rhs),
            rhs.validity()?,
            operator,
            nullability,
            ctx,
        )
        .map(Some)
    }
}

fn truthy_bit_buffer(array: ArrayView<'_, ByteBool>) -> BitBuffer {
    let bytes = array.truthy_bytes();
    BitBuffer::collect_bool(bytes.len(), |idx| bytes[idx] != 0)
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::BoolArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::compute::conformance::mask::test_mask_conformance;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use super::*;
    use crate::ByteBoolArray;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = array_session();
        crate::initialize(&session);
        session
    });

    fn bb(v: Vec<bool>) -> ByteBoolArray {
        ByteBool::from_vec(v, Validity::AllValid)
    }

    fn bb_opt(v: Vec<Option<bool>>) -> ByteBoolArray {
        ByteBool::from_option_vec(v)
    }

    #[test]
    fn test_slice() {
        let original = vec![Some(true), Some(true), None, Some(false), None];
        let vortex_arr = bb_opt(original);

        let sliced_arr = vortex_arr.slice(1..4).unwrap();

        let expected = bb_opt(vec![Some(true), None, Some(false)]);
        assert_arrays_eq!(
            sliced_arr,
            expected.into_array(),
            &mut SESSION.create_execution_ctx()
        );
    }

    #[test]
    fn test_compare_all_equal() {
        let lhs = bb(vec![true; 5]);
        let rhs = bb(vec![true; 5]);

        let arr = lhs
            .into_array()
            .binary(rhs.into_array(), Operator::Eq)
            .unwrap();

        let expected = bb(vec![true; 5]);
        assert_arrays_eq!(
            arr,
            expected.into_array(),
            &mut SESSION.create_execution_ctx()
        );
    }

    #[test]
    fn test_compare_all_different() {
        let lhs = bb(vec![false; 5]);
        let rhs = bb(vec![true; 5]);

        let arr = lhs
            .into_array()
            .binary(rhs.into_array(), Operator::Eq)
            .unwrap();

        let expected = bb(vec![false; 5]);
        assert_arrays_eq!(
            arr,
            expected.into_array(),
            &mut SESSION.create_execution_ctx()
        );
    }

    #[test]
    fn test_compare_with_nulls() {
        let lhs = bb(vec![true; 5]);
        let rhs = bb_opt(vec![Some(true), Some(true), Some(true), Some(false), None]);

        let arr = lhs
            .into_array()
            .binary(rhs.into_array(), Operator::Eq)
            .unwrap();

        let expected = bb_opt(vec![Some(true), Some(true), Some(true), Some(false), None]);
        assert_arrays_eq!(
            arr,
            expected.into_array(),
            &mut SESSION.create_execution_ctx()
        );
    }

    #[test]
    fn test_boolean_kernel_kleene() -> VortexResult<()> {
        let lhs = bb_opt(vec![Some(false), Some(true), None, Some(false), None]);
        let rhs = bb_opt(vec![None, None, Some(true), Some(false), None]).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let and_result =
            <ByteBool as BooleanKernel>::boolean(lhs.as_view(), &rhs, Operator::And, &mut ctx)?
                .ok_or_else(|| vortex_err!("ByteBool should handle ByteBool boolean AND"))?;
        assert_arrays_eq!(
            and_result,
            BoolArray::from_iter([Some(false), None, None, Some(false), None]),
            &mut ctx
        );

        let or_result =
            <ByteBool as BooleanKernel>::boolean(lhs.as_view(), &rhs, Operator::Or, &mut ctx)?
                .ok_or_else(|| vortex_err!("ByteBool should handle ByteBool boolean OR"))?;
        assert_arrays_eq!(
            or_result,
            BoolArray::from_iter([None, Some(true), Some(true), Some(false), None]),
            &mut ctx
        );

        Ok(())
    }

    #[test]
    fn test_mask_byte_bool() {
        test_mask_conformance(&bb(vec![true, false, true, true, false]).into_array());
        test_mask_conformance(
            &bb_opt(vec![Some(true), Some(true), None, Some(false), None]).into_array(),
        );
    }

    #[test]
    fn test_filter_byte_bool() {
        test_filter_conformance(&bb(vec![true, false, true, true, false]).into_array());
        test_filter_conformance(
            &bb_opt(vec![Some(true), Some(true), None, Some(false), None]).into_array(),
        );
    }

    #[rstest]
    #[case(bb(vec![true, false, true, true, false]))]
    #[case(bb_opt(vec![Some(true), Some(true), None, Some(false), None]))]
    #[case(bb(vec![true, false]))]
    #[case(bb(vec![true]))]
    fn test_take_byte_bool_conformance(#[case] array: ByteBoolArray) {
        test_take_conformance(&array.into_array());
    }

    #[test]
    fn test_cast_bytebool_to_nullable() {
        let array = bb(vec![true, false, true, false]);
        let casted = array
            .into_array()
            .cast(DType::Bool(Nullability::Nullable))
            .unwrap();
        assert_eq!(casted.dtype(), &DType::Bool(Nullability::Nullable));
        assert_eq!(casted.len(), 4);
    }

    #[rstest]
    #[case(bb(vec![true, false, true, true, false]))]
    #[case(bb_opt(vec![Some(true), Some(false), None, Some(true), None]))]
    #[case(bb(vec![false]))]
    #[case(bb(vec![true]))]
    #[case(bb_opt(vec![Some(true), None]))]
    fn test_cast_bytebool_conformance(#[case] array: ByteBoolArray) {
        test_cast_conformance(&array.into_array());
    }

    #[rstest]
    #[case::non_nullable(bb(vec![true, false, true, true, false]))]
    #[case::nullable(bb_opt(vec![Some(true), Some(false), None, Some(true), None]))]
    #[case::all_true(bb(vec![true, true, true, true]))]
    #[case::all_false(bb(vec![false, false, false, false]))]
    #[case::single_true(bb(vec![true]))]
    #[case::single_false(bb(vec![false]))]
    #[case::single_null(bb_opt(vec![None]))]
    #[case::mixed_with_nulls(bb_opt(vec![Some(true), None, Some(false), None, Some(true)]))]
    fn test_bytebool_consistency(#[case] array: ByteBoolArray) {
        let ctx = &mut array_session().create_execution_ctx();
        test_array_consistency(&array.into_array(), ctx);
    }
}
