// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::IntoArray;
use crate::ProbeState;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::ConstantArray;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::scalar_fn::vtable::ScalarFn;
use crate::columnar::Columnar;
use crate::scalar::Scalar;
use crate::scalar_fn::VecExecutionArgs;

impl OperationsVTable<ScalarFn> for ScalarFn {
    type ProbeState = ();

    /// The evaluated scalar carries the row's nullness, so no separate validity check is needed.
    fn probe_scalar(
        state: &mut ProbeState<'_, ScalarFn>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        let mut inputs = Vec::with_capacity(array.child_count());
        for slot in 0..array.child_count() {
            let mut child = state
                .slot(slot)?
                .vortex_expect("ScalarFnArray child slot is present");
            inputs.push(ConstantArray::new(child.execute_scalar(index, ctx)?, 1).into_array());
        }

        let args = VecExecutionArgs::new(inputs, 1);
        let result = array.scalar_fn().execute(&args, ctx)?;

        let scalar = match result.execute::<Columnar>(ctx)? {
            Columnar::Canonical(arr) => {
                tracing::info!(
                    "Scalar function {} returned non-constant array from execution over all scalar inputs",
                    array.scalar_fn(),
                );
                arr.into_array().execute_scalar(0, ctx)?
            }
            Columnar::Constant(constant) => constant.scalar().clone(),
        };

        debug_assert_eq!(
            scalar.dtype(),
            array.dtype(),
            "Scalar function {} returned dtype {:?} but expected {:?}",
            array.scalar_fn(),
            scalar.dtype(),
            array.dtype()
        );

        Ok(scalar)
    }

    fn scalar_at(
        array: ArrayView<'_, ScalarFn>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::ArraySlots;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array::Array;
    use crate::array::ArrayParts;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::ScalarFnArray;
    use crate::arrays::scalar_fn::ScalarFnArrayExt;
    use crate::arrays::scalar_fn::array::ScalarFnData;
    use crate::arrays::scalar_fn::vtable::ScalarFn;
    use crate::assert_arrays_eq;
    use crate::compute::conformance::consistency::test_scalar_probe_consistency;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::scalar::Scalar;
    use crate::scalar_fn::TypedScalarFnInstance;
    use crate::scalar_fn::fns::binary::Binary;
    use crate::scalar_fn::fns::list_contains::PreparedSet;
    use crate::scalar_fn::fns::list_contains::PreparedSetData;
    use crate::scalar_fn::fns::literal::Literal;
    use crate::scalar_fn::fns::operators::Operator;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn test_scalar_fn_add() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = buffer![1i32, 2, 3].into_array();
        let rhs = buffer![10i32, 20, 30].into_array();

        let scalar_fn = TypedScalarFnInstance::new(Binary, Operator::Add).erased();
        let scalar_fn_array = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs])?;

        assert_eq!(scalar_fn_array.len(), 3);

        let expected = buffer![11i32, 22, 33].into_array();
        assert_arrays_eq!(scalar_fn_array, expected, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_scalar_fn_inferred_len_rejects_mismatched_children() {
        let lhs = buffer![1i32, 2, 3].into_array();
        let rhs = buffer![10i32, 20].into_array();

        let scalar_fn = TypedScalarFnInstance::new(Binary, Operator::Add).erased();
        let err = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs])
            .expect_err("ScalarFnArray::try_new must reject mismatched child lengths");

        assert!(
            err.to_string()
                .contains("ScalarFnArray must have children equal to the array length")
        );
    }

    #[test]
    fn rejects_wrong_arity() {
        let child = buffer![1i32, 2, 3].into_array();
        let scalar_fn = TypedScalarFnInstance::new(Binary, Operator::Add).erased();

        let Err(err) = ScalarFnArray::try_new(scalar_fn.clone(), vec![child.clone()]) else {
            panic!("Binary must reject one child");
        };
        assert!(
            err.to_string()
                .contains("ScalarFnArray requires 2 children, got 1")
        );

        let Err(err) = ScalarFnArray::try_new(scalar_fn, vec![child.clone(), child.clone(), child])
        else {
            panic!("Binary must reject three children");
        };
        assert!(
            err.to_string()
                .contains("ScalarFnArray requires 2 children, got 3")
        );
    }

    #[test]
    fn rejects_missing_child_slot() {
        let lhs = buffer![1i32, 2, 3].into_array();
        let dtype = lhs.dtype().clone();
        let scalar_fn = TypedScalarFnInstance::new(Binary, Operator::Add).erased();
        let vtable = ScalarFn { id: scalar_fn.id() };
        let data = ScalarFnData { scalar_fn };
        let slots = [Some(lhs), None].into_iter().collect::<ArraySlots>();

        let Err(err) =
            Array::<ScalarFn>::try_from_parts(ArrayParts::new(vtable, dtype, 3, data, slots))
        else {
            panic!("ScalarFnArray must reject missing child slots");
        };

        assert!(
            err.to_string()
                .contains("ScalarFnArray requires every child slot to be present")
        );
    }

    #[test]
    fn test_scalar_fn_without_children_requires_explicit_len() -> VortexResult<()> {
        let list = Scalar::list(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            vec![Scalar::from(1i32)],
            Nullability::NonNullable,
        );
        let scalar_fn =
            TypedScalarFnInstance::new(PreparedSet, PreparedSetData::try_new(list.clone())?)
                .erased();

        let Err(err) = ScalarFnArray::try_new(scalar_fn.clone(), vec![]) else {
            panic!("ScalarFnArray::try_new should reject zero children");
        };
        assert!(
            err.to_string()
                .contains("ScalarFnArray length cannot be inferred without children")
        );

        let scalar_fn_array = ScalarFnArray::try_new_with_len(scalar_fn, vec![], 3)?;
        assert_eq!(scalar_fn_array.len(), 3);
        assert_eq!(scalar_fn_array.child_count(), 0);
        let scalar_fn_array = scalar_fn_array.into_array();
        test_scalar_probe_consistency(
            &scalar_fn_array,
            &mut SESSION.create_execution_ctx(),
        );
        assert_arrays_eq!(
            scalar_fn_array,
            ConstantArray::new(list, 3),
            &mut SESSION.create_execution_ctx()
        );

        Ok(())
    }

    #[test]
    fn rejects_literal() {
        let scalar_fn = TypedScalarFnInstance::new(Literal, Scalar::from(1i32)).erased();

        let Err(err) = ScalarFnArray::try_new_with_len(scalar_fn.clone(), vec![], 3) else {
            panic!("ScalarFnArray::try_new_with_len must reject a literal");
        };
        assert!(
            err.to_string()
                .contains("ScalarFnArray cannot wrap a literal")
        );

        let vtable = ScalarFn { id: scalar_fn.id() };
        let data = ScalarFnData { scalar_fn };
        let parts = ArrayParts::new(vtable, PType::I32.into(), 3, data, ArraySlots::new());

        let Err(err) = Array::<ScalarFn>::try_from_parts(parts) else {
            panic!("ScalarFnArray must reject a literal from parts");
        };
        assert!(
            err.to_string()
                .contains("ScalarFnArray cannot wrap a literal")
        );
    }

    #[test]
    fn test_scalar_fn_mul() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = buffer![2i32, 3, 4].into_array();
        let rhs = buffer![5i32, 6, 7].into_array();

        let scalar_fn = TypedScalarFnInstance::new(Binary, Operator::Mul).erased();
        let scalar_fn_array = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs])?;

        let expected = buffer![10i32, 18, 28].into_array();
        assert_arrays_eq!(scalar_fn_array, expected, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_scalar_fn_with_nullable() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::AllValid).into_array();
        let rhs = PrimitiveArray::new(
            buffer![10i32, 20, 30],
            Validity::from_iter([true, false, true]),
        )
        .into_array();

        let scalar_fn = TypedScalarFnInstance::new(Binary, Operator::Add).erased();
        let scalar_fn_array = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs])?;

        let expected = PrimitiveArray::new(
            buffer![11i32, 0, 33],
            Validity::from_iter([true, false, true]),
        )
        .into_array();
        assert_arrays_eq!(scalar_fn_array, expected, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_scalar_fn_comparison() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = buffer![1i32, 5, 3].into_array();
        let rhs = buffer![2i32, 5, 1].into_array();

        let scalar_fn = TypedScalarFnInstance::new(Binary, Operator::Eq).erased();
        let scalar_fn_array = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs])?;

        let expected = BoolArray::from_iter([false, true, false]).into_array();
        assert_arrays_eq!(scalar_fn_array, expected, &mut ctx);

        Ok(())
    }

    /// Mul(Add(a1, a2), a3)
    fn scalar_nested_add() -> VortexResult<ArrayRef> {
        let lhs_n = PrimitiveArray::from_option_iter((0..10i64).map(|i| (i % 4 != 0).then_some(i)));
        let rhs_r = PrimitiveArray::from_iter(0..10i64);

        let scalar_fn_n = TypedScalarFnInstance::new(Binary, Operator::Add).erased();
        let args_n = vec![lhs_n.into_array(), rhs_r.into_array()];

        let lhs = ScalarFnArray::try_new(scalar_fn_n, args_n)?.into_array();
        let rhs = PrimitiveArray::from_iter(0..10i64);

        let scalar_fn = TypedScalarFnInstance::new(Binary, Operator::Mul).erased();
        let args = vec![lhs, rhs.into_array()];

        Ok(ScalarFnArray::try_new(scalar_fn, args)?.into_array())
    }

    #[test]
    fn scalar_fn_probe() -> VortexResult<()> {
        let ctx = &mut SESSION.create_execution_ctx();
        let array = scalar_nested_add()?;
        let mut probe = array.probe();

        assert!(!probe.execute_is_valid(0, ctx)?);
        assert!(probe.execute_scalar(0, ctx)?.is_null());
        assert!(probe.execute_is_valid(9, ctx)?);
        assert_eq!(probe.execute_scalar(9, ctx)?, Scalar::from(Some(162i64)));

        Ok(())
    }

    #[test]
    fn scalar_fn_repeated_probe() -> VortexResult<()> {
        let ctx = &mut SESSION.create_execution_ctx();
        let array = scalar_nested_add()?;
        let mut probe = array.repeated_probe();

        assert!(!probe.execute_is_valid(0, ctx)?);
        assert!(probe.execute_scalar(0, ctx)?.is_null());
        assert!(probe.execute_is_valid(9, ctx)?);
        assert_eq!(probe.execute_scalar(9, ctx)?, Scalar::from(Some(162i64)));

        Ok(())
    }

    #[test]
    fn scalar_fn_probe_preserves_nested_nulls_and_slices() -> VortexResult<()> {
        let array = scalar_nested_add()?;
        test_scalar_probe_consistency(&array, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[test]
    fn scalar_fn_all_valid() -> VortexResult<()> {
        let ctx = &mut SESSION.create_execution_ctx();
        let array = scalar_nested_add()?;
        assert!(!array.all_valid(ctx)?);
        assert!(!array.all_invalid(ctx)?);
        assert_eq!(array.valid_count(ctx)?, 7);
        assert_eq!(array.invalid_count(ctx)?, 3);
        Ok(())
    }
}
