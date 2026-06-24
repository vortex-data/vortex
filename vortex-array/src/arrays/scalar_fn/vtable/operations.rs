// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::ConstantArray;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::scalar_fn::vtable::ScalarFn;
use crate::columnar::Columnar;
use crate::scalar::Scalar;
use crate::scalar_fn::VecExecutionArgs;

impl OperationsVTable<ScalarFn> for ScalarFn {
    fn scalar_at(
        array: ArrayView<'_, ScalarFn>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let inputs: Vec<_> = array
            .children()
            .iter()
            .map(|child| Ok(ConstantArray::new(child.execute_scalar(index, ctx)?, 1).into_array()))
            .collect::<VortexResult<_>>()?;

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
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::ScalarFnArray;
    use crate::arrays::scalar_fn::ScalarFnArrayExt;
    use crate::assert_arrays_eq;
    use crate::scalar::Scalar;
    use crate::scalar_fn::TypedScalarFnInstance;
    use crate::scalar_fn::fns::binary::Binary;
    use crate::scalar_fn::fns::literal::Literal;
    use crate::scalar_fn::fns::operators::Operator;
    use crate::validity::Validity;

    #[test]
    fn test_scalar_fn_add() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let lhs = buffer![1i32, 2, 3].into_array();
        let rhs = buffer![10i32, 20, 30].into_array();

        let scalar_fn = TypedScalarFnInstance::new(Binary, Operator::Add).erased();
        let scalar_fn_array = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs])?;

        assert_eq!(scalar_fn_array.len(), 3);

        let result = scalar_fn_array
            .into_array()
            .execute::<Canonical>(&mut array_session().create_execution_ctx())?
            .into_array();
        let expected = buffer![11i32, 22, 33].into_array();
        assert_arrays_eq!(result, expected, &mut ctx);

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
    fn test_scalar_fn_without_children_requires_explicit_len() -> VortexResult<()> {
        let scalar_fn = TypedScalarFnInstance::new(Literal, Scalar::from(1i32)).erased();

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

        Ok(())
    }

    #[test]
    fn test_scalar_fn_mul() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let lhs = buffer![2i32, 3, 4].into_array();
        let rhs = buffer![5i32, 6, 7].into_array();

        let scalar_fn = TypedScalarFnInstance::new(Binary, Operator::Mul).erased();
        let scalar_fn_array = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs])?;

        let result = scalar_fn_array
            .into_array()
            .execute::<Canonical>(&mut array_session().create_execution_ctx())?
            .into_array();
        let expected = buffer![10i32, 18, 28].into_array();
        assert_arrays_eq!(result, expected, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_scalar_fn_with_nullable() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let lhs = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::AllValid).into_array();
        let rhs = PrimitiveArray::new(
            buffer![10i32, 20, 30],
            Validity::from_iter([true, false, true]),
        )
        .into_array();

        let scalar_fn = TypedScalarFnInstance::new(Binary, Operator::Add).erased();
        let scalar_fn_array = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs])?;

        let result = scalar_fn_array
            .into_array()
            .execute::<Canonical>(&mut array_session().create_execution_ctx())?
            .into_array();
        let expected = PrimitiveArray::new(
            buffer![11i32, 0, 33],
            Validity::from_iter([true, false, true]),
        )
        .into_array();
        assert_arrays_eq!(result, expected, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_scalar_fn_comparison() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let lhs = buffer![1i32, 5, 3].into_array();
        let rhs = buffer![2i32, 5, 1].into_array();

        let scalar_fn = TypedScalarFnInstance::new(Binary, Operator::Eq).erased();
        let scalar_fn_array = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs])?;

        let result = scalar_fn_array
            .into_array()
            .execute::<Canonical>(&mut array_session().create_execution_ctx())?
            .into_array();
        let expected = BoolArray::from_iter([false, true, false]).into_array();
        assert_arrays_eq!(result, expected, &mut ctx);

        Ok(())
    }
}
