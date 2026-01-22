// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_scalar::Scalar;

use crate::Array;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::ConstantArray;
use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::vtable::ScalarFnVTable;
use crate::expr::ExecutionArgs;
use crate::expr::ExecutionResult;
use crate::vtable::OperationsVTable;

impl OperationsVTable<ScalarFnVTable> for ScalarFnVTable {
    fn scalar_at(array: &ScalarFnArray, index: usize) -> Scalar {
        let inputs: Vec<_> = array
            .children
            .iter()
            .map(|child| ConstantArray::new(child.scalar_at(index), 1).into_array())
            .collect::<_>();

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let args = ExecutionArgs {
            inputs,
            row_count: 1,
            ctx: &mut ctx,
        };

        let result = array
            .scalar_fn
            .execute(args)
            .vortex_expect("todo vortex result return");

        let scalar = match result {
            ExecutionResult::Array(arr) => {
                tracing::info!(
                    "Scalar function {} returned non-constant array from execution over all scalar inputs",
                    array.scalar_fn,
                );
                arr.as_ref().scalar_at(0)
            }
            ExecutionResult::Scalar(constant) => constant.scalar().clone(),
        };

        debug_assert_eq!(
            scalar.dtype(),
            &array.dtype,
            "Scalar function {} returned dtype {:?} but expected {:?}",
            array.scalar_fn,
            scalar.dtype(),
            array.dtype
        );

        scalar
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_scalar::Scalar;

    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::scalar_fn::array::ScalarFnArray;
    use crate::expr::Operator;
    use crate::expr::ScalarFn;
    use crate::expr::binary::Binary;

    #[test]
    fn test_scalar_at_add() -> VortexResult<()> {
        let lhs = buffer![1i32, 2, 3].into_array();
        let rhs = buffer![10i32, 20, 30].into_array();

        let scalar_fn = ScalarFn::new(Binary, Operator::Add);
        let scalar_fn_array = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs], 3)?;

        assert_eq!(scalar_fn_array.scalar_at(0), Scalar::from(11i32));
        assert_eq!(scalar_fn_array.scalar_at(1), Scalar::from(22i32));
        assert_eq!(scalar_fn_array.scalar_at(2), Scalar::from(33i32));

        Ok(())
    }

    #[test]
    fn test_scalar_at_mul() -> VortexResult<()> {
        let lhs = buffer![2i32, 3, 4].into_array();
        let rhs = buffer![5i32, 6, 7].into_array();

        let scalar_fn = ScalarFn::new(Binary, Operator::Mul);
        let scalar_fn_array = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs], 3)?;

        assert_eq!(scalar_fn_array.scalar_at(0), Scalar::from(10i32));
        assert_eq!(scalar_fn_array.scalar_at(1), Scalar::from(18i32));
        assert_eq!(scalar_fn_array.scalar_at(2), Scalar::from(28i32));

        Ok(())
    }

    #[test]
    fn test_scalar_at_with_nullable() -> VortexResult<()> {
        use crate::validity::Validity;

        let lhs = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::AllValid).into_array();
        let rhs = PrimitiveArray::new(
            buffer![10i32, 20, 30],
            Validity::from_iter([true, false, true]),
        )
        .into_array();

        let scalar_fn = ScalarFn::new(Binary, Operator::Add);
        let scalar_fn_array = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs], 3)?;

        assert_eq!(scalar_fn_array.scalar_at(0), Scalar::from(11i32));
        assert!(scalar_fn_array.scalar_at(1).is_null());
        assert_eq!(scalar_fn_array.scalar_at(2), Scalar::from(33i32));

        Ok(())
    }

    #[test]
    fn test_scalar_at_comparison() -> VortexResult<()> {
        let lhs = buffer![1i32, 5, 3].into_array();
        let rhs = buffer![2i32, 5, 1].into_array();

        let scalar_fn = ScalarFn::new(Binary, Operator::Eq);
        let scalar_fn_array = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs], 3)?;

        assert_eq!(scalar_fn_array.scalar_at(0), Scalar::from(false));
        assert_eq!(scalar_fn_array.scalar_at(1), Scalar::from(true));
        assert_eq!(scalar_fn_array.scalar_at(2), Scalar::from(false));

        Ok(())
    }
}
