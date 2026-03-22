// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::DynArray;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::ConstantArray;
use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::vtable::ScalarFnVTable;
use crate::columnar::Columnar;
use crate::scalar::Scalar;
use crate::scalar_fn::VecExecutionArgs;
use crate::vtable::OperationsVTable;

impl OperationsVTable<ScalarFnVTable> for ScalarFnVTable {
    fn scalar_at(array: &ScalarFnArray, index: usize) -> VortexResult<Scalar> {
        let inputs: Vec<_> = array
            .children
            .iter()
            .map(|child| Ok(ConstantArray::new(child.scalar_at(index)?, 1).into_array()))
            .collect::<VortexResult<_>>()?;

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let args = VecExecutionArgs::new(inputs, 1);
        let result = array.scalar_fn().execute(&args, &mut ctx)?;

        let scalar = match result.execute::<Columnar>(&mut ctx)? {
            Columnar::Canonical(arr) => {
                tracing::info!(
                    "Scalar function {} returned non-constant array from execution over all scalar inputs",
                    array.scalar_fn(),
                );
                arr.as_ref().scalar_at(0)?
            }
            Columnar::Constant(constant) => constant.scalar().clone(),
        };

        debug_assert_eq!(
            scalar.dtype(),
            &array.dtype,
            "Scalar function {} returned dtype {:?} but expected {:?}",
            array.scalar_fn(),
            scalar.dtype(),
            array.dtype
        );

        Ok(scalar)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::scalar_fn::array::ScalarFnArray;
    use crate::assert_arrays_eq;
    use crate::scalar_fn::ScalarFn;
    use crate::scalar_fn::fns::binary::Binary;
    use crate::scalar_fn::fns::operators::Operator;
    use crate::validity::Validity;

    #[test]
    fn test_scalar_fn_add() -> VortexResult<()> {
        let lhs = buffer![1i32, 2, 3].into_array();
        let rhs = buffer![10i32, 20, 30].into_array();

        let scalar_fn = ScalarFn::new(Binary, Operator::Add).erased();
        let scalar_fn_array = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs], 3)?;

        let result = scalar_fn_array.to_canonical()?.into_array();
        let expected = buffer![11i32, 22, 33].into_array();
        assert_arrays_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_scalar_fn_mul() -> VortexResult<()> {
        let lhs = buffer![2i32, 3, 4].into_array();
        let rhs = buffer![5i32, 6, 7].into_array();

        let scalar_fn = ScalarFn::new(Binary, Operator::Mul).erased();
        let scalar_fn_array = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs], 3)?;

        let result = scalar_fn_array.to_canonical()?.into_array();
        let expected = buffer![10i32, 18, 28].into_array();
        assert_arrays_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_scalar_fn_with_nullable() -> VortexResult<()> {
        let lhs = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::AllValid).into_array();
        let rhs = PrimitiveArray::new(
            buffer![10i32, 20, 30],
            Validity::from_iter([true, false, true]),
        )
        .into_array();

        let scalar_fn = ScalarFn::new(Binary, Operator::Add).erased();
        let scalar_fn_array = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs], 3)?;

        let result = scalar_fn_array.to_canonical()?.into_array();
        let expected = PrimitiveArray::new(
            buffer![11i32, 0, 33],
            Validity::from_iter([true, false, true]),
        )
        .into_array();
        assert_arrays_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_scalar_fn_comparison() -> VortexResult<()> {
        let lhs = buffer![1i32, 5, 3].into_array();
        let rhs = buffer![2i32, 5, 1].into_array();

        let scalar_fn = ScalarFn::new(Binary, Operator::Eq).erased();
        let scalar_fn_array = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs], 3)?;

        let result = scalar_fn_array.to_canonical()?.into_array();
        let expected = BoolArray::from_iter([false, true, false]).into_array();
        assert_arrays_eq!(result, expected);

        Ok(())
    }
}
