// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::scalar_fn::array::ScalarFnArray;
    use crate::assert_arrays_eq;
    use crate::expr::Operator;
    use crate::expr::ScalarFn;
    use crate::expr::binary::Binary;
    use crate::validity::Validity;

    #[test]
    fn test_scalar_fn_add() -> VortexResult<()> {
        let lhs = buffer![1i32, 2, 3].into_array();
        let rhs = buffer![10i32, 20, 30].into_array();

        let scalar_fn = ScalarFn::new(Binary, Operator::Add);
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

        let scalar_fn = ScalarFn::new(Binary, Operator::Mul);
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

        let scalar_fn = ScalarFn::new(Binary, Operator::Add);
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

        let scalar_fn = ScalarFn::new(Binary, Operator::Eq);
        let scalar_fn_array = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs], 3)?;

        let result = scalar_fn_array.to_canonical()?.into_array();
        let expected = BoolArray::from_iter([false, true, false]).into_array();
        assert_arrays_eq!(result, expected);

        Ok(())
    }
}
