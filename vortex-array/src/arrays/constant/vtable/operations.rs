// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::Constant;
use crate::scalar::Scalar;

impl OperationsVTable<Constant> for Constant {
    fn scalar_at(
        array: ArrayView<'_, Constant>,
        _index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Ok(array.scalar.clone())
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::UnionVariants;
    use crate::scalar::Scalar;

    #[test]
    fn scalar_at_preserves_union_scalar() -> VortexResult<()> {
        let variants = UnionVariants::try_new(
            ["int", "string"].into(),
            vec![
                DType::Primitive(PType::I32, Nullability::Nullable),
                DType::Utf8(Nullability::NonNullable),
            ],
            vec![5, 9],
        )?;

        let scalar = Scalar::union(
            variants.clone(),
            5,
            Scalar::primitive(42_i32, Nullability::Nullable),
        )?;
        let array = ConstantArray::new(scalar.clone(), 3).into_array();
        let mut ctx = crate::array_session().create_execution_ctx();

        assert_eq!(array.execute_scalar(1, &mut ctx)?, scalar);

        let null = Scalar::null(DType::Union(variants));
        let array = ConstantArray::new(null.clone(), 3).into_array();

        assert_eq!(array.execute_scalar(1, &mut ctx)?, null);

        Ok(())
    }
}
