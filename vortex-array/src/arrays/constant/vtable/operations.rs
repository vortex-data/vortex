// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::ProbeState;
use crate::arrays::Constant;
use crate::scalar::Scalar;

impl OperationsVTable<Constant> for Constant {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, Constant>,
        _index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Ok(state.array().scalar.clone())
    }

    fn scalar_at(
        array: ArrayView<'_, Constant>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::UnionVariants;
    use crate::scalar::Scalar;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

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
            Nullability::Nullable,
        )?;
        let array = ConstantArray::new(scalar.clone(), 3).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let actual = array.execute_scalar(1, &mut ctx)?;
        assert_eq!(actual.dtype(), scalar.dtype());
        assert_eq!(actual, scalar);

        let null = Scalar::null(DType::Union(variants, Nullability::Nullable));
        let array = ConstantArray::new(null.clone(), 3).into_array();

        let actual = array.execute_scalar(1, &mut ctx)?;
        assert_eq!(actual.dtype(), null.dtype());
        assert_eq!(actual, null);

        Ok(())
    }
}
