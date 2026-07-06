// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexResult;

use crate::RunEnd;
use crate::array::RunEndArrayExt;
use crate::decompress_bool::runend_decode_bools;

impl CompareKernel for RunEnd {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // If the RHS is constant, then we just need to compare against our encoded values.
        if let Some(const_scalar) = rhs.as_constant() {
            let values = lhs.values().binary(
                ConstantArray::new(const_scalar, lhs.values().len()).into_array(),
                Operator::from(operator),
            )?;
            return runend_decode_bools(
                lhs.ends().clone().execute::<PrimitiveArray>(ctx)?,
                values.execute::<BoolArray>(ctx)?,
                lhs.offset(),
                lhs.len(),
                ctx,
            )
            .map(Some);
        }

        // Otherwise, fall back
        Ok(None)
    }
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_session::VortexSession;

    use crate::RunEnd;
    use crate::RunEndArray;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    fn ree_array(ctx: &mut ExecutionCtx) -> RunEndArray {
        RunEnd::encode(
            PrimitiveArray::from_iter([1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5]).into_array(),
            ctx,
        )
        .unwrap()
    }

    #[test]
    fn compare_run_end() {
        let mut ctx = SESSION.create_execution_ctx();
        let arr = ree_array(&mut ctx);
        let res = arr
            .into_array()
            .binary(ConstantArray::new(5, 12).into_array(), Operator::Eq)
            .unwrap();
        let expected = BoolArray::from_iter([
            false, false, false, false, false, false, false, false, true, true, true, true,
        ]);
        assert_arrays_eq!(res, expected, &mut ctx);
    }
}
