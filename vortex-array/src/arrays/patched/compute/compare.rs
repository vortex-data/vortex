// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::patched::BoolPatchStrategy;
use crate::arrays::patched::FloatPatchStrategy;
use crate::arrays::patched::IntegerPatchStrategy;
use crate::arrays::patched::PatchStrategy;
use crate::arrays::patched::Patched;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::builtins::ArrayBuiltins;
use crate::scalar_fn::builtin::compare::CompareKernel;
use crate::scalar_fn::builtin::operators::CompareOperator;

impl CompareKernel for Patched {
    fn compare(
        array: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        op: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let (PatchStrategy::Bool(BoolPatchStrategy::Replace)
        | PatchStrategy::Integer(IntegerPatchStrategy::Replace)
        | PatchStrategy::Float(FloatPatchStrategy::Replace)) = array.strategy();
        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };
        let base = array
            .inner()
            .compare(rhs.clone(), op)?
            .execute::<BoolArray>(ctx)?;
        let patches = array.absolute_patches(ctx)?.map_values(|values| {
            let rhs = ConstantArray::new(constant.clone(), values.len()).into_array();
            values.compare(rhs, op)
        })?;
        Ok(Some(base.patch(&patches, ctx)?.into_array()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::Patched;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::slice::SliceKernel;
    use crate::assert_arrays_eq;
    use crate::patches::Patches;
    use crate::scalar_fn::builtin::compare::CompareKernel;
    use crate::scalar_fn::builtin::operators::CompareOperator;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn test_basic() {
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = PrimitiveArray::from_iter(0u32..512).into_array();
        let patches = Patches::new(
            512,
            0,
            buffer![509u16, 510, 511].into_array(),
            buffer![u32::MAX; 3].into_array(),
            None,
            &mut ctx,
        )
        .unwrap();

        let mut ctx = SESSION.create_execution_ctx();

        let lhs = Patched::from_array_and_patches(lhs, &patches, &mut ctx)
            .unwrap()
            .into_array()
            .try_downcast::<Patched>()
            .unwrap();

        let rhs = ConstantArray::new(u32::MAX, 512).into_array();

        let result =
            <Patched as CompareKernel>::compare(lhs.as_view(), &rhs, CompareOperator::Eq, &mut ctx)
                .unwrap()
                .unwrap();

        let expected =
            BoolArray::from_indices(512, [509, 510, 511], Validity::NonNullable).into_array();

        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[test]
    fn test_with_offset() {
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = PrimitiveArray::from_iter(0u32..512).into_array();
        let patches = Patches::new(
            512,
            0,
            buffer![5u16, 510, 511].into_array(),
            buffer![u32::MAX; 3].into_array(),
            None,
            &mut ctx,
        )
        .unwrap();

        let mut ctx = SESSION.create_execution_ctx();

        let lhs = Patched::from_array_and_patches(lhs, &patches, &mut ctx).unwrap();
        // Slice the array so that the first patch should be skipped.
        let lhs_ref = <Patched as SliceKernel>::slice(lhs.as_view(), 10..512, &mut ctx)
            .unwrap()
            .unwrap();
        let lhs = lhs_ref.try_downcast::<Patched>().unwrap();

        assert_eq!(lhs.len(), 502);

        let rhs = ConstantArray::new(u32::MAX, lhs.len()).into_array();

        let result =
            <Patched as CompareKernel>::compare(lhs.as_view(), &rhs, CompareOperator::Eq, &mut ctx)
                .unwrap()
                .unwrap();

        let expected = BoolArray::from_indices(502, [500, 501], Validity::NonNullable).into_array();

        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[test]
    fn test_subnormal_f32() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Subnormal f32 values are smaller than f32::MIN_POSITIVE but greater than 0
        let subnormal: f32 = f32::MIN_POSITIVE / 2.0;
        assert!(subnormal > 0.0 && subnormal < f32::MIN_POSITIVE);

        let lhs = PrimitiveArray::from_iter((0..512).map(|i| i as f32)).into_array();

        let patches = Patches::new(
            512,
            0,
            buffer![509u16, 510, 511].into_array(),
            buffer![f32::NAN, subnormal, f32::NEG_INFINITY].into_array(),
            None,
            &mut ctx,
        )?;

        let mut ctx = SESSION.create_execution_ctx();
        let lhs = Patched::from_array_and_patches(lhs, &patches, &mut ctx)?
            .into_array()
            .try_downcast::<Patched>()
            .map_err(|_| vortex_err!("expected patched array"))?;

        let rhs = ConstantArray::new(subnormal, 512).into_array();

        let result = <Patched as CompareKernel>::compare(
            lhs.as_view(),
            &rhs,
            CompareOperator::Eq,
            &mut ctx,
        )?
        .ok_or_else(|| vortex_err!("expected compare result"))?;

        let expected = BoolArray::from_indices(512, [510], Validity::NonNullable).into_array();

        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_pos_neg_zero() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = PrimitiveArray::from_iter([-0.0f32; 10]).into_array();

        let patches = Patches::new(
            10,
            0,
            buffer![5u16, 6, 7, 8, 9].into_array(),
            buffer![f32::NAN, f32::NEG_INFINITY, 0f32, -0.0f32, f32::INFINITY].into_array(),
            None,
            &mut ctx,
        )?;

        let mut ctx = SESSION.create_execution_ctx();
        let lhs = Patched::from_array_and_patches(lhs, &patches, &mut ctx)?
            .into_array()
            .try_downcast::<Patched>()
            .map_err(|_| vortex_err!("expected patched array"))?;

        let rhs = ConstantArray::new(0.0f32, 10).into_array();

        let result = <Patched as CompareKernel>::compare(
            lhs.as_view(),
            &rhs,
            CompareOperator::Eq,
            &mut ctx,
        )?
        .ok_or_else(|| vortex_err!("expected compare result"))?;

        let expected = BoolArray::from_indices(10, [7], Validity::NonNullable).into_array();

        assert_arrays_eq!(result, expected, &mut ctx);

        Ok(())
    }
}
