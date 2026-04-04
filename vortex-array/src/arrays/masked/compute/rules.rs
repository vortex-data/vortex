// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::Masked;
use crate::arrays::ScalarFnArray;
use crate::arrays::ScalarFnVTable;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::scalar_fn::AnyScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::ArrayOptimizer;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::fns::mask::Mask as MaskExpr;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) const PARENT_RULES: ParentRuleSet<Masked> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&FilterReduceAdaptor(Masked)),
    ParentRuleSet::lift(&MaskReduceAdaptor(Masked)),
    ParentRuleSet::lift(&MaskedScalarFnPushDownRule),
    ParentRuleSet::lift(&SliceReduceAdaptor(Masked)),
    ParentRuleSet::lift(&TakeReduceAdaptor(Masked)),
]);

#[derive(Debug)]
struct MaskedScalarFnPushDownRule;

impl ArrayParentReduceRule<Masked> for MaskedScalarFnPushDownRule {
    type Parent = AnyScalarFn;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Masked>,
        parent: ArrayView<'_, ScalarFnVTable>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let signature = parent.scalar_fn().signature();
        if signature.is_null_sensitive() || signature.is_fallible() {
            return Ok(None);
        }

        if !parent
            .iter_children()
            .enumerate()
            .all(|(idx, child)| idx == child_idx || child.is::<Constant>())
        {
            return Ok(None);
        }

        let pushed_child = ScalarFnArray::try_new(
            parent.scalar_fn().clone(),
            parent
                .iter_children()
                .enumerate()
                .map(|(idx, child)| {
                    if idx == child_idx {
                        array.child().clone()
                    } else {
                        child.clone()
                    }
                })
                .collect(),
            parent.len(),
        )?
        .into_array()
        .optimize()?;

        Ok(Some(MaskExpr.try_new_array(
            parent.len(),
            EmptyOptions,
            [pushed_child, array.validity()?.to_array(parent.len())],
        )?))
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::*;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::Dict;
    use crate::arrays::DictArray;
    use crate::arrays::MaskedArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::ScalarFnVTable as ScalarFnArrayVTable;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::scalar_fn::ScalarFnVTable as ScalarFnTrait;
    use crate::scalar_fn::fns::binary::Binary;
    use crate::scalar_fn::fns::is_null::IsNull;
    use crate::scalar_fn::fns::operators::Operator;
    use crate::validity::Validity;

    #[test]
    fn pushes_down_compare_to_masked_dict() -> VortexResult<()> {
        let masked = masked_dict_fixture()?;
        let optimized = masked.binary(
            ConstantArray::new(9i32, masked.len()).into_array(),
            Operator::Eq,
        )?;

        let encoded_result = if optimized.is::<Dict>() {
            optimized.clone()
        } else {
            assert!(root_scalar_fn_is::<MaskExpr>(&optimized));
            optimized.nth_child(0).unwrap()
        };
        assert!(encoded_result.is::<Dict>());
        assert!(!encoded_result.is::<Masked>());

        let expected = BoolArray::from_iter([
            Some(false),
            None,
            Some(false),
            Some(true),
            None,
            Some(false),
        ])
        .into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_arrays_eq!(
            optimized.execute::<Canonical>(&mut ctx)?.into_array(),
            expected
        );
        Ok(())
    }

    #[test]
    fn keeps_is_null_on_fallback_path() -> VortexResult<()> {
        let optimized = masked_dict_fixture()?.is_null()?;

        assert!(root_scalar_fn_is::<IsNull>(&optimized));
        assert!(optimized.nth_child(0).unwrap().is::<Masked>());
        Ok(())
    }

    #[test]
    fn keeps_fallible_binary_on_fallback_path() -> VortexResult<()> {
        let masked = masked_dict_fixture()?;
        let optimized = masked.binary(
            ConstantArray::new(2i32, masked.len()).into_array(),
            Operator::Div,
        )?;

        assert!(root_scalar_fn_is::<Binary>(&optimized));
        assert!(optimized.nth_child(0).unwrap().is::<Masked>());
        Ok(())
    }

    #[test]
    fn keeps_non_constant_sibling_on_fallback_path() -> VortexResult<()> {
        let masked = masked_dict_fixture()?;
        let sibling = PrimitiveArray::from_iter([7i32, 9, 11, 9, 7, 11]).into_array();
        let optimized = masked.binary(sibling, Operator::Eq)?;

        assert!(root_scalar_fn_is::<Binary>(&optimized));
        assert!(optimized.nth_child(0).unwrap().is::<Masked>());
        Ok(())
    }

    fn masked_dict_fixture() -> VortexResult<ArrayRef> {
        let dict = DictArray::try_new(
            PrimitiveArray::from_iter([0u8, 1, 2, 1, 0, 2]).into_array(),
            PrimitiveArray::from_iter([7i32, 9, 11]).into_array(),
        )?
        .into_array();

        Ok(MaskedArray::try_new(
            dict,
            Validity::from_iter([true, false, true, true, false, true]),
        )?
        .into_array())
    }

    fn root_scalar_fn_is<F: ScalarFnTrait>(array: &ArrayRef) -> bool {
        array
            .as_opt::<ScalarFnArrayVTable>()
            .is_some_and(|scalar_fn| scalar_fn.scalar_fn().is::<F>())
    }
}
