// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Filter;
use crate::arrays::ScalarFn;
use crate::arrays::ScalarFnArray;
use crate::arrays::Slice;
use crate::arrays::StructArray;
use crate::arrays::filter::prepare_mask_for_reuse;
use crate::arrays::scalar_fn::AnyScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ArrayReduceRule;
use crate::optimizer::rules::ReduceRuleSet;
use crate::scalar_fn::ArrayReduceNode;
use crate::scalar_fn::fns::mask::Mask;
use crate::scalar_fn::fns::pack::Pack;
use crate::validity::Validity;

pub(super) const RULES: ReduceRuleSet<ScalarFn> =
    ReduceRuleSet::new(&[&ScalarFnPackToStructRule, &ScalarFnAbstractReduceRule]);

pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_scalar_fn_reduce_parent_rule(Filter, const { &ScalarFilterPushdownRule });
    kernels.register_scalar_fn_reduce_parent_rule(Slice, const { &ScalarFnSliceReduceRule });
    kernels.register_scalar_fn_reduce_parent_rule(AnyScalarFn, const { &MaskLiftRule });
}

/// Converts a ScalarFnArray with Pack into a StructArray directly.
#[derive(Debug)]
struct ScalarFnPackToStructRule;
impl ArrayReduceRule<ScalarFn> for ScalarFnPackToStructRule {
    fn reduce(&self, array: ArrayView<'_, ScalarFn>) -> VortexResult<Option<ArrayRef>> {
        let Some(pack_options) = array.scalar_fn().as_opt::<Pack>() else {
            return Ok(None);
        };

        let validity = match pack_options.nullability {
            crate::dtype::Nullability::NonNullable => Validity::NonNullable,
            crate::dtype::Nullability::Nullable => Validity::AllValid,
        };

        Ok(Some(
            StructArray::try_new(
                pack_options.names.clone(),
                array.children(),
                array.len(),
                validity,
            )?
            .into_array(),
        ))
    }
}

#[derive(Debug)]
struct ScalarFnSliceReduceRule;
impl ArrayParentReduceRule<ScalarFn> for ScalarFnSliceReduceRule {
    type Parent = Slice;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, ScalarFn>,
        parent: ArrayView<'_, Slice>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let range = parent.slice_range();

        let children: Vec<_> = array
            .iter_children()
            .map(|c| c.slice(range.clone()))
            .collect::<VortexResult<_>>()?;

        Ok(Some(
            ScalarFnArray::try_new_with_len(array.scalar_fn().clone(), children, range.len())?
                .into_array(),
        ))
    }
}

#[derive(Debug)]
struct ScalarFnAbstractReduceRule;
impl ArrayReduceRule<ScalarFn> for ScalarFnAbstractReduceRule {
    fn reduce(&self, array: ArrayView<'_, ScalarFn>) -> VortexResult<Option<ArrayRef>> {
        let node = ArrayReduceNode::new(array.as_ref());
        if let Some(reduced) = array.scalar_fn().reduce_array(&node)? {
            return Ok(Some(reduced.into_array()));
        }
        Ok(None)
    }
}

#[derive(Debug)]
struct ScalarFilterPushdownRule;

impl ArrayParentReduceRule<ScalarFn> for ScalarFilterPushdownRule {
    type Parent = Filter;

    fn reduce_parent(
        &self,
        child: ArrayView<'_, ScalarFn>,
        parent: ArrayView<'_, Filter>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let nchildren = child
            .iter_children()
            .filter(|c| !c.is::<Constant>())
            .count();
        if nchildren > 1
            && let Some(values) = parent.filter_mask().values()
        {
            prepare_mask_for_reuse(values, nchildren);
        }

        // Selection precedes value evaluation, including errors in unselected rows.
        let new_children: Vec<_> = child
            .iter_children()
            .map(|c| match c.as_opt::<Constant>() {
                Some(array) => {
                    Ok(ConstantArray::new(array.scalar().clone(), parent.len()).into_array())
                }
                None => c.filter(parent.filter_mask().clone()),
            })
            .try_collect()?;

        Ok(Some(
            ScalarFnArray::try_new_with_len(child.scalar_fn().clone(), new_children, parent.len())?
                .into_array(),
        ))
    }
}

/// Rewrites `f(.., mask(x, m), ..)` to `mask(f(.., x, ..), m)`.
///
/// This lets `f` reach the encoding of `x`, such as the values of a run-end array, which the mask
/// would otherwise hide.
#[derive(Debug)]
struct MaskLiftRule;

impl ArrayParentReduceRule<ScalarFn> for MaskLiftRule {
    type Parent = AnyScalarFn;

    fn reduce_parent(
        &self,
        child: ArrayView<'_, ScalarFn>,
        parent: ArrayView<'_, ScalarFn>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        if child.scalar_fn().as_opt::<Mask>().is_none() {
            return Ok(None);
        }
        // A row where `m` is false is null on both sides only if `f` is strict. `f` now also runs
        // on the values that `m` hides, so it must not be able to fail on them.
        // Lifting a mask above another mask would swap the two forever.
        let scalar_fn = parent.scalar_fn();
        if scalar_fn.as_opt::<Mask>().is_some()
            || !scalar_fn.signature().is_strict()
            || !scalar_fn.signature().is_infallible()
        {
            return Ok(None);
        }

        let mut children = parent.children();
        children[child_idx] = child.get_child(0).clone();
        let lifted = ScalarFnArray::try_new_with_len(scalar_fn.clone(), children, parent.len())?;
        let masked = Mask::try_new(lifted.into_array(), child.get_child(1).clone())?.into_array();
        if masked.dtype() != parent.dtype() {
            return Ok(None);
        }
        Ok(Some(masked))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use super::ScalarFilterPushdownRule;
    use crate::ArrayRef;
    use crate::VortexSessionExecute;
    use crate::array::IntoArray;
    use crate::arrays::BoolArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::Constant;
    use crate::arrays::DictArray;
    use crate::arrays::Filter;
    use crate::arrays::FilterArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::ScalarFn;
    use crate::arrays::ScalarFnArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::runend::RunEnd;
    use crate::arrays::scalar_fn::ExactScalarFn;
    use crate::arrays::scalar_fn::ScalarFnArrayExt;
    use crate::arrays::scalar_fn::rules::ConstantArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::cast;
    use crate::expr::is_not_null;
    use crate::expr::is_null;
    use crate::expr::root;
    use crate::optimizer::ArrayOptimizer;
    use crate::optimizer::rules::ArrayParentReduceRule;
    use crate::scalar::Scalar;
    use crate::scalar_fn::EmptyOptions;
    use crate::scalar_fn::ScalarFnVTableExt;
    use crate::scalar_fn::TypedScalarFnInstance;
    use crate::scalar_fn::fns::binary::Binary;
    use crate::scalar_fn::fns::list_contains::PreparedSet;
    use crate::scalar_fn::fns::lower::Lower;
    use crate::scalar_fn::fns::mask::Mask as MaskFn;
    use crate::scalar_fn::fns::operators::Operator;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[rstest]
    #[case(0, [14, 14, 14, 14, 14])]
    #[case(1, [7, 27, 47, 67, 87])]
    #[case(2, [0, 40, 80, 120, 160])]
    fn test_filter_pushdown(
        #[case] nchildren: usize,
        #[case] expected: [i32; 5],
    ) -> VortexResult<()> {
        let children = (0..2)
            .map(|index| {
                if index < nchildren {
                    PrimitiveArray::from_iter(0i32..100).into_array()
                } else {
                    ConstantArray::new(7i32, 100).into_array()
                }
            })
            .collect::<Vec<_>>();
        let array = ScalarFnArray::try_new(
            TypedScalarFnInstance::new(Binary, Operator::Add).erased(),
            children,
        )?;
        let mask = Mask::from_iter((0..100).map(|index| index % 20 == 0));
        let values = mask
            .values()
            .ok_or_else(|| vortex_err!("expected mask values"))?;
        assert!(values.cached_indices().is_none());
        let parent = FilterArray::try_new(array.clone().into_array(), mask.clone())?;

        let result = ScalarFilterPushdownRule
            .reduce_parent(array.as_view(), parent.as_view(), 0)?
            .ok_or_else(|| vortex_err!("expected filter pushdown"))?;

        assert_eq!(values.cached_indices().is_some(), nchildren > 1);
        let scalar_fn = result.as_::<ScalarFn>();
        assert_eq!(scalar_fn.len(), 5);
        for (index, child) in scalar_fn.iter_children().enumerate() {
            assert_eq!(child.len(), 5);
            if index < nchildren {
                assert!(child.is::<Filter>());
            } else {
                assert!(child.is::<Constant>());
            }
        }
        let expected = PrimitiveArray::from_iter(expected);
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[test]
    fn test_filter_pushdown_without_children() -> VortexResult<()> {
        let list = Scalar::list(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            vec![Scalar::from(7i32)],
            Nullability::NonNullable,
        );
        let array = PreparedSet::try_new_array(list.clone(), 3)?;
        let parent = FilterArray::try_new(array.clone(), Mask::from_iter([true, false, true]))?;

        let result = ScalarFilterPushdownRule
            .reduce_parent(array.as_::<ScalarFn>(), parent.as_view(), 0)?
            .ok_or_else(|| vortex_err!("expected filter pushdown"))?;

        assert_eq!(result.as_::<ScalarFn>().nchildren(), 0);
        assert_eq!(result.len(), 2);
        assert_arrays_eq!(
            result,
            ConstantArray::new(list, 2),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn test_empty_constants() {
        let array = ChunkedArray::try_new(
            vec![
                ConstantArray::new(Some(1u64), 0).into_array(),
                PrimitiveArray::from_iter(vec![2u64])
                    .into_array()
                    .apply(&cast(
                        root(),
                        DType::Primitive(PType::U64, Nullability::Nullable),
                    ))
                    .vortex_expect("casted"),
            ],
            DType::Primitive(PType::U64, Nullability::Nullable),
        )
        .vortex_expect("construction")
        .into_array();

        let expr = is_null(root());
        let result = array.apply(&expr).vortex_expect("expr evaluation");
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([false]),
            &mut SESSION.create_execution_ctx()
        );
    }

    #[test]
    fn test_reduce_isnull() -> VortexResult<()> {
        let ctx = &mut SESSION.create_execution_ctx();
        let validity = BoolArray::from_iter([true, false, true]).into_array();

        assert_arrays_eq!(
            validity.clone().apply(&is_null(root()))?,
            ConstantArray::new(false, 3),
            ctx
        );
        assert_arrays_eq!(
            validity.clone().apply(&is_not_null(root()))?,
            ConstantArray::new(true, 3),
            ctx
        );

        let buffer = buffer![1, 2, 3];
        let nullable = PrimitiveArray::new(buffer, Validity::Array(validity.clone())).into_array();

        assert_arrays_eq!(nullable.clone().apply(&is_not_null(root()))?, validity, ctx);
        assert_arrays_eq!(nullable.apply(&is_null(root()))?, validity.not()?, ctx);

        Ok(())
    }

    #[test]
    fn test_mask_lifted_above_strict_function() -> VortexResult<()> {
        // Rows: ["Great", "Great", "LAKES", "Great", "Great", null].
        let dict = DictArray::try_new(
            PrimitiveArray::from_iter([0u32, 1, 0, 2]).into_array(),
            VarBinViewArray::from_iter_nullable_str([Some("Great"), Some("LAKES"), None])
                .into_array(),
        )?
        .into_array();
        let runend = RunEnd::try_new(
            PrimitiveArray::from_iter([2u32, 3, 5, 6]).into_array(),
            dict,
            &mut SESSION.create_execution_ctx(),
        )?
        .into_array();
        let masked = MaskFn::try_new(
            runend,
            BoolArray::from_iter([true, false, true, true, true, true]).into_array(),
        )?
        .into_array();
        let lowered = ScalarFnArray::try_new(Lower.bind(EmptyOptions), [masked])?
            .into_array()
            .optimize_ctx(&*SESSION)?;

        insta::assert_snapshot!(lowered.display_tree_encodings_only().to_string(), @r"
        root: vortex.mask(utf8?, len=6)
          input: vortex.runend(utf8?, len=6)
            ends: vortex.primitive(u32, len=4)
            values: vortex.dict(utf8?, len=4)
              codes: vortex.primitive(u32, len=4)
              values: vortex.lower(utf8?, len=3)
                input: vortex.varbinview(utf8?, len=3)
                  validity: vortex.bool(bool, len=3)
          mask: vortex.bool(bool, len=6)
        ");
        assert_arrays_eq!(
            lowered,
            VarBinViewArray::from_iter_nullable_str([
                Some("great"),
                None,
                Some("lakes"),
                Some("great"),
                Some("great"),
                None,
            ]),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    #[rstest]
    #[case::strict_infallible(
        Operator::Lt,
        PrimitiveArray::from_option_iter([Some(1i32), Some(2), Some(5), None]).into_array(),
        PrimitiveArray::from_iter([2i32, 3, 4, 5]).into_array(),
        true
    )]
    #[case::fallible(
        Operator::Sub,
        PrimitiveArray::from_option_iter([Some(1i32), Some(2), Some(5), None]).into_array(),
        PrimitiveArray::from_iter([2i32, 3, 4, 5]).into_array(),
        false
    )]
    #[case::non_strict(
        Operator::And,
        BoolArray::from_iter([Some(true), Some(false), Some(true), None]).into_array(),
        BoolArray::from_iter([true, true, false, true]).into_array(),
        false
    )]
    fn test_mask_lift_with_non_constant_sibling(
        #[case] operator: Operator,
        #[case] lhs: ArrayRef,
        #[case] rhs: ArrayRef,
        #[case] lifted: bool,
    ) -> VortexResult<()> {
        let masked = MaskFn::try_new(
            rhs,
            BoolArray::from_iter([true, false, true, true]).into_array(),
        )?
        .into_array();
        let array = ScalarFnArray::try_new(
            TypedScalarFnInstance::new(Binary, operator).erased(),
            [lhs, masked],
        )?
        .into_array();
        let optimized = array.optimize_ctx(&*SESSION)?;

        assert_eq!(optimized.is::<ExactScalarFn<MaskFn>>(), lifted);
        assert_arrays_eq!(optimized, array, &mut SESSION.create_execution_ctx());
        Ok(())
    }
}
