// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::matcher::Matcher;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::builtin::cast::Cast;
use vortex_array::scalar_fn::builtin::cast::CastReduceAdaptor;
use vortex_array::scalar_fn::builtin::not::Not;
use vortex_array::scalar_fn::builtin::not::NotReduce;
use vortex_array::scalar_fn::builtin::not::NotReduceAdaptor;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::Sparse;
use crate::SparseExt as _;

pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Cast, Sparse, const { &CastReduceAdaptor(Sparse) });
    kernels.register_reduce_parent_rule(Not, Sparse, const { &NotReduceAdaptor(Sparse) });
    kernels.register_reduce_parent_rule(Dict, Sparse, const { &DictSparseCodesRule });
}

/// Rewrites a dictionary whose codes are sparse with a null fill into a sparse array of the
/// looked-up patches: `Dict<Sparse<patches, null>, values>` becomes
/// `Sparse<Dict<patch_codes, values>, null>`.
///
/// Every row outside the patches is null either way, so only the patch codes need a lookup, and
/// the result stays as small as the codes instead of growing to one value per row.
#[derive(Debug)]
struct DictSparseCodesRule;

impl ArrayParentReduceRule<Sparse> for DictSparseCodesRule {
    type Parent = Dict;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Sparse>,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only handle the codes child (index 0); a non-null fill selects a value, which needs a
        // buffer read to become the new fill.
        if child_idx != 0 || !array.fill_scalar().is_null() {
            return Ok(None);
        }
        let values = parent.values();
        let patches = array.patches().map_values(|codes| values.take(codes))?;
        Ok(Some(
            Sparse::try_new_from_patches(patches, Scalar::null(parent.dtype().as_nullable()))?
                .into_array(),
        ))
    }
}

impl NotReduce for Sparse {
    fn invert(array: ArrayView<'_, Self>) -> VortexResult<Option<ArrayRef>> {
        let inverted_fill = array.fill_scalar().as_bool().invert().into_scalar();
        let inverted_patches = array.patches().map_values(|values| values.not())?;
        Ok(Some(
            Sparse::try_new_from_patches(inverted_patches, inverted_fill)?.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_array::scalar::Scalar;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Sparse;
    use crate::SparseExt as _;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    fn values() -> ArrayRef {
        VarBinViewArray::from_iter_str(["alpha", "beta", "gamma"]).into_array()
    }

    /// Regression for #4784: a take with mostly-null sparse indices must stay sparse instead of
    /// expanding to one code per row.
    #[test]
    fn null_fill_codes_become_sparse() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let codes = Sparse::try_new(
            buffer![3u64, 5, 7].into_array(),
            PrimitiveArray::from_option_iter([Some(1u32), None, Some(2)]).into_array(),
            10,
            Scalar::null_native::<u32>(),
            &mut ctx,
        )?
        .into_array();
        let dict = DictArray::try_new(codes, values())?.into_array();

        let optimized = dict.optimize_ctx(&*SESSION)?;

        assert!(optimized.is::<Sparse>(), "{}", optimized.display_tree());
        assert_arrays_eq!(
            optimized,
            VarBinViewArray::from_iter_nullable_str([
                None,
                None,
                None,
                Some("beta"),
                None,
                None,
                None,
                Some("gamma"),
                None,
                None,
            ]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn sliced_null_fill_codes_become_sparse() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let codes = Sparse::try_new(
            buffer![1u64, 4, 8].into_array(),
            PrimitiveArray::from_option_iter([Some(0u32), Some(2), Some(1)]).into_array(),
            10,
            Scalar::null_native::<u32>(),
            &mut ctx,
        )?;
        let patches = codes
            .patches()
            .slice(2..9, &mut ctx)?
            .vortex_expect("range holds patches");
        assert_ne!(patches.offset(), 0);
        let codes =
            Sparse::try_new_from_patches(patches, Scalar::null_native::<u32>())?.into_array();
        let dict = DictArray::try_new(codes, values())?.into_array();

        let optimized = dict.optimize_ctx(&*SESSION)?;

        assert!(optimized.is::<Sparse>(), "{}", optimized.display_tree());
        assert_arrays_eq!(
            optimized,
            VarBinViewArray::from_iter_nullable_str([
                None,
                None,
                Some("gamma"),
                None,
                None,
                None,
                Some("beta"),
            ]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn non_null_fill_codes_stay_dict() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let codes = Sparse::try_new(
            buffer![1u64].into_array(),
            PrimitiveArray::from_iter([2u32]).into_array(),
            4,
            Scalar::from(0u32),
            &mut ctx,
        )?
        .into_array();
        let dict = DictArray::try_new(codes, values())?.into_array();

        let optimized = dict.optimize_ctx(&*SESSION)?;

        assert!(optimized.is::<Dict>(), "{}", optimized.display_tree());
        assert_arrays_eq!(
            optimized,
            VarBinViewArray::from_iter_str(["alpha", "gamma", "alpha", "alpha"]),
            &mut ctx
        );
        Ok(())
    }
}
