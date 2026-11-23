// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Patched;
use crate::arrays::ScalarFn;
use crate::arrays::ScalarFnArray;
use crate::arrays::Slice;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::arrays::scalar_fn::AnyScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar_fn::builtin::binary::Binary;
use crate::scalar_fn::builtin::cast::Cast;
use crate::scalar_fn::builtin::cast::CastKernel;
use crate::scalar_fn::builtin::cast::CastReduce;
use crate::scalar_fn::builtin::cast::CastReduceAdaptor;
use crate::scalar_fn::builtin::pack::Pack;

pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Slice, Patched, const { &SliceReduceAdaptor(Patched) });
    kernels.register_reduce_parent_rule(Cast, Patched, const { &CastReduceAdaptor(Patched) });
    kernels.register_reduce_parent_rule(
        AnyScalarFn,
        Patched,
        const { &PatchedScalarFnPushDownRule },
    );
}

/// Push an element-wise scalar function down into the inner array and the patch values.
///
/// Every row of a patched array is either a row of the inner array or a patch value, so applying
/// an element-wise function to both children and patching the results is equivalent to applying
/// the function to the patched array. Comparisons against a constant, for example, run over the
/// encoded inner array and the handful of patch values, and their boolean result keeps the same
/// patch layout.
///
/// The inner array holds filler values at patched positions, and the patch values of a sliced
/// array may include patches outside the slice. Neither is visible in the result, so only
/// infallible functions are pushed down: a fallible one could fail on an invisible value.
#[derive(Debug)]
struct PatchedScalarFnPushDownRule;

impl ArrayParentReduceRule<Patched> for PatchedScalarFnPushDownRule {
    type Parent = AnyScalarFn;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Patched>,
        parent: ArrayView<'_, ScalarFn>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let scalar_fn = parent.scalar_fn();

        // Preserve pack expressions so exporters can unpack them later, and leave casts to the
        // cast rule.
        if scalar_fn.is::<Pack>() || scalar_fn.is::<Cast>() {
            return Ok(None);
        }
        if !scalar_fn.signature().is_infallible() {
            return Ok(None);
        }

        let other_children_are_constant = parent
            .iter_children()
            .enumerate()
            .all(|(idx, child)| idx == child_idx || child.is::<Constant>());
        if !other_children_are_constant {
            return Ok(None);
        }

        let apply = |child: &ArrayRef| -> VortexResult<ArrayRef> {
            let len = child.len();
            let children = parent
                .iter_children()
                .enumerate()
                .map(|(idx, other)| {
                    if idx == child_idx {
                        child.clone()
                    } else {
                        let scalar = other.as_::<Constant>().scalar().clone();
                        ConstantArray::new(scalar, len).into_array()
                    }
                })
                .collect::<Vec<_>>();
            Ok(ScalarFnArray::try_new(scalar_fn.clone(), children)?.into_array())
        };

        // Comparisons are left to the compare kernel, which evaluates them in one step.
        if scalar_fn
            .as_opt::<Binary>()
            .is_some_and(|op| op.to_compare_operator().is_some())
        {
            return Ok(None);
        }

        let inner = apply(array.inner())?;
        let values = apply(array.patch_values())?;
        let dtype = parent.dtype().clone();
        if inner.dtype() != &dtype || values.dtype() != &dtype {
            return Ok(None);
        }

        // SAFETY: the function preserves the lengths of the children, and both results have the
        // parent's dtype.
        let patched = unsafe { Patched::with_inner_and_values(&array, dtype, inner, values) };
        Ok(Some(patched.into_array()))
    }
}

impl CastReduce for Patched {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // A cast can fail on the filler values the inner array holds at patched positions, or on
        // patches outside a sliced array, so only casts that cannot fail are pushed down here.
        if !casts_infallibly(array.dtype(), dtype) {
            return Ok(None);
        }

        let inner = array.inner().cast(dtype.clone())?;
        let values = array.patch_values().cast(dtype.clone())?;

        // SAFETY: casts preserve length, and both children now have the target dtype.
        let patched =
            unsafe { Patched::with_inner_and_values(&array, dtype.clone(), inner, values) };
        Ok(Some(patched.into_array()))
    }
}

impl CastKernel for Patched {
    /// Pushes a fallible cast down over a constant inner array, as for a sparse array.
    ///
    /// The cast is pushed down only when every stored patch is visible, so that it fails exactly
    /// when casting the array row by row would.
    fn cast(
        array: ArrayView<'_, Self>,
        dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(constant) = array.inner().as_opt::<Constant>() else {
            return Ok(None);
        };
        let layout = array.execute_layout(ctx)?;
        let layout = layout.layout()?;
        if layout.all_patches().len() != array.patch_values().len()
            || layout.has_out_of_bounds_patches()
        {
            return Ok(None);
        }
        // The fill value may be hidden by patches at every position, so a failure to cast it is
        // left to the row-by-row cast to report.
        let Ok(fill) = constant.scalar().cast(dtype) else {
            return Ok(None);
        };

        let inner = ConstantArray::new(fill, array.len()).into_array();
        let values = array.patch_values().cast(dtype.clone())?;
        // SAFETY: casts preserve length, and both children now have the target dtype.
        let patched =
            unsafe { Patched::with_inner_and_values(&array, dtype.clone(), inner, values) };
        Ok(Some(patched.into_array()))
    }
}

/// Returns `true` if every value of `from` can be cast to `to`.
fn casts_infallibly(from: &DType, to: &DType) -> bool {
    // Removing nullability fails on nulls.
    if from.is_nullable() && !to.is_nullable() {
        return false;
    }
    if from.eq_ignore_nullability(to) {
        return true;
    }
    match (from, to) {
        (DType::Primitive(from, _), DType::Primitive(to, _)) => from.casts_losslessly_to(*to),
        _ => false,
    }
}
