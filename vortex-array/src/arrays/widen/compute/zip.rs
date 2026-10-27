// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::ScalarFn;
use crate::arrays::Widen;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::compute::cast_keeping_nullability;
use crate::arrays::widen::compute::ptype_holding_both_values;
use crate::arrays::widen::compute::ptype_holding_values_and_scalar;
use crate::arrays::widen::compute::wrap_values;
use crate::builtins::ArrayBuiltins;
use crate::kernel::ExecuteParentKernel;
use crate::scalar_fn::fns::zip::Zip;
use crate::scalar_fn::fns::zip::ZipKernel;

impl ZipKernel for Widen {
    fn zip(
        array: ArrayView<'_, Self>,
        if_false: &ArrayRef,
        mask: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        zip_values(array, if_false, mask, Branch::IfTrue)
    }
}

/// Zips a [`Widen`] in the `if_false` branch.
///
/// [`ZipExecuteAdaptor`] runs [`ZipKernel`] only for the `if_true` branch.
///
/// [`ZipExecuteAdaptor`]: crate::scalar_fn::fns::zip::ZipExecuteAdaptor
#[derive(Debug)]
pub(crate) struct WidenZipIfFalseKernel;

impl ExecuteParentKernel<Widen> for WidenZipIfFalseKernel {
    type Parent = ExactScalarFn<Zip>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, Widen>,
        parent: ScalarFnArrayView<'_, Zip>,
        child_idx: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 1 {
            return Ok(None);
        }

        let zip = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");

        zip_values(array, zip.get_child(0), zip.get_child(2), Branch::IfFalse)
    }
}

/// The branch of a zip that holds the [`Widen`].
enum Branch {
    IfTrue,
    IfFalse,
}

/// Zips the values of `array` with the values of `other`, a `Widen` or a constant, at the type
/// that holds both. Any other `other` declines.
fn zip_values(
    array: ArrayView<'_, Widen>,
    other: &ArrayRef,
    mask: &ArrayRef,
    branch: Branch,
) -> VortexResult<Option<ArrayRef>> {
    let values = array.values();
    let (other_values, ptype) = if let Some(other) = other.as_opt::<Widen>() {
        let Some(ptype) = ptype_holding_both_values(array, other) else {
            return Ok(None);
        };
        (other.values().clone(), ptype)
    } else if let Some(constant) = other.as_constant() {
        let Some(ptype) = ptype_holding_values_and_scalar(array, constant)? else {
            return Ok(None);
        };
        (other.clone(), ptype)
    } else {
        return Ok(None);
    };

    let values = cast_keeping_nullability(values, ptype)?;
    let other_values = cast_keeping_nullability(&other_values, ptype)?;

    let zipped = match branch {
        Branch::IfTrue => mask.zip(values, other_values)?,
        Branch::IfFalse => mask.zip(other_values, values)?,
    };
    let dtype = array.dtype().with_nullability(zipped.dtype().nullability());

    wrap_values(zipped, dtype).map(Some)
}
