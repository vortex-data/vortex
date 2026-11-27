// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Kernels that push [`VariantGet`] through the masks that wrap a Variant read from a file.
//!
//! The kernels that extract in bulk are keyed on the Variant encodings. A masked Variant reaches
//! `variant_get` without one, so these kernels move the extraction to the array that the mask
//! wraps. A mask only nulls rows out, so it commutes with extraction. `vortex-array` already
//! splits a unary function such as `variant_get` over the chunks of a chunked array.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Masked;
use vortex_array::arrays::ScalarFn;
use vortex_array::arrays::masked::MaskedArrayExt;
use vortex_array::arrays::masked::MaskedArraySlotsExt;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::optimizer::kernels::ExecuteParentFn;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::builtin::mask::Mask;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::variant_get::VariantGet;
use crate::variant_get::VariantGetOptions;

pub(super) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(VariantGet.id(), Masked, VariantGetMaskedKernel);
    // The array of a scalar function has the function's ID as its encoding ID.
    kernels.register_execute_parent(
        VariantGet.id(),
        Mask.id(),
        &[execute_variant_get_of_mask as ExecuteParentFn],
    );
}

/// Extracts from the masked Variant and masks the result.
#[derive(Default, Debug)]
struct VariantGetMaskedKernel;

impl ExecuteParentKernel<Masked> for VariantGetMaskedKernel {
    type Parent = ExactScalarFn<VariantGet>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, Masked>,
        parent: ScalarFnArrayView<'_, VariantGet>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }

        let mask = array.masked_validity().to_array(array.len());
        execute_variant_get(array.child().clone(), parent.options, ctx)?
            .mask(mask)?
            .execute::<ArrayRef>(ctx)
            .map(Some)
    }
}

/// Extracts from the input of the `mask` function and masks the result, as
/// [`VariantGetMaskedKernel`] does for a masked array.
fn execute_variant_get_of_mask(
    child: &ArrayRef,
    parent: &ArrayRef,
    child_idx: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let (0, Some(parent), Some(mask)) = (
        child_idx,
        parent.as_opt::<ExactScalarFn<VariantGet>>(),
        child.as_opt::<ScalarFn>(),
    ) else {
        return Ok(None);
    };
    if mask.scalar_fn().id() != Mask.id() {
        return Ok(None);
    }

    execute_variant_get(mask.child_at(0).clone(), parent.options, ctx)?
        .mask(mask.child_at(1).clone())?
        .execute::<ArrayRef>(ctx)
        .map(Some)
}

fn execute_variant_get(
    input: ArrayRef,
    options: &VariantGetOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    VariantGet::try_new(input, options.clone())?
        .into_array()
        .execute::<ArrayRef>(ctx)
}
