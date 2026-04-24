// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::dtype::DType;
use crate::scalar_fn::ScalarFnRef;

/// An argument that may be able to peel one refinement layer and retry scalar-fn dtype
/// resolution on its storage dtype.
pub(crate) trait RefinementFallbackArg {
    fn current_dtype(&self) -> &DType;

    fn peel_one_refinement_layer(&mut self) -> bool;
}

/// Resolve a scalar function's return dtype, retrying after peeling one refinement layer from all
/// currently peelable children when the original argument dtypes are rejected.
pub(crate) fn resolve_return_dtype_with_refinement_fallback<A: RefinementFallbackArg>(
    scalar_fn: &ScalarFnRef,
    args: &mut [A],
) -> VortexResult<DType> {
    loop {
        let arg_dtypes: Vec<_> = args.iter().map(|arg| arg.current_dtype().clone()).collect();

        match scalar_fn.return_dtype(&arg_dtypes) {
            Ok(dtype) => return Ok(dtype),
            Err(err) => {
                let mut any_peeled = false;
                for arg in args.iter_mut() {
                    any_peeled |= arg.peel_one_refinement_layer();
                }
                if !any_peeled {
                    return Err(err);
                }
            }
        }
    }
}
