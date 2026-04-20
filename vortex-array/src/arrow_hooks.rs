// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Extension points for the Arrow-based compute kernels.
//!
//! `vortex-array` does not depend on any Arrow crate. When an Arrow-based fallback is
//! needed - for example, the default implementations of `numeric`, `compare`, `boolean`,
//! `like`, `zip`, and Arrow-backed filtering on VarBinView - the implementation is
//! provided by the `vortex-arrow` crate, which registers an [`ArrowCompute`] via
//! [`inventory`] during static initialisation.
//!
//! If `vortex-arrow` is not linked into the binary, these Arrow fallback code paths
//! return an error explaining that the user needs to add `vortex-arrow` as a dependency.

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::MaskValues;

use crate::ArrayRef;
use crate::aliases::inventory;
use crate::arrays::VarBinViewArray;
use crate::scalar::NumericOperator;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::like::LikeOptions;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::Operator;

/// Function pointers for Arrow-backed compute fallbacks.
///
/// The canonical implementation lives in the `vortex-arrow` crate. Simply adding
/// `vortex-arrow` as a dependency is enough to register it - `vortex-arrow`
/// submits a registration via `inventory::submit!` during static initialisation.
pub struct ArrowCompute {
    /// Pointwise comparison of two arrays, returning a boolean array.
    pub compare:
        fn(&ArrayRef, &ArrayRef, CompareOperator) -> VortexResult<ArrayRef>,
    /// Pointwise numeric operation of two arrays.
    pub numeric: fn(&ArrayRef, &ArrayRef, NumericOperator) -> VortexResult<ArrayRef>,
    /// Pointwise Kleene-logical boolean operation between two boolean arrays.
    pub boolean: fn(&ArrayRef, &ArrayRef, Operator) -> VortexResult<ArrayRef>,
    /// SQL LIKE-pattern matching of two arrays.
    pub like: fn(&ArrayRef, &ArrayRef, LikeOptions) -> VortexResult<ArrayRef>,
    /// Zip two arrays together using a boolean condition array.
    pub zip: fn(&ArrayRef, &ArrayRef, &ArrayRef) -> VortexResult<ArrayRef>,
    /// Filter a `VarBinViewArray` using a dense mask.
    pub filter_varbinview:
        fn(&VarBinViewArray, &Arc<MaskValues>) -> VortexResult<VarBinViewArray>,
    /// Compare a varbin array with a constant scalar.
    pub varbin_compare_with_const:
        fn(&ArrayRef, &Scalar, CompareOperator) -> VortexResult<ArrayRef>,
}

/// Inventory registration for an [`ArrowCompute`].
///
/// Submitted from `vortex-arrow` so `inventory::iter::<ArrowComputeRegistration>()`
/// yields the single available implementation.
pub struct ArrowComputeRegistration(pub ArrowCompute);

inventory::collect!(ArrowComputeRegistration);

/// Return the Arrow-backed compute implementation, or an error if `vortex-arrow`
/// is not linked in.
pub fn arrow_compute() -> VortexResult<&'static ArrowCompute> {
    match inventory::iter::<ArrowComputeRegistration>()
        .into_iter()
        .next()
    {
        Some(reg) => Ok(&reg.0),
        None => vortex_bail!(
            "No Arrow compute implementation has been registered. Add `vortex-arrow` as a \
             dependency to enable Arrow-backed compute fallbacks."
        ),
    }
}
