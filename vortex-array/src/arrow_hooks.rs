// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Extension point for the Arrow-based compute kernels.
//!
//! `vortex-array` does not depend on any Arrow crate. When Arrow-based behaviour is
//! required (for example, the default implementations of `numeric`, `compare`, `boolean`,
//! `like`, `zip`, and Arrow-backed filtering), it is provided by the `vortex-arrow`
//! crate, which registers an implementation of [`ArrowCompute`] at program start-up
//! via [`register_arrow_compute`].
//!
//! If `vortex-arrow` is not linked into the binary, the Arrow fallback code paths will
//! return an error explaining that the user needs to add `vortex-arrow` as a dependency.

use std::sync::Arc;
use std::sync::OnceLock;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::MaskValues;

use crate::ArrayRef;
use crate::arrays::VarBinViewArray;
use crate::scalar::NumericOperator;
use crate::scalar_fn::fns::binary::LikeOptions;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::Operator;

/// Arrow-backed implementations of compute kernels that `vortex-array` falls back to
/// when it does not have a specialised native path.
///
/// The canonical implementation lives in the `vortex-arrow` crate. Simply adding
/// `vortex-arrow` as a dependency is enough to register it - `vortex-arrow` calls
/// [`register_arrow_compute`] during its static initialisation.
pub trait ArrowCompute: Send + Sync + 'static {
    /// Pointwise comparison of two arrays, returning a boolean array.
    fn compare(
        &self,
        lhs: &ArrayRef,
        rhs: &ArrayRef,
        op: CompareOperator,
    ) -> VortexResult<ArrayRef>;

    /// Pointwise numeric operation of two arrays.
    fn numeric(
        &self,
        lhs: &ArrayRef,
        rhs: &ArrayRef,
        op: NumericOperator,
    ) -> VortexResult<ArrayRef>;

    /// Pointwise Kleene-logical boolean operation between two boolean arrays.
    fn boolean(&self, lhs: &ArrayRef, rhs: &ArrayRef, op: Operator) -> VortexResult<ArrayRef>;

    /// SQL LIKE-pattern matching of two arrays.
    fn like(
        &self,
        array: &ArrayRef,
        pattern: &ArrayRef,
        options: LikeOptions,
    ) -> VortexResult<ArrayRef>;

    /// Zip two arrays together using a boolean condition array.
    fn zip(
        &self,
        condition: &ArrayRef,
        lhs: &ArrayRef,
        rhs: &ArrayRef,
    ) -> VortexResult<ArrayRef>;

    /// Filter a `VarBinViewArray` using a dense mask.
    fn filter_varbinview(
        &self,
        array: &VarBinViewArray,
        mask: &Arc<MaskValues>,
    ) -> VortexResult<VarBinViewArray>;

    /// Compare a varbin array with a constant scalar on the RHS.
    fn varbin_compare_with_const(
        &self,
        lhs: &ArrayRef,
        rhs_const: &crate::arrays::ConstantArray,
        op: CompareOperator,
    ) -> VortexResult<ArrayRef>;
}

static ARROW_COMPUTE: OnceLock<&'static dyn ArrowCompute> = OnceLock::new();

/// Register an implementation of [`ArrowCompute`] globally.
///
/// Typically called from `vortex-arrow`'s initialisation code. Calling this multiple
/// times is harmless - only the first call is honoured.
pub fn register_arrow_compute(impl_: &'static dyn ArrowCompute) {
    let _ = ARROW_COMPUTE.set(impl_);
}

/// Return the registered Arrow compute implementation, or an error if none is registered.
pub fn arrow_compute() -> VortexResult<&'static dyn ArrowCompute> {
    match ARROW_COMPUTE.get() {
        Some(c) => Ok(*c),
        None => vortex_bail!(
            "No Arrow compute implementation has been registered. Add `vortex-arrow` as a \
             dependency to enable Arrow-backed compute fallbacks."
        ),
    }
}
