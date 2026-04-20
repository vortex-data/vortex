// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Extension points for the Arrow-based compute kernels.
//!
//! `vortex-array` does not depend on any Arrow crate. When an Arrow-based
//! fallback is needed - for example, the default implementations of `numeric`,
//! `compare`, `boolean`, `like`, `zip`, and Arrow-backed filtering on
//! `VarBinView` - the implementation is provided by the `vortex-arrow` crate,
//! which calls [`register_arrow_compute`] during its `init()` function.
//!
//! To sidestep Cargo's diamond-dependency behaviour (which can produce two
//! independently-linked copies of `vortex-array`'s statics during
//! `cargo test`), the actual storage slot lives in the tiny [`vortex-arrow-hook`]
//! crate, which is compiled exactly once and therefore has a single shared
//! `AtomicPtr`.

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::MaskValues;

use crate::ArrayRef;
use crate::arrays::VarBinViewArray;
use crate::scalar::NumericOperator;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::like::LikeOptions;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::Operator;

/// Function pointers for Arrow-backed compute fallbacks.
///
/// The canonical implementation lives in the `vortex-arrow` crate.
pub struct ArrowCompute {
    /// Pointwise comparison of two arrays, returning a boolean array.
    pub compare: fn(&ArrayRef, &ArrayRef, CompareOperator) -> VortexResult<ArrayRef>,
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

/// Install an [`ArrowCompute`] implementation globally. Subsequent calls are
/// ignored - the first installer wins.
pub fn register_arrow_compute(compute: ArrowCompute) {
    let leaked: *const ArrowCompute = Box::leak(Box::new(compute));
    if !vortex_arrow_hook::set(leaked as *const ()) {
        // Someone else already installed a compute; drop ours.
        // SAFETY: we just leaked this box and no one else has seen this pointer.
        unsafe { drop(Box::from_raw(leaked as *mut ArrowCompute)) };
    }
}

/// Return the registered Arrow-backed compute implementation, or an error if
/// none is registered (typically because `vortex-arrow` is not linked).
pub fn arrow_compute() -> VortexResult<&'static ArrowCompute> {
    let ptr = vortex_arrow_hook::get() as *const ArrowCompute;
    if ptr.is_null() {
        vortex_bail!(
            "No Arrow compute implementation has been registered. Add `vortex-arrow` \
             as a dependency and ensure its `init()` runs (done automatically on \
             first use) to enable Arrow-backed compute fallbacks."
        );
    }
    // SAFETY: once set, the pointer is leaked for the life of the process.
    Ok(unsafe { &*ptr })
}
