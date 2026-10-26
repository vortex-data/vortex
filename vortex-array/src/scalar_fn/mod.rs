// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar function vtable machinery.
//!
//! This module contains the [`ScalarFnVTable`] trait, the machinery to bind, reduce and execute
//! scalar functions, and in [`builtin`] the few functions vortex-array itself is built on. Other scalar
//! functions live in the `vortex-functions` crate. Expressions ([`crate::expr::Expression`])
//! reference scalar functions at each node.
//!
//! Strict functions with row-at-a-time kernels can implement `unstable::row::RowFn`. It handles
//! decoding, constants, null propagation, output construction, and validity. This API requires the
//! `unstable_row_fns` feature and has no compatibility guarantees. Implement [`ScalarFnVTable`]
//! directly for columnar kernels and functions that alias an input or can produce null from valid
//! inputs.

use vortex_session::registry::Id;

mod vtable;
pub use vtable::*;

mod plugin;
pub use plugin::*;

mod foreign;
pub use foreign::*;

mod typed;
pub use typed::*;

mod erased;
pub use erased::*;

mod options;
pub use options::*;

mod signature;
pub use signature::*;

#[cfg(feature = "unstable_row_fns")]
pub mod unstable;
#[cfg(not(feature = "unstable_row_fns"))]
#[allow(dead_code, unused_imports)]
pub(crate) mod unstable;

pub mod builtin;
pub mod internal;
pub mod session;

/// A unique identifier for a scalar function.
pub type ScalarFnId = Id;

/// Private module to seal [`typed::DynScalarFn`].
mod sealed {
    use crate::scalar_fn::ScalarFnVTable;
    use crate::scalar_fn::typed::TypedScalarFnInstance;

    /// Marker trait to prevent external implementations of [`super::typed::DynScalarFn`].
    pub(crate) trait Sealed {}

    /// This can be the **only** implementor for [`super::typed::DynScalarFn`].
    impl<V: ScalarFnVTable> Sealed for TypedScalarFnInstance<V> {}
}
