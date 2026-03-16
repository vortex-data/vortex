// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar function vtable machinery.
//!
//! This module contains the [`ScalarFnVTable`] trait and all built-in scalar function
//! implementations. Expressions ([`crate::expr::Expression`]) reference scalar functions
//! at each node.

use arcref::ArcRef;

mod vtable;
pub use vtable::*;

mod plugin;
pub use plugin::*;

mod typed;
pub use typed::*;

mod erased;
pub use erased::*;

mod options;
pub use options::*;

mod signature;
pub use signature::*;

pub mod fns;
pub mod session;

/// A unique identifier for a scalar function.
pub type ScalarFnId = ArcRef<str>;

/// Private module to seal [`typed::DynScalarFn`].
mod sealed {
    use crate::scalar_fn::ScalarFnVTable;
    use crate::scalar_fn::typed::ScalarFn;

    /// Marker trait to prevent external implementations of [`super::typed::DynScalarFn`].
    pub(crate) trait Sealed {}

    /// This can be the **only** implementor for [`super::typed::DynScalarFn`].
    impl<V: ScalarFnVTable> Sealed for ScalarFn<V> {}
}
