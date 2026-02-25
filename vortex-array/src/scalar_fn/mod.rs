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

pub(crate) mod typed;

mod erased;
pub use erased::*;

pub(crate) mod options;
pub use options::*;

pub(crate) mod signature;
pub use signature::*;

pub mod session;

pub(crate) mod fns;
pub use fns::*;

/// A unique identifier for a scalar function.
pub type ScalarFnId = ArcRef<str>;

/// Private module to seal [`typed::DynScalarFn`].
mod sealed {
    use crate::scalar_fn::ScalarFnVTable;
    use crate::scalar_fn::typed::ScalarFnInner;

    /// Marker trait to prevent external implementations of [`super::typed::DynScalarFn`].
    pub(crate) trait Sealed {}

    /// This can be the **only** implementor for [`super::typed::DynScalarFn`].
    impl<V: ScalarFnVTable> Sealed for ScalarFnInner<V> {}
}
