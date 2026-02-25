// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar function vtable machinery.
//!
//! This module contains the [`ScalarFnVTable`] trait and all built-in scalar function
//! implementations. Expressions ([`crate::expr::Expression`]) reference scalar functions
//! at each node.

use arcref::ArcRef;

mod bound;
pub(crate) mod fns;
pub(crate) mod options;
mod plugin;
pub mod session;
pub(crate) mod signature;
mod vtable;

pub use bound::*;
pub use fns::*;
pub use options::*;
pub use plugin::*;
pub use signature::*;
pub use vtable::*;

/// A unique identifier for a scalar function.
pub type ScalarFnId = ArcRef<str>;
