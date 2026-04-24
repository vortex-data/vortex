// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Aggregate function vtable machinery.
//!
//! This module contains the [`AggregateFnVTable`] trait, the [`Accumulator`] trait, and the
//! type-erasure infrastructure for aggregate functions.

use vortex_session::registry::Id;

mod accumulator;
pub use accumulator::*;

mod accumulator_grouped;
pub use accumulator_grouped::*;

mod vtable;
pub use vtable::*;

mod plugin;
pub use plugin::*;

mod foreign;
pub(crate) use foreign::*;

mod typed;
pub use typed::*;

mod erased;
pub use erased::*;

mod options;
pub use options::*;

pub mod fns;
pub mod kernels;
pub mod proto;
pub mod session;

/// A unique identifier for an aggregate function.
pub type AggregateFnId = Id;

/// Private module to seal [`typed::DynAggregateFn`].
mod sealed {
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::typed::AggregateFnInner;

    /// Marker trait to prevent external implementations of [`super::typed::DynAggregateFn`].
    pub(crate) trait Sealed {}

    /// This can be the **only** implementor for [`super::typed::DynAggregateFn`].
    impl<V: AggregateFnVTable> Sealed for AggregateFnInner<V> {}
}
