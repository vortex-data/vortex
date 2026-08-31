// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Higher-order function vtable machinery.
//!
//! Higher-order functions accept ordinary arguments and lambda arguments. This module provides
//! their identity, type erasure, per-call options, and session registry.

use vortex_session::registry::Id;

mod erased;
pub use erased::HigherOrderFnRef;

mod options;
pub use options::HigherOrderFnOptions;

mod typed;
pub use typed::TypedHigherOrderFnInstance;

mod plugin;
pub use plugin::*;

pub mod session;

mod vtable;
pub use vtable::*;

/// A globally unique identifier for a higher-order function.
pub type HigherOrderFnId = Id;
