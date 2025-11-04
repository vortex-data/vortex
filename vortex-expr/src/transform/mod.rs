// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A collection of transformations that can be applied to a [`crate::Expression`].
pub mod annotations;
pub mod immediate_access;
pub(crate) mod match_between;
mod partition;
mod remove_merge;
mod remove_select;
mod replace;
mod simplify;
mod simplify_typed;

pub use partition::*;
pub use replace::*;
pub use simplify::*;
pub use simplify_typed::*;
