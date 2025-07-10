// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A collection of transformations that can be applied to a [`crate::ExprRef`].
pub mod annotations;
pub mod field_mask;
pub mod immediate_access;
pub(crate) mod match_between;
pub mod partition;
mod remove_merge;
mod remove_select;
pub mod replace;
pub mod simplify;
pub mod simplify_typed;
