//! A collection of transformations that can be applied to a [`crate::ExprRef`].
mod immediate_access;
pub mod field_mask;
pub mod partition;
pub(crate) mod remove_select;
pub mod simplify;
pub mod simplify_typed;
