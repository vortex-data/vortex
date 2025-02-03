//! A collection of transformations that can be applied to a [`crate::ExprRef`].
pub mod field_mask;
pub mod immediate_access;
pub mod partition;
mod remove_merge;
mod remove_select;
pub mod simplify;
pub mod simplify_typed;
