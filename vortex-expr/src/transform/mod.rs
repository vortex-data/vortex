//! A collection of transformations that can be applied to a [`crate::ExprRef`].
pub mod partition;
pub(crate) mod remove_select;
pub mod simplify;
