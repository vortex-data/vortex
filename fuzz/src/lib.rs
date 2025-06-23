#![feature(error_generic_member_access)]

mod array;
pub mod error;
mod file;

pub use array::{Action, ExpectedValue, FuzzArrayAction, sort_canonical_array};
pub use file::FuzzFileAction;
