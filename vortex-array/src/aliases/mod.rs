//! Aliases for fast HashMap and HashSet implementations.
//!
//! Should be preferred over the standard library variants or other alternatives.
//! Currently defers to the excellent [hashbrown](https://docs.rs/hashbrown/latest/hashbrown/) crate.

pub mod hash_map;
pub mod hash_set;

pub use hashbrown::DefaultHashBuilder;
