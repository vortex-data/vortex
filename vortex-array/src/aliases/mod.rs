//! Re-exports of third-party crates we use in the API.
//!
//! The HashMap/Set should be preferred over the standard library variants or other alternatives.
//! Currently defers to the excellent [hashbrown](https://docs.rs/hashbrown/latest/hashbrown/) crate.

pub mod hash_map;
pub mod hash_set;

pub use hashbrown::DefaultHashBuilder;

pub mod paste {
    //! Re-export of [`paste`](https://docs.rs/paste/latest/paste/).
    pub use paste::paste;
}

// Re-export of [`inventory`](https://docs.rs/inventory/latest/inventory/).
pub use inventory;
