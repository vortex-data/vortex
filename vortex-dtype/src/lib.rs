#![cfg(target_endian = "little")]
#![deny(missing_docs)]

//! A type system for Vortex
//!
//! This crate contains the core logical type system for Vortex, including the definition of data types,
//! and (optionally) logic for their serialization and deserialization.

pub use dtype::*;
pub use extension::*;
pub use half;
pub use nullability::*;
pub use ptype::*;

#[cfg(feature = "arbitrary")]
mod arbitrary;
mod dtype;
mod extension;
pub mod field;
mod nullability;
mod ptype;
mod serde;

#[cfg(feature = "proto")]
pub mod proto {
    //! Protocol buffer representations for DTypes
    //!
    //! This module contains the code to serialize and deserialize DTypes to and from protocol buffers.

    pub use vortex_proto::dtype;
}

#[cfg(feature = "flatbuffers")]
pub mod flatbuffers {
    //! Flatbuffer representations for DTypes
    //!
    //! This module contains the code to serialize and deserialize DTypes to and from flatbuffers.

    pub use vortex_flatbuffers::dtype::*;

    pub use super::serde::flatbuffers::*;
}
